"""robots.txt parser and checker for polite scraping.

Public API
----------
    check_robots(url, session, user_agent="*", cache=…) → bool
    is_allowed(url, user_agent="*", cache=…) → bool
    clear_cache() → None
    RobotsCache(ttl=…, user_agent=…) → owned cache, for per-crawl isolation

:func:`check_robots` and :func:`is_allowed` both default to a process-wide
cache, so a caller can ask a question without owning anything. Both take the same
optional ``cache=``, and a crawler that runs more than one independent crawl
should own a :class:`RobotsCache` per run and pass it to *both* — then the write
and the read cannot land in different caches, and one run can never inherit
another's policies.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp

from .config import DEFAULT_TIMEOUT

__all__ = ["RobotsCache", "check_robots", "clear_cache", "is_allowed"]

#: How long a fetched policy stays authoritative before it is re-fetched.
#:
#: A robots.txt can be tightened while a crawl is running, and a crawl can easily
#: outlive the copy it read at the start. One hour keeps the common case at one
#: fetch per host while bounding how stale the enforced policy can get; a run
#: that wants the old "never re-read" behaviour passes a very large ``ttl``.
_DEFAULT_TTL = 3600.0


def _base_of(url: str) -> str:
    """Return the ``scheme://netloc`` key identifying a host's policy."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


#: What this module announces itself as when the caller has declared no identity.
#:
#: RFC 9309 §2.2.1: a crawler whose product token matches no group is governed by
#: the ``User-agent: *`` group, and ``*`` is the token the spec reserves for "any
#: other crawler". Sending it asks the site exactly the question the site
#: reserves for an unidentified client — and, unlike sending nothing, it is never
#: quietly answered by aiohttp's own ``Python/3.x aiohttp/3.y``, which is how the
#: sitemap path used to announce itself.
_WILDCARD_IDENTITY = "*"


class _Policy(RobotFileParser):
    """
    A robots policy with an explicit, typed allow-everything flag.

    ``RobotFileParser.allow_all`` is an undocumented implementation detail. It
    *is* initialised in ``__init__`` and *is* read by ``can_fetch``, but it is
    absent from the type stubs, so assigning it costs a ``type: ignore`` and its
    meaning is only guaranteed by CPython's internals. Declaring the field here
    pins the behaviour to this module: ``allow_all`` is a real attribute of our
    own class on every supported Python version, no suppression needed.
    """

    allow_all: bool = False

    @classmethod
    def allow_everything(cls) -> _Policy:
        """Build a policy that permits every URL (there is no policy to enforce)."""
        policy = cls()
        policy.allow_all = True
        return policy


async def _load_robots(
    base: str,
    session: aiohttp.ClientSession,
    user_agent: str = _WILDCARD_IDENTITY,
) -> RobotFileParser | None:
    """
    Fetch and parse robots.txt for *base*. Returns ``None`` if the fetch failed.

    Deliberately knows nothing about caching: its only job is to decide whether
    the site gave us an answer *at all*. That distinction — "allow everything"
    versus "we could not ask" — is the one the caller must not blur.

    *user_agent* is sent as the ``User-Agent`` header, because a site may answer
    differently depending on who is asking — Google serves a crawler-specific
    robots.txt, and plenty of sites publish a ``User-agent: <browser>`` group at
    all. Asking without it means the body parsed here is the body some *other*
    identity was served, which is the mirror image of evaluating one identity
    while transmitting another. The default is the RFC 9309 "no identity declared"
    token rather than whatever the session would otherwise have sent: an inherited
    default is an identity nobody in this call stack chose.
    """
    robots_url = urljoin(base, "/robots.txt")
    try:
        timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
        async with session.get(
            robots_url, timeout=timeout, headers={"User-Agent": user_agent}
        ) as resp:
            if resp.status == 200:
                policy = _Policy()
                policy.parse((await resp.text()).splitlines())
                return policy
            if 400 <= resp.status < 500:
                # 4xx is how a site says "there is no robots.txt here" (RFC 9309
                # treats an unavailable file as no restrictions). That is a real
                # answer, so the caller is allowed to remember it.
                return _Policy.allow_everything()
            # 5xx is the site failing, not declining to publish a policy. Reporting
            # that as "allow everything" lets one bad response wave a whole run
            # through, so hand back "unknown" and let the next page re-ask.
            return None
    except Exception:
        # DNS failure, connection reset, timeout, a body that is not decodable
        # text: none of these say anything about the site's rules.
        return None


def _discard_outcome(task: asyncio.Task[RobotFileParser | None]) -> None:
    """
    Mark a finished fetch's exception as retrieved, then drop it.

    ``Task.exception()`` is what clears the flag behind asyncio's GC-time "Task
    exception was never retrieved" report. ``asyncio.shield`` already retrieves
    the outcome for the callers it shields, and a *cancelled* task is never
    reported, so this callback is insurance rather than the mechanism: it makes
    "a dropped fetch never reports a stray exception" a property of this module
    instead of a property of a stdlib internal. ``task.cancelled()`` is checked
    first because ``exception()`` re-raises ``CancelledError``.
    """
    if not task.cancelled():
        task.exception()


@dataclass(frozen=True)
class _Entry:
    """A cached policy and the monotonic time it was fetched."""

    policy: RobotFileParser
    fetched_at: float


class RobotsCache:
    """
    robots.txt policies, fetched at most once per concurrent burst per host.

    An engine should own one of these per crawl. Owning it is what makes the
    cache testable: a test constructs its own instance instead of reaching into
    module globals to reset state that a previous test left behind.

    *user_agent* is the identity this cache announces itself with when it fetches
    a policy, and it defaults to the RFC 9309 "no identity declared" token. Set it
    when the crawl has an identity to declare — the point being that it is stated
    once, here, rather than inherited from whatever headers the session happens to
    carry. Nothing in this module reads ``config``: an identity is the caller's to
    choose, and the one thing that must never happen is deciding it twice in two
    places.
    """

    def __init__(self, ttl: float = _DEFAULT_TTL, user_agent: str = _WILDCARD_IDENTITY) -> None:
        self._entries: dict[str, _Entry] = {}
        self._user_agent = user_agent
        # In-flight fetches, so concurrent page checks for one host share a
        # single request. Without this the cache check happened before the await
        # and every concurrent worker missed it: 6 concurrent pages meant 6
        # robots.txt fetches.
        self._inflight: dict[str, asyncio.Task[RobotFileParser | None]] = {}
        self._ttl = ttl

    # ── read path (no I/O) ───────────────────────────────────────────────────

    def lookup(self, url: str) -> RobotFileParser | None:
        """
        Return the policy cached for *url*'s host, or ``None`` if there is none.

        ``None`` means "no policy in hand": nothing fetched yet, the last fetch
        failed, or the entry has expired. A stale entry is dropped rather than
        returned, so the next caller re-fetches instead of enforcing a policy
        from hours ago.
        """
        return self._fresh(_base_of(url))

    def is_allowed(self, url: str, user_agent: str = _WILDCARD_IDENTITY) -> bool:
        """
        Check *url* against the policy already cached for its host. Never fetches.

        Returns True when no policy is cached, because there is nothing to
        enforce. This is a cheap re-check of a URL whose host is already known;
        it does not honour robots.txt on its own. Use :meth:`check` for that.

        ``True`` here is ambiguous between "the site's rules allow it" and "no
        policy in hand", and the two mean opposite things for a caller that is
        about to fetch. Use :meth:`lookup` to tell them apart.
        """
        policy = self.lookup(url)
        if policy is None:
            return True
        return policy.can_fetch(user_agent, url)

    async def sitemaps(self, base: str, session: aiohttp.ClientSession) -> list[str]:
        """
        The ``Sitemap:`` URLs robots.txt advertises, in the order it lists them.

        Read from the cached policy, so asking costs no extra request: the crawl
        already fetched robots.txt to decide what it was allowed to fetch.

        A ``None`` policy means robots.txt could not be read, and that is not an
        answer about sitemaps — it yields none, and the caller falls back to the
        conventional ``/sitemap.xml``.
        """
        policy = await self._policy_for(_base_of(base), session)
        if policy is None:
            return []
        # urllib's site_maps() returns None rather than [] when the file has no
        # Sitemap: lines — the common case for most sites — so list() over it
        # raises. Treat that as "no sitemaps", which is what it means.
        maps = getattr(policy, "site_maps", None)
        if maps is None:
            return []
        listed = maps()
        return list(listed or ())

    # ── enforcing path ──────────────────────────────────────────────────────

    async def check(
        self,
        url: str,
        session: aiohttp.ClientSession,
        user_agent: str = _WILDCARD_IDENTITY,
    ) -> bool:
        """
        Fetch the host's robots.txt if needed and report whether *url* is allowed.

        Returns True when the site's rules allow *url* — and also when robots.txt
        could not be fetched, because then the rules are unknown. The failure is
        not remembered, so the next page tries again; see :meth:`_policy_for`.

        ``user_agent`` is the identity the rules are evaluated against, and it is
        also the identity the fetch is made under: passing a different one to
        :func:`check_robots` and to the page request is what this module's docs
        warn against, so both halves now take the same argument. See
        :func:`check_robots` for why that matters.
        """
        policy = await self._policy_for(_base_of(url), session, user_agent)
        if policy is None:
            return True
        return policy.can_fetch(user_agent, url)

    async def _policy_for(
        self,
        base: str,
        session: aiohttp.ClientSession,
        user_agent: str | None = None,
    ) -> RobotFileParser | None:
        """
        Return the host's policy, collapsing a concurrent burst onto one fetch.

        ``None`` means the fetch failed, and that answer is deliberately *not*
        cached. A transient DNS blip or a single HTTP 500 must not become an
        allow-everything policy for the rest of the run: "we could not ask" is not
        "the site said yes", and recording the former as the latter makes a
        polite scraper impolite for the life of the process. Callers treat
        ``None`` as "allow this one request, learn nothing", so the next page
        re-asks — the burst collapsing onto a single request is what keeps that
        from turning into a storm against a host whose robots.txt is broken.
        """
        policy = self._fresh(base)
        if policy is not None:
            return policy

        task = self._inflight.get(base)
        if task is None:
            # The identity is the caller's, defaulted to the cache's own. It is
            # never read off the session: an inherited session header is an
            # identity nobody here chose, which is the whole defect.
            agent = self._user_agent if user_agent is None else user_agent
            task = asyncio.ensure_future(_load_robots(base, session, agent))
            task.add_done_callback(_discard_outcome)
            self._inflight[base] = task

        try:
            # shield(): one caller giving up (its own timeout, shutdown) must not
            # cancel the shared fetch that the other workers are still waiting on.
            policy = await asyncio.shield(task)
        except Exception:
            # The shared fetch blew up. `_load_robots` already swallows ordinary
            # failures, so this is belt and braces — but it keeps a hostile
            # session double from turning a robots check into a crawl crash.
            return None
        finally:
            # Dropped either way. Leaving a *failed* fetch in place would hand the
            # same exception to every later page instead of letting them retry.
            if self._inflight.get(base) is task:
                del self._inflight[base]

        if policy is None:
            return None

        self._entries[base] = _Entry(policy=policy, fetched_at=time.monotonic())
        return policy

    def _fresh(self, base: str) -> RobotFileParser | None:
        """Return the unexpired policy for *base*, dropping it if it has expired."""
        entry = self._entries.get(base)
        if entry is None:
            return None
        if time.monotonic() - entry.fetched_at >= self._ttl:
            del self._entries[base]
            return None
        return entry.policy

    def clear(self) -> None:
        """
        Forget every cached policy and cancel the fetches still in flight.

        In-flight fetches are cancelled rather than left running so none of them
        can write a policy back into the cache after the reset. A caller already
        awaiting one sees ``CancelledError``; that is the honest signal for "the
        cache went away underneath you".
        """
        self._entries.clear()
        for task in self._inflight.values():
            task.cancel()
        self._inflight.clear()

    def __len__(self) -> int:
        """Number of hosts currently cached (test and debug affordance)."""
        return len(self._entries)


#: Process-wide default cache behind the module-level helpers below.
_default_cache = RobotsCache()


async def check_robots(
    url: str,
    session: aiohttp.ClientSession,
    user_agent: str = _WILDCARD_IDENTITY,
    *,
    cache: RobotsCache | None = None,
) -> bool:
    """
    Fetch robots.txt for *url*'s host and report whether *url* may be fetched.

    Returns True if the site's rules allow *url* — and also if robots.txt could
    not be fetched, because then the rules are unknown. An unknown answer is
    never cached, so the next page asks again.

    ``user_agent`` is the identity the rules are evaluated against, and it is
    also the identity the robots.txt request is sent under — both halves take
    this one argument, so they cannot drift. Evaluating one identity while
    transmitting another asks the site about a policy it never agreed to: urllib
    reduces the argument to the token before the first "/", so the full
    "Mozilla/5.0 … Chrome/124.0.0.0 Safari/537.36" string scores as the identity
    ``mozilla`` — which matches a ``User-agent: Mozilla`` group, never a
    ``User-agent: Googlebot`` one, and leaves ``User-agent: * Disallow: /`` sites
    gated by the wrong group.

    The default ``"*"`` is the RFC 9309 answer for a crawler that has not
    declared an identity: obey the ``User-agent: *`` group and nothing else. A
    caller that rotates a real User-Agent per request should pass the one it is
    about to send — which, for a rotating client, means deciding the agent
    *before* the robots check rather than after it.
    """
    target = cache if cache is not None else _default_cache
    return await target.check(url, session, user_agent)


def is_allowed(
    url: str,
    user_agent: str = _WILDCARD_IDENTITY,
    *,
    cache: RobotsCache | None = None,
) -> bool:
    """
    Check *url* against the policy already cached for its host, without fetching.

    Kept because it is the cheap way to re-check a URL on a host the crawl
    already has a policy for. Returns True when nothing is cached. See
    :meth:`RobotsCache.is_allowed`.

    ``cache`` mirrors :func:`check_robots`, and it has to. These two are the read
    and write halves of one decision, so reading the default cache after writing
    an owned one made "allowed" mean two different things depending on which
    argument was passed — True because the site's rules permitted it, or True
    because a cache this call never consulted had nothing to say. Pass the cache
    the crawl owns and the pair cannot answer differently about the same URL.

    An owned cache is never written into the default: that is the other half of
    the guarantee, and it is why the default is consulted only when no cache is
    named rather than as a fallback.

    ``True`` is still ambiguous on its own — "allowed" versus "nothing cached" are
    the same boolean. :meth:`RobotsCache.lookup` distinguishes them.
    """
    target = cache if cache is not None else _default_cache
    return target.is_allowed(url, user_agent)


def clear_cache() -> None:
    """
    Reset the process-wide default cache and cancel its in-flight fetches.

    Thin wrapper over the default cache: the counterpart to :func:`check_robots`
    using that cache implicitly. Pass an owned :class:`RobotsCache` to
    :func:`check_robots` and call ``.clear()`` on it instead when a run needs its
    own state.
    """
    _default_cache.clear()
