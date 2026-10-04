"""
protor.fetcher
~~~~~~~~~~~~~~
Deep HTTP fetch module. Owns retry, conditional caching, hooks, and User-Agent
rotation, so the rest of the codebase never touches aiohttp session details.

Public API
----------
    fetch(session, url, *, timeout, max_retries, cache, hooks) → FetchResult
    download_file(session, url, dest) → bool
    random_user_agent() → str
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import aiohttp

from .config import (
    DEFAULT_TIMEOUT,
    JS_DOWNLOAD_TIMEOUT,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    RETRYABLE_STATUS,
    USER_AGENTS,
)
from .exceptions import FetchError
from .http_cache import CacheEntry, HTTPCache
from .netguard import describe_block

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = ["FetchResult", "download_file", "fetch", "random_user_agent"]

#: Statuses that mean "the real page is at the Location header".
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

#: Redirect hops allowed before a fetch is called a loop.
MAX_REDIRECTS = 5


@dataclass
class FetchResult:
    """
    A successfully fetched page: its text, byte size, HTTP status, and type.

    ``content_type`` is the response's own ``Content-Type``, kept because a body
    is not a page by virtue of arriving over HTTP: without it a PDF is scraped
    into two thousand characters of ``%PDF-1.4`` and a PNG into binary noise,
    both reported as successfully scraped text.
    """

    text: str
    nbytes: int
    status: int = 200
    content_type: str = ""
    #: True when the server answered 304, so this body is the one already held
    #: rather than a fresh one. Without it a re-crawl of an unchanged site
    #: reports every page as freshly scraped, which is the "reports something
    #: untrue" shape this codebase keeps having to correct.
    not_modified: bool = False


def random_user_agent() -> str:
    """Return a random User-Agent from the rotation pool."""
    return random.choice(USER_AGENTS)


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter for retry delays."""
    delay: float = RETRY_BACKOFF_BASE * (2**attempt) + random.uniform(0.0, 0.5)
    return delay


#: Ceiling on an honoured ``Retry-After``. A server may ask for an hour, or for
#: a date far in the future; a scraper that blocks on it for longer than this is
#: indistinguishable from a hung one. Over the cap the backoff is used instead,
#: so the run degrades to "try again soon" rather than "stop".
_MAX_RETRY_AFTER = 120.0


def _retry_after_seconds(value: str | None) -> float | None:
    """
    Parse a ``Retry-After`` header into seconds.

    Two forms are defined: delta-seconds and an HTTP-date. Only the first is
    worth much to a scraper — an HTTP-date is rounded to the second and is
    almost always in the past by the time it is read, since the server computed
    it when it sent the response — but it is cheap to support, and a date a few
    seconds ahead is still an instruction.

    A negative delta is treated as malformed rather than as "no delay": it means
    the server is already misbehaving, and retrying instantly against whatever
    is rate-limiting us is the wrong response to that.
    """
    if not value:
        return None
    # Anything that is not a string cannot be a header value — a test double or
    # an exotic client may hand back anything — and treating that as absent is
    # both correct and total, where `.strip()` on it would raise.
    if not isinstance(value, str):
        return None
    raw = value.strip()
    try:
        seconds = float(raw)
    except ValueError:
        pass
    else:
        return seconds if seconds >= 0 else None
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    import datetime as _dt

    now = _dt.datetime.now(tz=when.tzinfo) if when.tzinfo else _dt.datetime.now()
    return max(0.0, (when - now).total_seconds())


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    """
    How long to wait before retrying: what the server asked for, else backoff.

    A 429 or 503 that carries ``Retry-After`` is a server asking for a specific
    pause, and ignoring it in favour of our own schedule is how a scraper gets
    itself rate-limited harder — or blocked outright. The asked-for delay wins,
    capped so one hostile or mistaken header cannot stall a run indefinitely.
    """
    asked = _retry_after_seconds(retry_after)
    if asked is None or asked > _MAX_RETRY_AFTER:
        return _backoff(attempt)
    return asked


def _conditional_headers(entry: CacheEntry | None) -> dict[str, str]:
    """
    Build revalidation headers from an entry's validators.

    Only ETag/Last-Modified survive serialisation, so those are the only
    validators available to revalidate with. No entry means no headers, which
    leaves the request an ordinary one.
    """
    if entry is None:
        return {}
    headers: dict[str, str] = {}
    if entry.etag:
        headers["If-None-Match"] = entry.etag
    if entry.last_modified:
        headers["If-Modified-Since"] = entry.last_modified
    return headers


def _from_cache(entry: CacheEntry, *, not_modified: bool = False) -> FetchResult:
    """
    Build a result from a cache hit.

    ``nbytes`` reflects the body actually served, not the bytes transferred.
    Reporting 0 made every cached page render as ``—`` in the results table and
    ``0 B total`` in the summary, which reads as "nothing was scraped".
    """
    return FetchResult(
        text=entry.body,
        nbytes=len(entry.body.encode("utf-8")),
        status=entry.status,
        content_type=entry.content_type,
        not_modified=not_modified,
    )


@dataclass
class _Response:
    """A completed HTTP response: everything the caller needs, nothing live."""

    status: int
    #: aiohttp exposes a multidict, not a dict; only ``.get`` is ever needed.
    headers: Mapping[str, str]
    data: bytes
    url: str


async def _get_following(
    session: aiohttp.ClientSession,
    url: str,
    *,
    headers: dict[str, str],
    timeout: int,
    allow_internal_redirects: bool,
) -> _Response:
    """
    GET *url*, following redirects by hand so each hop can be checked.

    Following redirects is left to aiohttp by default, which means a site
    answering ``302 Location: http://169.254.169.254/latest/meta-data/`` sends
    the request to the host's metadata service and the credentials it returns
    are written into the output directory as though they were a web page.
    Nothing in the saved output distinguishes that from a successful scrape.

    Each hop is resolved against the response's own URL, so a relative
    ``Location`` behaves as a browser would, and a hop that leaves the public
    internet is refused unless the caller opted in. See :mod:`protor.netguard`
    for exactly what is and is not blocked, and why loopback is allowed.

    The body is read here rather than handed back live, so the response is
    always released — including on the error paths, where a half-read body would
    otherwise leak the connection.
    """
    current = url
    for _hop in range(MAX_REDIRECTS + 1):
        async with session.get(
            current,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=False,
        ) as response:
            if response.status not in _REDIRECT_STATUSES:
                return _Response(
                    status=response.status,
                    headers=response.headers,
                    data=await response.read(),
                    url=str(response.url),
                )

            location = response.headers.get("Location", "")
            status = response.status
            resolved = urljoin(str(response.url), location) if location else ""
            await response.read()

        if not location:
            raise FetchError(url, f"HTTP {status} with no Location header")
        blocked = describe_block(resolved)
        if blocked is not None and not allow_internal_redirects:
            raise FetchError(url, f"refused to follow a redirect: {blocked}")
        current = resolved

    raise FetchError(url, f"more than {MAX_REDIRECTS} redirects")


async def fetch(
    session: aiohttp.ClientSession,
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    cache: HTTPCache | None = None,
    hooks: dict[str, list[Callable[..., Any]]] | None = None,
    allow_internal_redirects: bool = False,
    user_agent: str | None = None,
) -> FetchResult:
    """
    Fetch *url* with retry logic and conditional caching.

    *user_agent* pins the identity for this request. It exists so a caller that
    has already asked robots.txt about a particular agent can send that same one:
    evaluating the ``User-agent: *`` group while transmitting a browser string
    asks the site about a policy it never agreed to. Omitted, the rotation pool
    picks one as before, so callers with no robots concern are unaffected.

    Each request carries a fresh User-Agent from the rotation pool, so callers
    never mutate a shared session to rotate UAs.

    Raises FetchError on HTTP >= 400 or connection problems.
    """
    # Read the entry without discarding it: an expired one still carries the
    # validators that make a conditional request — and a 304 — possible at all.
    entry = cache.entry_for(url) if cache is not None else None
    if entry is not None and not entry.is_expired:
        body = entry.body
        if entry.nbytes and len(body.encode("utf-8")) != entry.nbytes:
            # The index says this entry holds a body, but reading it produced a
            # different amount — the file was deleted or truncated behind our
            # back. Serving that as a hit returned an empty page marked
            # successful, which is worse than re-fetching. Fall through and get
            # the real content.
            entry = None
        else:
            return _from_cache(entry)

    hook_ctx: dict[str, Any] = {"url": url, "headers": {}}
    for hook in (hooks or {}).get("before_fetch", []):
        with contextlib.suppress(Exception):
            hook(url, hook_ctx)

    headers = {**_conditional_headers(entry), "User-Agent": user_agent or random_user_agent()}
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            r = await _get_following(
                session,
                url,
                headers=headers,
                timeout=timeout,
                allow_internal_redirects=allow_internal_redirects,
            )
            if r.status == 304:
                # 304 means "what you already have is current". Serving it
                # needs a cache; without one there is nothing to serve, and
                # passing the empty body off as a page would report a blank
                # site as successfully scraped.
                if cache is None:
                    raise FetchError(url, "304 Not Modified with no cache entry")
                # The entry a 304 refers to is, by definition, the one that
                # was stale — which get() will not return.
                served = cache.entry_for(url)
                if served is None:
                    raise FetchError(url, "304 Not Modified with no cache entry")
                # Refresh it so the next visit is a disk hit, not another
                # round trip to re-validate the same unchanged page.
                cache.touch(url)
                return _from_cache(served, not_modified=True)
            if r.status >= 400:
                if r.status in RETRYABLE_STATUS and attempt < max_retries - 1:
                    await asyncio.sleep(_retry_delay(attempt, r.headers.get("Retry-After")))
                    continue
                raise FetchError(url, f"HTTP {r.status}")
            data = r.data
            text = data.decode("utf-8", errors="replace")
            if cache:
                cache.put(
                    url,
                    CacheEntry(
                        etag=r.headers.get("ETag"),
                        last_modified=r.headers.get("Last-Modified"),
                        body=text,
                        status=r.status,
                        content_type=r.headers.get("Content-Type", ""),
                    ),
                )
            for hook in (hooks or {}).get("after_fetch", []):
                with contextlib.suppress(Exception):
                    hook(url, {"status": r.status, "body": text})
            return FetchResult(
                text=text,
                nbytes=len(data),
                status=r.status,
                content_type=r.headers.get("Content-Type", ""),
            )
        except TimeoutError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt))
                continue
            raise FetchError(url, "timeout") from exc
        except aiohttp.ClientError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt))
                continue
            raise FetchError(url, str(exc)) from exc

    raise FetchError(url, f"failed after {max_retries} retries") from last_exc


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    dest: str | Path,
    *,
    allow_internal_redirects: bool = False,
) -> bool:
    """
    Download *url* to *dest* (best-effort). Returns True on success.

    Redirects are followed by hand for the same reason :func:`_get_following`
    does it for pages: left to aiohttp, a ``<script src>`` that answers ``302
    Location: http://169.254.169.254/...`` sends the request to the host's
    metadata service and writes the credentials it returns into the output
    directory as though they were a script. The page path was guarded and this
    one was not, on the default path — ``--download-js`` is on unless asked
    otherwise — so the guarantee the README states without qualification did not
    hold for every request protor makes.

    Best-effort throughout, so a refused hop is a ``False`` rather than an
    exception: a script that will not download is not worth failing a page over.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    current = url
    try:
        for _hop in range(MAX_REDIRECTS + 1):
            async with session.get(
                current,
                timeout=aiohttp.ClientTimeout(total=JS_DOWNLOAD_TIMEOUT),
                allow_redirects=False,
            ) as r:
                if r.status in _REDIRECT_STATUSES:
                    location = r.headers.get("Location", "")
                    resolved = urljoin(str(r.url), location) if location else ""
                    await r.read()
                    if not location:
                        return False
                    if describe_block(resolved) is not None and not allow_internal_redirects:
                        return False
                    current = resolved
                    continue
                if r.status == 200:
                    dest.write_bytes(await r.read())
                    return True
                return False
    except Exception:
        pass
    return False
