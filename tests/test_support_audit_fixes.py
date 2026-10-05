"""Regressions for the support-layer audit: scaler, progress and robots.

Each class below corresponds to one defect found by reading the support modules
against their own documented contracts. Every test says in its docstring what
was broken, why it mattered, and — where the defect was a docstring rather than
behaviour — what the code actually does instead.
"""

from __future__ import annotations

import inspect
import io
import re
import socket
from typing import Any

import pytest
from rich.console import Console

from protor import netguard
from protor import robots as robots_module
from protor.config import (
    SCALING_COOLDOWN,
    SCALING_DOWN_THRESHOLD,
    SCALING_MAX_CONCURRENCY,
    SCALING_MIN_CONCURRENCY,
    SCALING_UP_THRESHOLD,
    SCALING_WINDOW,
)
from protor.progress import live_display, live_enabled, probing
from protor.robots import RobotsCache, check_robots, is_allowed
from protor.scaler import AutoScaler

# ── helpers ───────────────────────────────────────────────────────────────────


def _pipe(width: int = 80) -> Console:
    """A console that is definitely not a terminal (a pipe or a CI log)."""
    return Console(file=io.StringIO(), width=width, force_terminal=False, legacy_windows=False)


def _terminal(encoding: str = "utf-8") -> Console:
    """A console that reports itself as a real terminal with *encoding*."""
    stream: Any = io.TextIOWrapper(io.BytesIO(), encoding=encoding, newline="")
    return Console(
        file=stream, width=80, force_terminal=True, color_system="truecolor", legacy_windows=False
    )


#: The escape sequences a rich spinner writes that are not colour: cursor-up,
#: erase-line, hide/show-cursor. A CI log renders these as literal text, and they
#: are what turn a log into an unreadable transcript. SGR colour codes are
#: deliberately not in this set — a console told it is a terminal is entitled to
#: colour, and colour is not the defect under test.
MOTION_ESCAPES = ("\x1b[?25l", "\x1b[?25h", "\x1b[1A", "\x1b[2K", "\x1b[1B")


def _motions(text: str) -> list[str]:
    """The cursor-motion escape sequences present in *text*."""
    return [seq for seq in MOTION_ESCAPES if seq in text]


def _parsed_entry(body: str) -> Any:
    """A cache entry holding an already-parsed policy (bypasses the network)."""
    import time as _time

    policy = robots_module._Policy()
    policy.parse(body.splitlines())
    return robots_module._Entry(policy=policy, fetched_at=_time.monotonic())


@pytest.fixture
def live_env(monkeypatch):
    """Neither switch set, so a test states its own rather than inheriting one."""
    monkeypatch.delenv("PROTOR_NO_LIVE", raising=False)
    monkeypatch.delenv("CI", raising=False)


# ── 1. the scaler escalates on a window it never consumes ─────────────────────


def _production_scaler(initial: int = 4) -> AutoScaler:
    """The scaler exactly as ``config`` configures it in production."""
    return AutoScaler(
        initial=initial,
        min_c=SCALING_MIN_CONCURRENCY,
        max_c=SCALING_MAX_CONCURRENCY,
        up_threshold=SCALING_UP_THRESHOLD,
        down_threshold=SCALING_DOWN_THRESHOLD,
        window=SCALING_WINDOW,
        cooldown=SCALING_COOLDOWN,
    )


@pytest.fixture
def clock(monkeypatch):
    """A hand-advanced monotonic clock, so cooldowns elapse without sleeping."""
    now = {"t": 10_000.0}
    monkeypatch.setattr("protor.scaler.time.monotonic", lambda: now["t"])
    return now


def _advance_one_cooldown(clock: dict[str, float]) -> None:
    clock["t"] += SCALING_COOLDOWN + 0.01


class TestScalerConsumesTheWindowItJudged:
    """
    ``record()`` appended to ``_results`` and trimmed it; nothing ever consumed it.

    ``maybe_scale()`` read the window and left every sample in place, so the same
    ten successes were re-judged once per cooldown for as long as they were the
    newest information the scaler had. With the production config
    (initial=4, window=10, up=0.8, down=0.5, max=20, cooldown=5.0) the very
    condition the scaler exists to react to — the site stops answering, so no new
    ``record()`` arrives — drove concurrency 4 → 6 → 8 → … → 20 off one sample
    with zero new evidence: five times the request rate against a host that had
    already gone quiet.

    The existing ``test_scale_resets_cooldown`` only asserted the second call
    inside the cooldown, so it could never see this.
    """

    def test_a_stalled_site_does_not_climb_on_one_sample(self, clock):
        scaler = _production_scaler(initial=4)
        for _ in range(SCALING_WINDOW):
            scaler.record(True)

        _advance_one_cooldown(clock)
        assert scaler.maybe_scale() == 6, "one full window of successes scales up once"

        # The site stalls. Nothing finishes, so ``record`` is never called again —
        # and no failure arrives either, so there is nothing to scale *down* on.
        for _ in range(8):
            _advance_one_cooldown(clock)
            scaler.maybe_scale()

        assert scaler.concurrency == 6, (
            f"the scaler climbed to {scaler.concurrency} on one window of results; "
            "the site had stopped answering"
        )

    def test_a_dead_site_does_not_fall_on_one_sample(self, clock):
        """The same defect read the other way: one bad window is one decision."""
        scaler = _production_scaler(initial=SCALING_MAX_CONCURRENCY)
        for _ in range(SCALING_WINDOW):
            scaler.record(False)

        _advance_one_cooldown(clock)
        assert scaler.maybe_scale() == SCALING_MAX_CONCURRENCY - 1

        for _ in range(8):
            _advance_one_cooldown(clock)
            scaler.maybe_scale()

        assert scaler.concurrency == SCALING_MAX_CONCURRENCY - 1

    def test_a_window_is_judged_once_and_then_spent(self, clock):
        """
        The invariant behind both: one window of samples buys one decision.

        Asserted directly, because "the second call must not move it again" only
        says anything if the cooldown is genuinely elapsed and the window is
        genuinely still full.
        """
        scaler = _production_scaler(initial=4)
        for _ in range(SCALING_WINDOW):
            scaler.record(True)
        assert len(scaler._results) == SCALING_WINDOW

        _advance_one_cooldown(clock)
        scaler.maybe_scale()

        assert scaler._results == [], (
            "the samples that produced a decision are still on the books; "
            "the next cooldown will judge them again"
        )

    def test_fresh_samples_still_scale_up(self, clock):
        """
        The control: consuming the window must not stop the scaler working.

        A live crawl keeps recording, so each round of ten fresh successes buys
        one increment. Before the fix this also passed — it is here to catch a
        fix that solves the stall by disabling scaling.
        """
        scaler = _production_scaler(initial=4)
        levels = []
        for _ in range(3):
            for _ in range(SCALING_WINDOW):
                scaler.record(True)
            _advance_one_cooldown(clock)
            levels.append(scaler.maybe_scale())

        assert levels == [6, 8, 10], levels

    def test_a_partial_window_is_not_judged(self, clock):
        """The gate that keeps a crawl from deciding on two samples is intact."""
        scaler = _production_scaler(initial=4)
        for _ in range(SCALING_WINDOW):
            scaler.record(True)
        _advance_one_cooldown(clock)
        scaler.maybe_scale()  # 4 -> 6, window spent

        # One straggler page arrives while the site is otherwise quiet.
        scaler.record(True)
        for _ in range(8):
            _advance_one_cooldown(clock)
            scaler.maybe_scale()

        assert scaler.concurrency == 6, "one sample must not re-open the decision"


# ── 2. probing() answered "is this a terminal?" instead of asking live_enabled ─


class TestProbingAsksLiveEnabled:
    """
    ``probing()`` decided animation from ``target.is_terminal`` on its own.

    ``live_enabled`` is this module's one source of truth for "can this stream
    take animation": it refuses ``PROTOR_NO_LIVE``, refuses CI, refuses a
    non-terminal and refuses an encoding that cannot draw the glyphs.
    ``live_display`` asks it. ``probing`` — added recently and reached from
    ``protor runtimes`` and ``protor analyze`` — asked ``is_terminal`` alone, so
    ``CI=true protor runtimes`` in a container built with ``tty: true`` or
    ``docker -t`` wrote cursor-up escape sequences into a CI log, and
    ``PROTOR_NO_LIVE=1`` was overridden by the caller it exists to overrule.

    A pty in CI is not an edge case: it is what GitHub Actions gives a step
    configured with ``tty: true``, and it is the configuration under which a
    human most wants the plain-line rendering.
    """

    def test_no_live_gets_a_plain_line_not_a_spinner(self, live_env, monkeypatch):
        monkeypatch.setenv("PROTOR_NO_LIVE", "1")
        buf = io.StringIO()
        con = Console(file=buf, width=80, force_terminal=True, color_system="truecolor")

        with probing("probing 17 runtimes...", con):
            pass

        out = buf.getvalue()
        assert _motions(out) == [], "PROTOR_NO_LIVE=1 must not animate"
        # Stripped of colour first: the console was built with force_terminal, so
        # rich highlights "17" and is entitled to. The words are what matters.
        assert "probing 17 runtimes" in re.sub(r"\x1b\[[0-9;]*m", "", out)

    def test_ci_with_a_pty_gets_a_plain_line(self, live_env, monkeypatch):
        monkeypatch.setenv("CI", "true")
        buf = io.StringIO()
        con = Console(file=buf, width=80, force_terminal=True, color_system="truecolor")

        with probing("probing 17 runtimes...", con):
            pass

        assert _motions(buf.getvalue()) == [], "a CI log must not receive cursor escapes"

    def test_a_cp1252_terminal_gets_a_plain_line(self, live_env):
        """
        The legacy console the glyph fallback exists for.

        Rich's spinner frame is a braille character, so animating here raised
        ``UnicodeEncodeError: 'charmap' codec can't encode character '\\u280b'``
        out of the middle of the status context — a crash, not a cosmetic
        substitution, and one that ``protor runtimes`` does not catch.
        """
        raw = io.BytesIO()
        stream: Any = io.TextIOWrapper(raw, encoding="cp1252", newline="")
        con = Console(
            file=stream,
            width=80,
            force_terminal=True,
            color_system="truecolor",
            legacy_windows=False,
        )

        with probing("probing 17 runtimes...", con):
            pass

        written = raw.getvalue().decode("cp1252", "replace")
        assert _motions(written) == [], "a console that cannot draw the spinner must not animate"
        assert "probing 17 runtimes" in re.sub(r"\x1b\[[0-9;]*m", "", written)

    def test_a_real_terminal_still_gets_the_spinner(self, live_env):
        """
        The control: routing through ``live_enabled`` must not cost the animation.

        Without it, a fix that simply always printed a plain line would pass every
        test above and quietly delete the feature the function exists for.
        """
        buf = io.StringIO()
        con = Console(file=buf, width=80, force_terminal=True, color_system="truecolor")

        with probing("probing 17 runtimes...", con):
            pass

        assert _motions(buf.getvalue()) != [], "a real terminal lost its spinner"

    def test_a_pipe_is_unaffected(self, live_env):
        """The pre-existing non-terminal path already printed one plain line."""
        buf = io.StringIO()
        con = Console(file=buf, width=80, force_terminal=False)

        with probing("probing 17 runtimes...", con):
            pass

        assert "\x1b" not in buf.getvalue()
        assert "probing 17 runtimes" in buf.getvalue()

    def test_an_exception_still_escapes(self, live_env):
        """A context manager that hides failures turns a crash into a hang."""
        with pytest.raises(RuntimeError), probing("working...", _terminal()):
            raise RuntimeError("probe blew up")


class TestLiveEnabledRefusesAnEncodingThatCannotDrawTheGlyphs:
    """
    ``live_enabled`` refused ``encoding == "ascii"`` and nothing else.

    Its own docstring justifies the check with "the glyphs would be replaced on
    every repaint", and ``theme.py`` names the environment that sentence is
    about: a cp1252 console cannot encode ``✓``/``✗``/``◌``/``→``, which is why
    the glyph fallback exists at all. Testing one hardcoded string against one
    hardcoded encoding answered a narrower question than the one asked — the
    Windows console the fallback was written for passed the gate.

    The check now asks the real question, against the console in hand, using the
    round-trip test ``theme._can_encode`` already established: an encoding that
    can encode a glyph but decode it back as a different character is as broken
    as one that cannot encode it (cp1252's em dash is the canonical example).
    """

    @pytest.mark.parametrize("encoding", ["ascii", "cp1252", "latin-1", "cp437"])
    def test_a_console_that_cannot_carry_the_glyphs_is_refused(self, live_env, encoding):
        assert live_enabled(_terminal(encoding)) is False, encoding

    def test_utf8_is_still_allowed(self, live_env):
        assert live_enabled(_terminal("utf-8")) is True

    def test_the_test_host_really_would_fail_the_encoding_check(self, live_env):
        """
        Otherwise the parameterisation above could pass for the wrong reason.

        If ``theme``'s tokens had already degraded to ASCII on this host (they
        resolve against ``sys.stdout`` at import), the glyph set would be
        encodable everywhere and every refusal would be a false positive.
        """
        from protor.theme import ACTIVE, ARROW, ERR, OK

        with pytest.raises(UnicodeEncodeError):
            "".join([OK, ERR, ACTIVE, ARROW]).encode("cp1252")


# ── 3. the robots.txt request did not carry the identity it was judged by ─────


BASE = "https://example.com"
ROBOTS = f"{BASE}/robots.txt"

#: One of the fifteen strings ``config.USER_AGENTS`` rotates between. urllib
#: reduces it to the token before the first "/", i.e. ``mozilla``.
CHROME_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

#: What the engine's session carries as a default, so a robots.txt request
#: inherited a browser identity the caller never chose. Every pool entry starts
#: ``Mozilla/5.0``, so a ``User-agent: Chrome`` group the site published was never
#: the group consulted.
SESSION_DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class _RecordingResponse:
    """A 200 robots.txt body, served differently per identity."""

    def __init__(self, body_for: Any) -> None:
        self.status = 200
        self._body_for = body_for
        #: Filled in by the session below, so the response can answer per identity.
        self.sent_agent = ""

    async def __aenter__(self) -> _RecordingResponse:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def text(self) -> str:
        return self._body_for(self.sent_agent)


class _RecordingSession:
    """Records the ``User-Agent`` of every request and serves per identity.

    ``tests.conftest.FakeSession`` swallows keyword arguments, so it could not see
    the defect: the request it stood in for carried no headers at all, and the
    test could not tell that apart from one carrying the right header.
    """

    def __init__(self, body_for: Any, session_agent: str = SESSION_DEFAULT_UA) -> None:
        self._body_for = body_for
        self._session_agent = session_agent
        self.sent_agents: list[str | None] = []
        self.requested: list[str] = []

    def get(self, url: str, headers: Any = None, **_kwargs: object) -> _RecordingResponse:
        self.requested.append(url)
        # A per-request header wins over the session default, which is what aiohttp
        # does; with none given the session's own default is what goes out.
        agent = None if headers is None else headers.get("User-Agent")
        if agent is None:
            agent = self._session_agent
        self.sent_agents.append(agent)

        response = _RecordingResponse(self._body_for)
        response.sent_agent = agent or ""
        return response


class TestRobotsRequestCarriesTheJudgedIdentity:
    """
    ``_load_robots`` could not send a User-Agent even when asked to.

    The module's headline guarantee (``check_robots``'s docstring, robots.py:306)
    is that ``user_agent`` "must be the string the request actually sends.
    Evaluating one identity while transmitting another asks the site about a
    policy it never agreed to" — and then ``_load_robots(base, session)`` took no
    identity at all and issued the robots.txt request with no headers. So the
    crawl evaluated the rules against one of fifteen rotated agents and fetched
    them under the session's fixed Chrome/124; because every pool entry starts
    ``Mozilla/5.0``, urllib reduced the evaluated identity to ``mozilla`` and a
    ``User-agent: Chrome``-specific group was never consulted.

    The sitemap path was worse: ``crawler._seed_from_sitemaps`` opens a session
    with no headers, so its robots.txt request announced itself as aiohttp's
    default ``Python/3.x aiohttp/3.y``.

    The agent is threaded as an explicit argument and never read from
    ``config.HEADERS`` — ``tests/test_robots.py`` pins that out of the module on
    purpose, because gating on the session's User-Agent was a past bug.
    """

    @staticmethod
    def _per_identity_policy(agent: str) -> str:
        """A site publishing a group only a real browser identity is given."""
        if "Chrome" in agent:
            return "User-agent: *\nDisallow: /\n\nUser-agent: Chrome\nAllow: /\n"
        return "User-agent: *\nDisallow: /\n"

    async def test_the_request_sends_the_agent_the_rules_were_judged_by(self):
        cache = RobotsCache()
        session = _RecordingSession(self._per_identity_policy)

        await cache.check(f"{BASE}/page", session, CHROME_UA)

        assert session.requested == [ROBOTS]
        assert session.sent_agents == [CHROME_UA], (
            "the robots.txt request went out under a different identity than the "
            "one the rules were evaluated against"
        )

    async def test_the_site_answers_about_the_identity_we_declared(self):
        """
        The bug's whole cost, end to end: one declared identity, two answers.

        A site that serves a crawler its own group and everyone else the wildcard
        block. Ask as ``protor`` and the body that comes back is the group that
        grants ``/`` — and the rules are then evaluated as ``protor``, so the
        answer is the one for the identity on the wire. Ask as anyone else and
        the same declaration would be refused, because the body served belongs to
        a policy ``protor`` never agreed to. Before the fix the header was simply
        absent, so the wildcard body came back and a correctly-declared crawler
        was refused by rules it had never been shown.
        """
        identity = "protor/2.9 (+https://example.invalid/bot)"

        def policy_for(agent: str) -> str:
            if agent.startswith("protor/"):
                return "User-agent: *\nDisallow: /\n\nUser-agent: protor\nAllow: /\n"
            return "User-agent: *\nDisallow: /\n"

        cache = RobotsCache()
        session = _RecordingSession(policy_for, session_agent="")

        assert await cache.check(f"{BASE}/page", session, identity) is True
        assert session.sent_agents == [identity]

    async def test_a_wildcard_identity_does_not_claim_to_be_chrome(self):
        """The RFC 9309 default still sends ``*``, not the session's Chrome string."""
        cache = RobotsCache()
        session = _RecordingSession(self._per_identity_policy)

        await cache.check(f"{BASE}/page", session)

        assert session.sent_agents == ["*"]

    async def test_the_loader_takes_the_agent_as_an_argument(self):
        """Directly, so the guarantee does not rest on ``check`` plumbing it."""
        session = _RecordingSession(self._per_identity_policy)

        policy = await robots_module._load_robots(BASE, session, CHROME_UA)

        assert policy is not None
        assert session.sent_agents == [CHROME_UA]

    async def test_the_sitemap_path_sends_an_identity_not_aiohttp_s_default(self):
        """
        ``sitemaps`` reads ``Sitemap:`` lines, so it evaluates no rule — but it
        still has to *ask*, and the request it made announced itself as
        ``Python/3.x aiohttp/3.y``. It now sends the cache's own identity.
        """
        cache = RobotsCache()
        session = _RecordingSession(
            lambda agent: "User-agent: *\nAllow: /\nSitemap: https://x.example/s.xml\n",
            session_agent="",  # a bare session: aiohttp's default is all there is
        )

        assert await cache.sitemaps(f"{BASE}/page", session) == ["https://x.example/s.xml"]
        assert session.sent_agents == ["*"]
        assert "aiohttp" not in (session.sent_agents[0] or "")

    async def test_a_crawl_can_configure_the_identity_its_robots_request_sends(self):
        """The thread-through, so a crawler can name the agent it rotates within."""
        cache = RobotsCache(user_agent=CHROME_UA)
        session = _RecordingSession(self._per_identity_policy)

        await cache.sitemaps(f"{BASE}/page", session)

        assert session.sent_agents == [CHROME_UA]

    async def test_the_module_still_does_not_read_the_session_header(self):
        """
        The deliberate constraint this fix had to respect.

        ``config.HEADERS``' fixed Chrome/124 must not come back as the source of
        truth: it is threaded in as an argument, so the identity is the caller's
        decision and never a module-level guess.
        """
        assert not hasattr(robots_module, "HEADERS")
        assert "HEADERS" not in inspect.getsource(robots_module)
        # ``config`` itself is fine — ``DEFAULT_TIMEOUT`` comes from it. What
        # must not come from it is an identity.
        from_config = [
            line
            for line in inspect.getsource(robots_module).splitlines()
            if line.startswith("from .config")
        ]
        assert from_config == ["from .config import DEFAULT_TIMEOUT"]

    async def test_the_crawl_path_is_repaired_end_to_end(self, monkeypatch):
        """
        Through the real entry point, with the real session the engine builds.

        The thread-through is only worth anything if the module-level wrapper
        honours it, so this goes via ``check_robots`` with the session's own
        fixed Chrome/124 default — the exact situation that made the evaluated
        identity (``mozilla``, from a rotated pool entry) and the sent identity
        (whatever ``config.HEADERS`` said) disagree.
        """
        from protor import config

        monkeypatch.setattr(robots_module, "_default_cache", RobotsCache())
        sent: list[str] = []

        class _EngineSession:
            """Models aiohttp's merge: request headers override session defaults."""

            def __init__(self) -> None:
                self.defaults = dict(config.HEADERS)

            def get(self, url: str, headers: Any = None, **_kwargs: object) -> _RecordingResponse:
                merged = {**self.defaults, **(headers or {})}
                agent_on_the_wire = merged.get("User-Agent", "")
                sent.append(agent_on_the_wire)
                response = _RecordingResponse(lambda agent: "User-agent: *\nAllow: /\n")
                response.sent_agent = agent_on_the_wire
                return response

        agent = config.USER_AGENTS[0]
        assert await check_robots(f"{BASE}/page", _EngineSession(), agent) is True

        assert sent == [agent], (
            "evaluated the rules against a rotated pool entry but fetched them "
            f"under the session's fixed {config.HEADERS['User-Agent'][:30]!r}…"
        )


# ── 4. is_allowed() read a different cache than check_robots(cache=…) wrote ────


class TestIsAllowedCannotDisagreeWithTheCacheItWasAskedAbout:
    """
    ``is_allowed`` was unconditionally bound to ``_default_cache``.

    ``check_robots(url, session, ua, cache=owned)`` honours a caller-supplied
    cache, and the module header says the two helpers "share a process-wide
    default cache" — true only on the no-``cache=`` path. So the flow the docs
    recommend (an owned :class:`RobotsCache` per crawl) left the public
    ``is_allowed`` answering True — "nothing cached, therefore allowed" — for URLs
    the owned cache blocks, and that False allowed was indistinguishable from a
    real one.

    The fix gives ``is_allowed`` the same optional ``cache`` argument
    ``check_robots`` has, so a caller who owns a cache can ask that cache. The
    default-cache read is unchanged, so an owned cache still cannot pollute the
    default — the property ``tests/test_robots.py`` pins in the other direction.
    """

    def test_is_allowed_accepts_the_cache_the_caller_owns(self):
        """Before the fix this raised TypeError: ``is_allowed`` took no cache."""
        owned = RobotsCache()
        owned._entries[BASE] = _parsed_entry("User-agent: *\nDisallow: /secret\n")

        assert is_allowed(f"{BASE}/secret", cache=owned) is False
        assert is_allowed(f"{BASE}/public", cache=owned) is True

    async def test_the_pair_cannot_disagree_for_one_crawl(self):
        """The whole point: one crawl, one cache, one answer."""
        owned = RobotsCache()
        session = _RecordingSession(
            lambda agent: "User-agent: *\nDisallow: /secret\n",
            session_agent="",
        )

        blocked = await check_robots(f"{BASE}/secret", session, cache=owned)
        assert blocked is False

        assert is_allowed(f"{BASE}/secret", cache=owned) is False, (
            "check_robots said no and is_allowed said yes for the same cache"
        )

    async def test_an_owned_cache_still_does_not_touch_the_default(self, monkeypatch):
        """
        The other direction, which ``tests/test_robots.py`` already pins.

        Fixing the disagreement must not be done by writing the owned cache's
        entries into the default one: a caller that owns a cache owns it precisely
        so one run cannot inherit another's policies.
        """
        monkeypatch.setattr(robots_module, "_default_cache", RobotsCache())
        owned = RobotsCache()
        session = _RecordingSession(
            lambda agent: "User-agent: *\nDisallow: /secret\n",
            session_agent="",
        )

        assert await check_robots(f"{BASE}/secret", session, cache=owned) is False

        assert len(owned) == 1
        assert len(robots_module._default_cache) == 0

    def test_the_bare_call_still_reads_the_default(self, monkeypatch):
        """The no-cache path is unchanged: the convenience wrapper still works."""
        default = RobotsCache()
        default._entries[BASE] = _parsed_entry("User-agent: *\nDisallow: /secret\n")
        monkeypatch.setattr(robots_module, "_default_cache", default)

        assert is_allowed(f"{BASE}/secret") is False
        assert is_allowed(f"{BASE}/public") is True

    def test_a_third_argument_is_still_the_identity_not_a_cache(self, monkeypatch):
        """
        Positional compatibility, asserted because the fix added a parameter.

        ``is_allowed(url, user_agent)`` is the documented signature and appears in
        the README; inserting ``cache`` before it would silently turn a
        ``User-agent: Googlebot`` argument into a cache object.
        """
        default = RobotsCache()
        default._entries[BASE] = _parsed_entry(
            "User-agent: Googlebot\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
        )
        monkeypatch.setattr(robots_module, "_default_cache", default)

        assert is_allowed(f"{BASE}/page", "Googlebot") is False
        assert is_allowed(f"{BASE}/page") is True


# ── 5. two docstrings described code that does not exist ──────────────────────


class TestIsMetadataHostMakesNoDnsCall:
    """
    ``is_metadata_host``'s docstring claimed a resolver it never calls.

    It reads "True when *host* is, **or resolves by name to**, a metadata
    endpoint." Nothing resolves: the body is a four-entry frozenset of names and
    an exact compare against three literal addresses after normalisation. The
    module header already states the real limitation honestly — a redirect to a
    hostname that resolves to an internal address is not caught — so the function
    docstring is the one place that overstates it, on the exact function a
    maintainer would read before deciding to relax it.

    **The docstring fix needs ``protor/netguard.py``, which is outside this
    audit's edit scope, so it is not applied here.** What follows pins the
    behaviour the docstring should be describing, so whoever edits that file has
    the contract in executable form and cannot widen the function by accident:
    correcting the text to match these assertions is a docstring-only change and
    breaks nothing here.
    """

    def test_no_resolver_is_called(self, monkeypatch):
        """The checkable form of the claim: a DNS lookup would be observable."""

        def explode(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must not run
            raise AssertionError(f"is_metadata_host resolved a name: {args!r}")

        monkeypatch.setattr(socket, "getaddrinfo", explode)
        monkeypatch.setattr(socket, "gethostbyname", explode)
        monkeypatch.setattr(socket, "gethostbyaddr", explode)

        assert netguard.is_metadata_host("metadata.google.internal") is True
        assert netguard.is_metadata_host("169.254.169.254") is True
        assert netguard.is_metadata_host("example.com") is False

    def test_a_name_that_would_resolve_to_metadata_is_still_allowed(self, monkeypatch):
        """
        The concrete consequence of there being no resolver.

        ``metadata.internal.example.com`` is not in the named set and is not an IP
        literal, so it is allowed — whatever it would resolve to. A docstring
        saying "or resolves by name to" is a promise this answer breaks.
        """

        def explode(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must not run
            raise AssertionError("is_metadata_host resolved a name")

        monkeypatch.setattr(socket, "getaddrinfo", explode)

        assert netguard.is_metadata_host("metadata.internal.example.com") is False

    def test_the_docstring_still_overstates_it(self):
        """
        The unfixed half, recorded as an assertion so the gap is visible.

        This one documents the defect rather than pinning the fix: it passes *because*
        the docstring still claims a lookup. Apply the netguard.py correction and this
        test fails by design, which is the signal to delete it. The three tests above
        are the durable ones — they pass either way and describe what the function
        actually does.
        """
        doc = netguard.is_metadata_host.__doc__ or ""
        assert "resolves by name to" in doc, (
            "netguard.is_metadata_host's docstring has been corrected to match the code; "
            "delete this test, which exists only to record the defect"
        )


class TestLiveDisplayDisabledBranchCommentMatchesTheCode:
    """
    ``live_display``'s disabled branch carried a comment about a print that does
    not exist.

    It opened "Print the final state once, so a pipe still gets the per-result
    detail…" and closed by invoking ``contextlib.suppress``, with a bare ``return``
    in between: nothing printed, ``render()`` was never called on that path, and
    ``contextlib`` is not imported by the module at all. Worse, the two halves
    argued opposite positions — the first said the pipe *needed* a final render,
    the second said the final render *would only repeat* what the caller had
    already written — while the code did neither.

    The behaviour is right and is pinned by ``tests/test_progress.py``, so this
    is text only.
    """

    def test_the_disabled_branch_calls_render_nothing(self):
        """
        The behaviour the comment claimed, stated as the fact it actually is.

        The comment promised a final print and described a suppressed exception
        around a call that does not happen; this is the checkable version — the
        disabled path builds no frame at all, on entry or on exit.
        """
        calls: list[int] = []
        console = _pipe()

        def render() -> str:
            calls.append(1)
            return "frame"

        with live_display(render, console=console, enabled=False) as display:
            display.update()
            display.update(force=True)

        assert calls == [], "the disabled path must not render a frame"
        assert console.file.getvalue() == "", "and must not print one either"

    def test_the_comment_no_longer_describes_a_print(self):
        source = inspect.getsource(live_display)
        assert "Print the final state once" not in source
        assert "contextlib.suppress" not in source

    def test_the_comment_takes_one_position(self):
        """
        The specific failure: a comment that says both "print the final state" and
        "the final table would only repeat what the caller already wrote".

        Whichever way the next reader takes it, one of those two sentences has to
        be false, so the fix has to drop the pair rather than pick a side. The
        surviving text may explain why *not* to print, but must not also promise
        to.
        """
        source = inspect.getsource(live_display)
        assert "would only repeat" in source, "the accurate explanation was dropped"
        assert "Print the final state" not in source, "the inaccurate promise is back"
        assert "so a pipe still gets" not in source
