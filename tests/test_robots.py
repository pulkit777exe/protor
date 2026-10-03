"""Tests for protor.robots module.

Uses the real async-context-manager doubles from ``tests.conftest`` rather
than AsyncMock sessions: an ``AsyncMock`` used as a *sync* context manager
never enters its body, so ``async with session.get(...)`` silently did
nothing and these tests passed through the ``except Exception`` fallback.

Every test builds its own :class:`RobotsCache` instead of resetting a module
global before and after each test. That autouse reset was a symptom: a shared
process-wide cache meant one test's fetch could answer another test's question.
The module-level default cache is only touched by the handful of tests that
exercise it, via the scoped ``default_cache`` fixture.
"""

import asyncio
import gc
import inspect
import time
from typing import Any

import pytest

from protor import robots as robots_module
from protor.robots import (
    RobotsCache,
    _discard_outcome,
    _load_robots,
    check_robots,
    clear_cache,
    is_allowed,
)

BASE = "https://example.com"
ROBOTS = f"{BASE}/robots.txt"
UA = "*"

# A rule set that only a declared identity is allowed through: the wildcard
# group blocks everything, the Googlebot group carves out one path. This is the
# case the old fixed-Chrome-string gate got wrong.
BOT_SPECIFIC = "User-agent: *\nDisallow: /\n\nUser-agent: Googlebot\nAllow: /\n"


@pytest.fixture
def cache() -> RobotsCache:
    """An isolated cache, so no test can observe another's state."""
    return RobotsCache()


@pytest.fixture
def default_cache() -> RobotsCache:
    """Isolate the tests that exercise the module-level default cache.

    Scoped to those tests deliberately: an autouse reset is exactly what the
    cache object exists to stop needing. Returns the cache itself so a test can
    assert a reset actually emptied it.
    """
    clear_cache()
    yield robots_module._default_cache
    clear_cache()


def robots_session(body: str = "User-agent: *\nAllow: /\n", status: int = 200):
    from tests.conftest import FakeResponse, FakeSession

    return FakeSession(routes={ROBOTS: FakeResponse(status=status, body=body)})


def failing_session(exc: BaseException):
    from tests.conftest import FakeSession

    return FakeSession(raises=exc)


class _BlockingResponse:
    """A response that only fails once *release* is set."""

    def __init__(self, release: asyncio.Event, exc: BaseException) -> None:
        self.status = 200
        self._release = release
        self._exc = exc

    async def __aenter__(self) -> "_BlockingResponse":
        await self._release.wait()
        raise self._exc

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _BlockingSession:
    """Stalls every fetch until *release* is set, then fails.

    Used to place a cancellation (or a cache reset) between "the fetch started"
    and "the fetch failed", which is the only window in which an orphaned task
    can still raise.
    """

    def __init__(self, release: asyncio.Event, exc: BaseException | None = None) -> None:
        self._release = release
        self._exc = exc or ConnectionRefusedError("refused")
        self.requested: list[str] = []

    def get(self, url: str, **_kwargs: object) -> _BlockingResponse:
        self.requested.append(url)
        return _BlockingResponse(self._release, self._exc)


class TestCaching:
    @pytest.mark.asyncio
    async def test_policy_is_cached_and_reused(self, cache: RobotsCache):
        session = robots_session("User-agent: *\nDisallow: /private\n")

        assert await cache.check(f"{BASE}/private", session) is False
        assert await cache.check(f"{BASE}/public", session) is True

        assert session.requested == [ROBOTS]
        assert cache.lookup(f"{BASE}/public") is not None

    @pytest.mark.asyncio
    async def test_parsed_rules_are_applied(self, cache: RobotsCache):
        """The fetched policy must actually gate URLs, not just be cached."""
        session = robots_session("User-agent: *\nDisallow: /private\n")

        rp = await cache._policy_for(BASE, session)

        assert rp is not None
        assert rp.can_fetch(UA, f"{BASE}/private/x") is False
        assert rp.can_fetch(UA, f"{BASE}/public") is True

    @pytest.mark.asyncio
    async def test_absent_robots_is_remembered(self, cache: RobotsCache):
        """A 404 is the site genuinely saying "no rules", so it is remembered.

        This is the case that *must* be cached: there is nothing to re-learn, and
        re-asking every page would hammer a site for an answer it has already
        given.
        """
        session = robots_session(status=404)

        assert await cache.check(f"{BASE}/page", session) is True
        assert await cache.check(f"{BASE}/other", session) is True

        assert session.requested == [ROBOTS]
        assert cache.lookup(f"{BASE}/page") is not None

    @pytest.mark.asyncio
    async def test_concurrent_checks_fetch_once(self, cache: RobotsCache):
        """Concurrent page checks must share a single robots.txt request."""
        session = robots_session()

        await asyncio.gather(*[cache.check(f"{BASE}/p{i}", session) for i in range(6)])

        assert session.requested == [ROBOTS]

    @pytest.mark.asyncio
    async def test_expired_policy_is_refetched(self):
        """A cached policy expires, so a long run notices a tightened robots.txt.

        ``ttl=0`` stands in for "the TTL has elapsed" without making the test
        sleep; the mechanism under test is the staleness check, not the clock.
        """
        stale = RobotsCache(ttl=0)
        session = robots_session("User-agent: *\nDisallow: /private\n")

        assert await stale.check(f"{BASE}/private", session) is False
        assert stale.lookup(f"{BASE}/private") is None

        assert await stale.check(f"{BASE}/private", session) is False
        assert session.requested == [ROBOTS, ROBOTS]

    @pytest.mark.asyncio
    async def test_unexpired_policy_is_not_refetched(self):
        session = robots_session("User-agent: *\nDisallow: /private\n")
        long_lived = RobotsCache(ttl=3600)

        await long_lived.check(f"{BASE}/private", session)
        await long_lived.check(f"{BASE}/private", session)

        assert session.requested == [ROBOTS]


class TestFailedFetchIsNotRemembered:
    """A failed fetch must never become a cached allow-everything decision."""

    @pytest.mark.asyncio
    async def test_transport_failure_is_not_cached(self, cache: RobotsCache):
        """One DNS blip must not license every later request for the whole run."""
        session = failing_session(ConnectionRefusedError("refused"))

        # The request itself is allowed: the site's rules are unknown, so there
        # is nothing to enforce.
        assert await cache.check(f"{BASE}/anything", session) is True

        assert cache.lookup(f"{BASE}/anything") is None
        assert cache.is_allowed(f"{BASE}/anything") is True

    @pytest.mark.asyncio
    async def test_failed_fetch_is_retried(self, cache: RobotsCache):
        """Because nothing was learned, the next page must ask again."""
        session = failing_session(ConnectionRefusedError("refused"))

        await cache.check(f"{BASE}/a", session)
        await cache.check(f"{BASE}/b", session)

        assert session.requested == [ROBOTS, ROBOTS]

    @pytest.mark.asyncio
    async def test_http_500_is_not_cached(self, cache: RobotsCache):
        """A 5xx is the site failing, not declining to publish a policy."""
        session = robots_session("User-agent: *\nDisallow: /\n", status=500)

        assert await cache.check(f"{BASE}/page", session) is True
        assert cache.lookup(f"{BASE}/page") is None

        await cache.check(f"{BASE}/page", session)
        assert session.requested == [ROBOTS, ROBOTS]

    @pytest.mark.asyncio
    async def test_concurrent_failures_share_one_fetch(self, cache: RobotsCache):
        """Not caching failures must not turn a broken robots.txt into a storm.

        The in-flight de-duplication is what keeps "ask again next time" polite:
        a whole burst of pages still costs exactly one request.
        """
        session = failing_session(ConnectionRefusedError("refused"))

        results = await asyncio.gather(*[cache.check(f"{BASE}/p{i}", session) for i in range(6)])

        assert results == [True] * 6
        assert session.requested == [ROBOTS]
        assert cache.lookup(f"{BASE}/p0") is None

    @pytest.mark.asyncio
    async def test_recovery_enforces_the_real_rules(self, cache: RobotsCache):
        """A retry after a failure must surface the policy, not the fallback."""
        from tests.conftest import FakeResponse

        session = failing_session(ConnectionRefusedError("refused"))
        assert await cache.check(f"{BASE}/private", session) is True

        session.raises = None
        session.routes[ROBOTS] = FakeResponse(
            status=200, body="User-agent: *\nDisallow: /private\n"
        )

        assert await cache.check(f"{BASE}/private", session) is False
        assert session.requested == [ROBOTS, ROBOTS]


class TestUserAgentIdentity:
    @pytest.mark.asyncio
    async def test_user_agent_argument_selects_the_rule_group(self, cache: RobotsCache):
        """Rules must be evaluated against the agent the caller says it is."""
        session = robots_session(BOT_SPECIFIC)

        assert await cache.check(f"{BASE}/page", session, "Googlebot") is True

    @pytest.mark.asyncio
    async def test_default_identity_obeys_the_wildcard_group(self, cache: RobotsCache):
        """With no declared identity the wildcard group applies — the safe answer."""
        session = robots_session(BOT_SPECIFIC)

        assert await cache.check(f"{BASE}/page", session) is False
        assert await cache.check(f"{BASE}/page", session, UA) is False

    @pytest.mark.asyncio
    async def test_module_does_not_gate_on_the_session_user_agent(self):
        """robots.py must not reach for config.HEADERS' fixed browser string.

        ``fetch()`` sends a random agent from config.USER_AGENTS per request, so a
        gate reading the session's Chrome/124 header checks a policy for an
        identity the page request does not use. robots.py can only fix its half of
        that by taking the agent as an argument; this pins the stale source of
        truth out of the module.
        """
        assert not hasattr(robots_module, "HEADERS")
        assert "HEADERS" not in inspect.getsource(robots_module)


class TestIsAllowed:
    """The cache-only read path: no I/O, no policy in hand means allowed."""

    def test_returns_true_when_no_policy(self, cache: RobotsCache):
        """No policy in hand means allowed — and nothing gets cached for asking."""
        assert cache.is_allowed(f"{BASE}/page") is True
        assert cache.lookup(f"{BASE}/page") is None

    def test_returns_true_when_allowed(self, cache: RobotsCache):
        cache._entries[BASE] = _entry("User-agent: *\nAllow: /\n")
        assert cache.is_allowed(f"{BASE}/page") is True

    def test_returns_false_when_disallowed(self, cache: RobotsCache):
        cache._entries[BASE] = _entry("User-agent: *\nDisallow: /secret\n")
        assert cache.is_allowed(f"{BASE}/secret") is False

    def test_custom_user_agent(self, cache: RobotsCache):
        cache._entries[BASE] = _entry(
            "User-agent: Googlebot\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
        )
        assert cache.is_allowed(f"{BASE}/page", user_agent="Googlebot") is False
        assert cache.is_allowed(f"{BASE}/page") is True

    def test_ignores_an_expired_policy(self):
        stale = RobotsCache(ttl=0)
        stale._entries[BASE] = _entry("User-agent: *\nDisallow: /secret\n")
        assert stale.is_allowed(f"{BASE}/secret") is True
        assert stale.lookup(f"{BASE}/secret") is None


class TestModuleApi:
    @pytest.mark.asyncio
    async def test_check_robots_uses_the_default_cache(self, default_cache):
        session = robots_session("User-agent: *\nDisallow: /private\n")

        assert await check_robots(f"{BASE}/private", session) is False
        assert is_allowed(f"{BASE}/private") is False
        assert session.requested == [ROBOTS]

    @pytest.mark.asyncio
    async def test_check_robots_handles_error(self, default_cache):
        session = failing_session(ConnectionRefusedError("refused"))
        assert await check_robots(f"{BASE}/page", session) is True

    @pytest.mark.asyncio
    async def test_check_robots_passes_the_user_agent_through(self, default_cache):
        session = robots_session(BOT_SPECIFIC)

        assert await check_robots(f"{BASE}/page", session, "Googlebot") is True
        assert await check_robots(f"{BASE}/page", session) is False

    @pytest.mark.asyncio
    async def test_an_owned_cache_does_not_leak_into_the_default(self, default_cache):
        """Passing a cache keeps its state to itself — no process-global bleed."""
        session = robots_session("User-agent: *\nDisallow: /secret\n")
        owned = RobotsCache()

        assert await check_robots(f"{BASE}/secret", session, cache=owned) is False

        assert owned.lookup(f"{BASE}/secret") is not None
        assert is_allowed(f"{BASE}/secret") is True

    def test_is_allowed_is_true_when_the_default_cache_is_empty(self, default_cache):
        assert is_allowed(f"{BASE}/page") is True

    def test_clear_cache_removes_entries(self, default_cache):
        default_cache._entries[BASE] = _entry("User-agent: *\nDisallow: /secret\n")
        assert len(default_cache) == 1

        clear_cache()

        assert len(default_cache) == 0
        assert is_allowed(f"{BASE}/secret") is True

    def test_module_declares_its_public_api(self):
        assert robots_module.__all__ == [
            "RobotsCache",
            "check_robots",
            "clear_cache",
            "is_allowed",
        ]
        for name in robots_module.__all__:
            assert hasattr(robots_module, name)


class TestResetSafety:
    @pytest.mark.asyncio
    async def test_clear_cancels_inflight_fetches(self, cache: RobotsCache):
        """A reset must not leave a request running or a waiter hanging on it."""
        release = asyncio.Event()
        session = _BlockingSession(release)
        waiter = asyncio.create_task(cache.check(f"{BASE}/page", session))
        await asyncio.sleep(0)  # let the check reach its stalled fetch

        assert len(cache) == 0
        assert cache._inflight

        cache.clear()

        assert len(cache) == 0
        assert cache._inflight == {}
        with pytest.raises(asyncio.CancelledError):
            await waiter

    @pytest.mark.asyncio
    async def test_the_next_check_after_a_reset_fetches_again(self, cache: RobotsCache):
        session = robots_session("User-agent: *\nDisallow: /secret\n")

        assert await cache.check(f"{BASE}/secret", session) is False
        cache.clear()
        assert await cache.check(f"{BASE}/secret", session) is False

        assert session.requested == [ROBOTS, ROBOTS]

    @pytest.mark.asyncio
    async def test_dropping_a_failing_fetch_logs_no_task_exception(self, monkeypatch):
        """End to end: an orphaned fetch that fails logs nothing.

        ``asyncio.shield`` keeps the shared fetch alive when the caller waiting on
        it gives up, so a fetch can finish with nobody awaiting it, and asyncio
        answers that at garbage collection with "Task exception was never
        retrieved" — noise that reads as a bug in the crawler.

        Note this guarantee holds because of ``shield``; see
        :func:`_discard_outcome`, which is insurance on top of it and is covered
        directly by the two tests below rather than by this one.
        """
        release = asyncio.Event()

        async def exploding_load(base: str, _session: Any) -> Any:
            # A failure the loader's own handler did not anticipate, which is
            # exactly the case the report above is about.
            await release.wait()
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(robots_module, "_load_robots", exploding_load)

        seen: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, ctx: seen.append(ctx))
        try:
            cache = RobotsCache()
            waiter = asyncio.create_task(cache.check(f"{BASE}/page", _BlockingSession(release)))
            await asyncio.sleep(0)  # the check now owns the shared fetch
            waiter.cancel()  # ... and now walks away from it
            await asyncio.sleep(0)
            release.set()  # the orphaned fetch fails with nobody awaiting it
            for _ in range(3):
                await asyncio.sleep(0)
            gc.collect()
        finally:
            loop.set_exception_handler(previous)

        assert [ctx.get("message") for ctx in seen] == []

    @pytest.mark.asyncio
    async def test_discard_outcome_clears_the_flag_asyncio_reports_on(self):
        """The helper must actually retrieve a failed fetch's exception.

        ``Future.set_exception`` arms ``_log_traceback``, and ``Task.__del__``
        reports on it if the outcome is never retrieved — which is precisely the
        "Task exception was never retrieved" message asyncio emits at GC time.
        """

        async def boom() -> None:
            raise ConnectionRefusedError("refused")

        task = asyncio.ensure_future(boom())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert task.done()
        assert task._log_traceback is True  # armed: nobody has read the outcome

        _discard_outcome(task)

        assert task._log_traceback is False

    def test_discard_outcome_tolerates_a_cancelled_task(self):
        """The reset helper is called on live tasks too, so it must not raise."""

        async def scenario() -> None:
            task = asyncio.ensure_future(asyncio.sleep(10))
            task.add_done_callback(_discard_outcome)
            task.cancel()
            await asyncio.sleep(0)
            assert task.cancelled()
            # Would raise CancelledError if the guard were missing.
            _discard_outcome(task)

        asyncio.run(scenario())


class TestAllowAllPolicy:
    def test_allow_everything_permits_every_url(self):
        policy = robots_module._Policy.allow_everything()

        assert policy.allow_all is True
        assert policy.can_fetch(UA, f"{BASE}/anything") is True

    def test_default_policy_is_not_allow_everything(self):
        """The flag is a real typed attribute, not a poke at a stdlib private."""
        policy = robots_module._Policy()

        assert policy.allow_all is False
        assert policy.can_fetch(UA, f"{BASE}/anything") is False

    def test_the_allow_all_flag_is_declared_here_not_borrowed(self):
        """``_Policy`` owns the field, so the module needs no type suppression.

        ``RobotFileParser.allow_all`` is undocumented and missing from the type
        stubs on every supported version, so reaching for it costs a
        ``type: ignore`` and depends on CPython's internals holding still.
        Declaring the field on our own subclass pins the behaviour and keeps
        ``mypy`` clean without a suppression.
        """
        assert "allow_all" in robots_module._Policy.__annotations__
        assert "# type: ignore" not in inspect.getsource(robots_module)

    def test_a_parsed_policy_enforces_its_rules(self):
        """Parsing a real body must produce a gating policy, not an allow-all one."""
        policy = robots_module._Policy()
        policy.parse(["User-agent: *", "Disallow: /"])

        assert policy.allow_all is False
        assert policy.can_fetch(UA, f"{BASE}/x") is False

    @pytest.mark.asyncio
    async def test_a_fetched_body_enforces_its_rules(self, cache: RobotsCache):
        session = robots_session("User-agent: *\nDisallow: /\n")

        assert await cache.check(f"{BASE}/x", session) is False
        assert session.requested == [ROBOTS]


def _entry(body: str) -> Any:
    """A cache entry holding an already-parsed policy (bypasses the network)."""
    policy = robots_module._Policy()
    policy.parse(body.splitlines())
    return robots_module._Entry(policy=policy, fetched_at=time.monotonic())


class TestLoaderContract:
    """``_load_robots`` owns the one distinction everything else depends on."""

    @pytest.mark.asyncio
    async def test_returns_none_on_transport_failure(self):
        assert await _load_robots(BASE, failing_session(OSError("no route"))) is None

    @pytest.mark.asyncio
    async def test_returns_a_policy_on_404(self):
        assert await _load_robots(BASE, robots_session(status=404)) is not None

    @pytest.mark.asyncio
    async def test_returns_none_on_503(self):
        assert await _load_robots(BASE, robots_session(status=503)) is None

    @pytest.mark.asyncio
    async def test_undecodable_body_is_a_failure_not_an_absence(self):
        """A body we cannot decode is a failed fetch, not a permissive answer."""
        from tests.conftest import FakeResponse, FakeSession

        class BadText(FakeResponse):
            async def text(self) -> str:
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        session = FakeSession(routes={ROBOTS: BadText(status=200, body=b"\xff")})

        assert await _load_robots(BASE, session) is None

    def test_loader_requires_a_session(self):
        """The blocking ``rp.read()`` branch is gone, so ``session`` is required.

        It was unreachable from check_robots (which always passes a session) and
        it called a *synchronous* network read on the event loop.
        """
        with pytest.raises(TypeError):
            _load_robots(BASE)  # type: ignore[call-arg]
