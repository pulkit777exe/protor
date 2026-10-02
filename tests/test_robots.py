"""Tests for protor.robots module.

Uses the real async-context-manager doubles from ``tests.conftest`` rather
than AsyncMock sessions: an ``AsyncMock`` used as a *sync* context manager
never enters its body, so ``async with session.get(...)`` silently did
nothing and these tests passed through the ``except Exception`` fallback.
"""

import pytest

from protor.robots import (
    _cache,
    check_robots,
    clear_cache,
    is_allowed,
)
from protor.robots import _fetch_robots as _fetch_robots_impl

BASE = "https://example.com"
ROBOTS = f"{BASE}/robots.txt"
UA = "*"


@pytest.fixture(autouse=True)
def clear_robots_cache():
    """Clear the robots.txt cache before and after each test."""
    _cache.clear()
    yield
    _cache.clear()


def robots_session(body: str = "User-agent: *\nAllow: /\n", status: int = 200):
    from tests.conftest import FakeResponse, FakeSession

    return FakeSession(routes={ROBOTS: FakeResponse(status=status, body=body)})


def failing_session(exc: BaseException):
    from tests.conftest import FakeSession

    return FakeSession(raises=exc)


class TestFetchRobots:
    @pytest.mark.asyncio
    async def test_fetch_robots_caches_result(self):
        session = robots_session("User-agent: *\nDisallow: /private\n")

        rp1 = await _fetch_robots_impl(BASE, session)
        rp2 = await _fetch_robots_impl(BASE, session)

        assert rp1 is rp2
        assert session.requested == [ROBOTS]

    @pytest.mark.asyncio
    async def test_parsed_rules_are_applied(self):
        """The fetched policy must actually gate URLs, not just be cached."""
        session = robots_session("User-agent: *\nDisallow: /private\n")

        rp = await _fetch_robots_impl(BASE, session)

        assert rp.can_fetch(UA, f"{BASE}/private/x") is False
        assert rp.can_fetch(UA, f"{BASE}/public") is True

    @pytest.mark.asyncio
    async def test_fetch_robots_handles_error(self):
        """A transport failure must fall back to allow-all, not crash."""
        session = failing_session(ConnectionRefusedError("refused"))

        rp = await _fetch_robots_impl(BASE, session)

        assert rp.can_fetch(UA, f"{BASE}/anything") is True

    @pytest.mark.asyncio
    async def test_missing_file_allows_all(self):
        session = robots_session(status=404)

        rp = await _fetch_robots_impl(BASE, session)

        assert rp.can_fetch(UA, f"{BASE}/page") is True

    @pytest.mark.asyncio
    async def test_concurrent_checks_fetch_once(self):
        """Concurrent page checks must share a single robots.txt request."""
        import asyncio

        session = robots_session()

        await asyncio.gather(*[check_robots(f"{BASE}/p{i}", session) for i in range(6)])

        assert session.requested == [ROBOTS]


class TestIsAllowed:
    def test_returns_true_when_no_cache(self):
        assert is_allowed(f"{BASE}/page") is True

    def test_returns_true_when_allowed(self):
        from urllib.robotparser import RobotFileParser

        rp = RobotFileParser()
        rp.parse(["User-agent: *", "Allow: /"])
        _cache[BASE] = rp
        assert is_allowed(f"{BASE}/page") is True

    def test_returns_false_when_disallowed(self):
        from urllib.robotparser import RobotFileParser

        rp = RobotFileParser()
        rp.parse(["User-agent: *", "Disallow: /secret"])
        _cache[BASE] = rp
        assert is_allowed(f"{BASE}/secret") is False

    def test_custom_user_agent(self):
        from urllib.robotparser import RobotFileParser

        rp = RobotFileParser()
        rp.parse(["User-agent: Googlebot", "Disallow: /", "User-agent: *", "Allow: /"])
        _cache[BASE] = rp
        assert is_allowed(f"{BASE}/page", user_agent="Googlebot") is False


class TestCheckRobots:
    @pytest.mark.asyncio
    async def test_check_robots_allowed(self):
        assert await check_robots(f"{BASE}/page", robots_session()) is True

    @pytest.mark.asyncio
    async def test_check_robots_disallowed(self):
        session = robots_session("User-agent: *\nDisallow: /private\n")
        assert await check_robots(f"{BASE}/private", session) is False

    @pytest.mark.asyncio
    async def test_check_robots_handles_error(self):
        session = failing_session(ConnectionRefusedError("refused"))
        assert await check_robots(f"{BASE}/page", session) is True


class TestClearCache:
    @pytest.mark.asyncio
    async def test_clear_cache_removes_entries(self):
        await check_robots(f"{BASE}/page", robots_session())
        assert _cache

        clear_cache()

        assert _cache == {}
