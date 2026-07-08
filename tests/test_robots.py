"""Tests for protor.robots module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from protor.robots import (
    _cache,
    _fetch_robots,
    check_robots,
    clear_cache,
    is_allowed,
)


@pytest.fixture(autouse=True)
def clear_robots_cache():
    """Clear the robots.txt cache before and after each test."""
    _cache.clear()
    yield
    _cache.clear()


class TestFetchRobots:
    @pytest.mark.asyncio
    async def test_fetch_robots_caches_result(self):
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="User-agent: *\nDisallow: /private\n")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_response)

        rp1 = await _fetch_robots("https://example.com", mock_session)
        rp2 = await _fetch_robots("https://example.com", mock_session)

        assert rp1 is rp2
        assert mock_session.get.call_count == 1

    @pytest.mark.asyncio
    async def test_fetch_robots_handles_error(self):
        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("Connection refused")

        rp = await _fetch_robots("https://example.com", mock_session)
        assert rp is not None


class TestIsAllowed:
    def test_returns_true_when_no_cache(self):
        assert is_allowed("https://example.com/page") is True

    def test_returns_true_when_allowed(self):
        from urllib.robotparser import RobotFileParser

        rp = RobotFileParser()
        rp.parse(["User-agent: *", "Allow: /"])
        _cache["https://example.com"] = rp
        assert is_allowed("https://example.com/page") is True

    def test_returns_false_when_disallowed(self):
        from urllib.robotparser import RobotFileParser

        rp = RobotFileParser()
        rp.parse(["User-agent: *", "Disallow: /secret"])
        _cache["https://example.com"] = rp
        assert is_allowed("https://example.com/secret") is False

    def test_custom_user_agent(self):
        from urllib.robotparser import RobotFileParser

        rp = RobotFileParser()
        rp.parse(["User-agent: Googlebot", "Disallow: /", "User-agent: *", "Allow: /"])
        _cache["https://example.com"] = rp
        assert is_allowed("https://example.com/page", user_agent="Googlebot") is False


class TestCheckRobots:
    @pytest.mark.asyncio
    async def test_check_robots_allowed(self):
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="User-agent: *\nAllow: /\n")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_response)

        result = await check_robots("https://example.com/page", mock_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_robots_disallowed(self):
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="User-agent: *\nDisallow: /page\n")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_response)

        result = await check_robots("https://example.com/page", mock_session)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_robots_handles_error(self):
        from protor.robots import _cache

        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("Connection refused")

        await check_robots("https://example.com/page", mock_session)
        # When robots.txt can't be fetched, an empty RobotFileParser is cached
        # and can_fetch returns True for an empty parser
        assert _cache.get("https://example.com") is not None


class TestClearCache:
    def test_clear_cache(self):
        from urllib.robotparser import RobotFileParser

        _cache["https://example.com"] = RobotFileParser()
        assert len(_cache) > 0

        clear_cache()
        assert len(_cache) == 0
