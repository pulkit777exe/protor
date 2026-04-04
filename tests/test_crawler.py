"""Unit tests for protor.crawler module"""
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from protor.crawler import Crawler, _CrawlLog, _render, _State

ROBOTS_PATCH = patch("protor.crawler.check_robots", new_callable=AsyncMock, return_value=True)

ROBOTS_PATCH = patch("protor.crawler.check_robots", new_callable=AsyncMock, return_value=True)

ROBOTS_PATCH = patch("protor.crawler.check_robots", new_callable=AsyncMock, return_value=True)


class TestCrawlLog:
    def test_defaults(self):
        log = _CrawlLog(status="ok", domain="example.com")
        assert log.status == "ok"
        assert log.domain == "example.com"
        assert log.note == ""

    def test_with_note(self):
        log = _CrawlLog(status="err", domain="example.com", note="timeout")
        assert log.note == "timeout"


class TestState:
    def test_defaults(self):
        s = _State()
        assert s.scraped == 0
        assert s.errors == 0
        assert s.current == ""
        assert s.queue_n == 0
        assert s.max_pages == 10
        assert s.log == []

    def test_custom_max_pages(self):
        s = _State(max_pages=50)
        assert s.max_pages == 50

    def test_log_is_independent(self):
        s1 = _State()
        s2 = _State()
        s1.log.append(_CrawlLog("ok", "a.com"))
        assert len(s2.log) == 0


class TestRender:
    def test_render_returns_group(self):
        state = _State(scraped=2, max_pages=10, queue_n=5)
        result = _render(state, "/tmp/output")
        assert result is not None

    def test_render_with_errors(self):
        state = _State(
            scraped=3, max_pages=10, errors=2,
            current="https://example.com/page", queue_n=3,
            log=[_CrawlLog("ok", "example.com"), _CrawlLog("err", "fail.com", note="timeout")]
        )
        result = _render(state, "/tmp/output")
        assert result is not None

    def test_render_empty_state(self):
        state = _State()
        result = _render(state, "/tmp/output")
        assert result is not None


class TestCrawlerInit:
    def test_default_values(self):
        c = Crawler("https://example.com")
        assert c.start_url == "https://example.com"
        assert c.max_pages == 10
        assert c._queue == deque(["https://example.com"])
        assert c._visited == set()

    def test_custom_values(self):
        c = Crawler("https://example.com", max_pages=50, output_dir="/tmp/test")
        assert c.max_pages == 50
        assert c.output_dir == Path("/tmp/test")

    def test_output_dir_defaults_to_home_downloads(self):
        c = Crawler("https://example.com")
        assert "Downloads" in str(c.output_dir)
        assert "protor" in str(c.output_dir)


class TestCrawlerCrawl:
    @pytest.mark.asyncio
    async def test_crawl_single_page_success(self):
        c = Crawler("https://example.com", max_pages=1)
        c._queue = deque(["https://example.com"])

        with patch("protor.crawler.aiohttp.ClientSession"), \
             patch("protor.crawler._fetch", new_callable=AsyncMock) as mock_fetch, \
             patch("protor.crawler.extract_links", return_value=[]), \
             patch("protor.crawler.scrape_site_async", new_callable=AsyncMock) as mock_scrape, \
             patch("protor.crawler.Live"), \
             ROBOTS_PATCH:

            mock_fetch.return_value = ("<html><body>Hello</body></html>", 100)
            mock_scrape.return_value = MagicMock()

            await c._run()

            assert c._state.scraped == 1
            assert c._state.errors == 0
            assert "https://example.com" in c._visited

    @pytest.mark.asyncio
    async def test_crawl_handles_errors(self):
        c = Crawler("https://example.com", max_pages=1)
        c._queue = deque(["https://example.com"])

        with patch("protor.crawler.aiohttp.ClientSession"), \
             patch("protor.crawler._fetch", new_callable=AsyncMock) as mock_fetch, \
             patch("protor.crawler.Live"), \
             ROBOTS_PATCH:

            mock_fetch.side_effect = Exception("Connection refused")

            await c._run()

            assert c._state.scraped == 0
            assert c._state.errors == 1
            assert c._state.log[-1].status == "err"
            assert "Connection refused" in c._state.log[-1].note

    @pytest.mark.asyncio
    async def test_crawl_respects_max_pages(self):
        c = Crawler("https://example.com", max_pages=2)
        c._queue = deque([
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ])

        with patch("protor.crawler.aiohttp.ClientSession"), \
             patch("protor.crawler._fetch", new_callable=AsyncMock) as mock_fetch, \
             patch("protor.crawler.extract_links", return_value=[]), \
             patch("protor.crawler.scrape_site_async", new_callable=AsyncMock) as mock_scrape, \
             patch("protor.crawler.Live"), \
             ROBOTS_PATCH:

            mock_fetch.return_value = ("<html><body>Hello</body></html>", 100)
            mock_scrape.return_value = MagicMock()

            await c._run()

            assert c._state.scraped == 2

    @pytest.mark.asyncio
    async def test_crawl_discovers_links(self):
        c = Crawler("https://example.com", max_pages=3)
        c._queue = deque(["https://example.com"])

        with patch("protor.crawler.aiohttp.ClientSession"), \
             patch("protor.crawler._fetch", new_callable=AsyncMock) as mock_fetch, \
             patch("protor.crawler.extract_links", return_value=[
                 "https://example.com/about",
                 "https://example.com/contact",
             ]), \
             patch("protor.crawler.scrape_site_async", new_callable=AsyncMock) as mock_scrape, \
             patch("protor.crawler.Live"), \
             ROBOTS_PATCH:

            mock_fetch.return_value = ("<html><body>Hello</body></html>", 100)
            mock_scrape.return_value = MagicMock()

            await c._run()

            assert "https://example.com/about" in c._queue or "https://example.com/about" in c._visited
            assert "https://example.com/contact" in c._queue or "https://example.com/contact" in c._visited

    @pytest.mark.asyncio
    async def test_crawl_skips_visited(self):
        c = Crawler("https://example.com", max_pages=5)
        c._queue = deque(["https://example.com", "https://example.com"])
        c._visited = set()

        with patch("protor.crawler.aiohttp.ClientSession"), \
             patch("protor.crawler._fetch", new_callable=AsyncMock) as mock_fetch, \
             patch("protor.crawler.extract_links", return_value=[]), \
             patch("protor.crawler.scrape_site_async", new_callable=AsyncMock) as mock_scrape, \
             patch("protor.crawler.Live"), \
             ROBOTS_PATCH:

            mock_fetch.return_value = ("<html><body>Hello</body></html>", 100)
            mock_scrape.return_value = MagicMock()

            await c._run()

            assert c._state.scraped == 1

    def test_crawl_method_runs(self):
        c = Crawler("https://example.com", max_pages=1)
        c._queue = deque()

        with patch("protor.crawler.asyncio.run") as mock_run:
            c.crawl()
            assert mock_run.called
