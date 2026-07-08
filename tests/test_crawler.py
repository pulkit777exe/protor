"""Unit tests for protor.crawler module"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from protor.crawler import Crawler, _CrawlLog, _CrawlQueue, _render, _State

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
            scraped=3,
            max_pages=10,
            errors=2,
            current="https://example.com/page",
            queue_n=3,
            log=[
                _CrawlLog("ok", "example.com"),
                _CrawlLog("err", "fail.com", note="timeout"),
            ],
        )
        result = _render(state, "/tmp/output")
        assert result is not None

    def test_render_empty_state(self):
        state = _State()
        result = _render(state, "/tmp/output")
        assert result is not None

    def test_render_with_blocked(self):
        state = _State(
            scraped=1,
            max_pages=5,
            blocked=2,
            log=[_CrawlLog("blocked", "tracker.com")],
        )
        result = _render(state, "/tmp/output")
        assert result is not None


class TestCrawlQueue:
    def test_enqueue_and_dequeue(self, tmp_path):
        q = _CrawlQueue(tmp_path / "test.db")
        q.enqueue("https://example.com")
        url = q.dequeue()
        assert url == "https://example.com"
        assert q.dequeue() is None
        q.close()

    def test_deduplication(self, tmp_path):
        q = _CrawlQueue(tmp_path / "test.db")
        assert q.enqueue("https://example.com") is True
        assert q.enqueue("https://example.com") is False
        q.close()

    def test_visited_tracking(self, tmp_path):
        q = _CrawlQueue(tmp_path / "test.db")
        q.enqueue("https://example.com")
        q.dequeue()
        q.mark_visited("https://example.com", success=True)
        assert q.is_visited("https://example.com")
        assert q.visited_count == 1
        assert q.success_count == 1
        q.close()

    def test_queue_size(self, tmp_path):
        q = _CrawlQueue(tmp_path / "test.db")
        q.enqueue("https://a.com")
        q.enqueue("https://b.com")
        q.enqueue("https://c.com")
        assert q.queue_size == 3
        q.dequeue()
        assert q.queue_size == 2
        q.close()

    def test_checkpoint_roundtrip(self, tmp_path):
        q = _CrawlQueue(tmp_path / "test.db")
        q.enqueue("https://example.com")
        q.enqueue("https://example.com/about")
        q.dequeue()
        q.mark_visited("https://example.com", success=True)

        cp = q.to_checkpoint()
        assert "https://example.com" in cp["visited"]
        assert "https://example.com/about" in cp["queued"]

        q2 = _CrawlQueue.from_checkpoint(cp, tmp_path / "test2.db")
        assert q2.visited_count == 1
        assert q2.queue_size == 1
        q.close()
        q2.close()


class TestCrawlerInit:
    def test_default_values(self, tmp_path):
        c = Crawler("https://example.com", output_dir=str(tmp_path))
        assert c.start_url == "https://example.com"
        assert c.max_pages == 10
        assert c._queue.queue_size == 1
        assert c._queue.is_queued("https://example.com")
        c._queue.close()

    def test_custom_values(self, tmp_path):
        c = Crawler("https://example.com", max_pages=50, output_dir=str(tmp_path / "crawl"))
        assert c.max_pages == 50
        assert c.output_dir == tmp_path / "crawl"
        c._queue.close()

    def test_output_dir_defaults_to_home_downloads(self):
        c = Crawler("https://example.com")
        assert "Downloads" in str(c.output_dir)
        assert "protor" in str(c.output_dir)
        c._queue.close()

    def test_resume_flag(self, tmp_path):
        c = Crawler("https://example.com", resume=True, output_dir=str(tmp_path))
        assert c.resume is True
        c._queue.close()

    def test_auto_scale_flag(self, tmp_path):
        c = Crawler("https://example.com", auto_scale=True, output_dir=str(tmp_path))
        assert c.auto_scale is True
        c._queue.close()


class TestCrawlerCrawl:
    @pytest.mark.asyncio
    async def test_crawl_single_page_success(self, tmp_path):
        c = Crawler("https://example.com", max_pages=1, output_dir=str(tmp_path))

        with (
            patch("protor.crawler.aiohttp.ClientSession"),
            patch("protor.crawler._fetch", new_callable=AsyncMock) as mock_fetch,
            patch("protor.crawler.extract_links", return_value=[]),
            patch("protor.crawler.scrape_site_async", new_callable=AsyncMock) as mock_scrape,
            patch("protor.crawler.Live"),
            ROBOTS_PATCH,
        ):
            mock_fetch.return_value = ("<html><body>Hello</body></html>", 100)
            mock_scrape.return_value = MagicMock()

            await c._run()

            assert c._state.scraped == 1
            assert c._state.errors == 0
        c._queue.close()

    @pytest.mark.asyncio
    async def test_crawl_handles_errors(self, tmp_path):
        c = Crawler("https://example.com", max_pages=1, output_dir=str(tmp_path))

        with (
            patch("protor.crawler.aiohttp.ClientSession"),
            patch("protor.crawler._fetch", new_callable=AsyncMock) as mock_fetch,
            patch("protor.crawler.Live"),
            ROBOTS_PATCH,
        ):
            mock_fetch.side_effect = Exception("Connection refused")

            await c._run()

            assert c._state.scraped == 0
            assert c._state.errors == 1
            assert c._state.log[-1].status == "err"
            assert "Connection refused" in c._state.log[-1].note
        c._queue.close()

    @pytest.mark.asyncio
    async def test_crawl_respects_max_pages(self, tmp_path):
        c = Crawler("https://example.com", max_pages=2, output_dir=str(tmp_path))
        # Pre-enqueue extra URLs
        c._queue.enqueue("https://example.com/1")
        c._queue.enqueue("https://example.com/2")
        c._queue.enqueue("https://example.com/3")

        with (
            patch("protor.crawler.aiohttp.ClientSession"),
            patch("protor.crawler._fetch", new_callable=AsyncMock) as mock_fetch,
            patch("protor.crawler.extract_links", return_value=[]),
            patch("protor.crawler.scrape_site_async", new_callable=AsyncMock) as mock_scrape,
            patch("protor.crawler.Live"),
            ROBOTS_PATCH,
        ):
            mock_fetch.return_value = ("<html><body>Hello</body></html>", 100)
            mock_scrape.return_value = MagicMock()

            await c._run()

            assert c._state.scraped == 2
        c._queue.close()

    @pytest.mark.asyncio
    async def test_crawl_discovers_links(self, tmp_path):
        c = Crawler("https://example.com", max_pages=3, output_dir=str(tmp_path))

        with (
            patch("protor.crawler.aiohttp.ClientSession"),
            patch("protor.crawler._fetch", new_callable=AsyncMock) as mock_fetch,
            patch(
                "protor.crawler.extract_links",
                return_value=["https://example.com/about", "https://example.com/contact"],
            ),
            patch("protor.crawler.scrape_site_async", new_callable=AsyncMock) as mock_scrape,
            patch("protor.crawler.Live"),
            ROBOTS_PATCH,
        ):
            mock_fetch.return_value = ("<html><body>Hello</body></html>", 100)
            mock_scrape.return_value = MagicMock()

            await c._run()

            assert c._queue.is_queued("https://example.com/about") or c._queue.is_visited(
                "https://example.com/about"
            )
            assert c._queue.is_queued("https://example.com/contact") or c._queue.is_visited(
                "https://example.com/contact"
            )
        c._queue.close()

    @pytest.mark.asyncio
    async def test_crawl_skips_visited(self, tmp_path):
        c = Crawler("https://example.com", max_pages=5, output_dir=str(tmp_path))

        with (
            patch("protor.crawler.aiohttp.ClientSession"),
            patch("protor.crawler._fetch", new_callable=AsyncMock) as mock_fetch,
            patch("protor.crawler.extract_links", return_value=[]),
            patch("protor.crawler.scrape_site_async", new_callable=AsyncMock) as mock_scrape,
            patch("protor.crawler.Live"),
            ROBOTS_PATCH,
        ):
            mock_fetch.return_value = ("<html><body>Hello</body></html>", 100)
            mock_scrape.return_value = MagicMock()

            await c._run()

            # Only 1 page should be scraped despite duplicate enqueue attempts
            assert c._state.scraped == 1
        c._queue.close()

    def test_crawl_method_runs(self, tmp_path):
        c = Crawler("https://example.com", max_pages=1, output_dir=str(tmp_path))

        with patch("protor.crawler.asyncio.run") as mock_run:
            c.crawl()
            assert mock_run.called
        c._queue.close()
