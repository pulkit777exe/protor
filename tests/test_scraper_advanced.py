"""Additional tests for protor.scraper module - async fetch and scrape_multiple."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from protor.exceptions import FetchError
from protor.fetcher import download_file, fetch
from protor.http_cache import CacheEntry, HTTPCache
from protor.parser import _extract_js_links, _extract_text
from protor.scraper import (
    _build_table,
    scrape_multiple,
)


class TestFetch:
    @pytest.mark.asyncio
    async def test_fetch_success(self):
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"<html>Hello</html>")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_response)

        result = await fetch(mock_session, "https://example.com")
        assert result.text == "<html>Hello</html>"
        assert result.nbytes == 18  # len(b"<html>Hello</html>")

    @pytest.mark.asyncio
    async def test_fetch_retries_on_500(self):
        mock_session = AsyncMock()
        mock_response_500 = AsyncMock()
        mock_response_500.status = 500
        mock_response_500.__aenter__ = AsyncMock(return_value=mock_response_500)
        mock_response_500.__aexit__ = AsyncMock(return_value=False)

        mock_response_200 = AsyncMock()
        mock_response_200.status = 200
        mock_response_200.read = AsyncMock(return_value=b"ok")
        mock_response_200.__aenter__ = AsyncMock(return_value=mock_response_200)
        mock_response_200.__aexit__ = AsyncMock(return_value=False)

        mock_session.get = MagicMock(side_effect=[mock_response_500, mock_response_200])

        result = await fetch(mock_session, "https://example.com", max_retries=3)
        assert result.text == "ok"

    @pytest.mark.asyncio
    async def test_fetch_raises_after_max_retries(self):
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_response)

        with pytest.raises(FetchError):
            await fetch(mock_session, "https://example.com", max_retries=2)

    @pytest.mark.asyncio
    async def test_fetch_cache_hit(self, fake_session):
        session = fake_session()
        cache = HTTPCache()
        cache.put("https://example.com", CacheEntry(body="cached"))

        result = await fetch(session, "https://example.com", cache=cache)
        assert result.text == "cached"
        # Reports the body it served, not 0 bytes, so cached pages do not
        # render as "—" in the results table.
        assert result.nbytes == len(b"cached")
        assert session.requested == []

    @pytest.mark.asyncio
    async def test_fetch_304_with_cache(self, fake_session):
        from tests.conftest import FakeResponse

        session = fake_session(
            routes={"https://example.com": FakeResponse(status=304)}
        )

        cache = HTTPCache()
        cache.put("https://example.com", CacheEntry(etag="abc", body="cached"))

        result = await fetch(session, "https://example.com", cache=cache)
        assert result.text == "cached"
        assert result.nbytes == len(b"cached")
        assert result.status == 200


class TestDownloadFile:
    @pytest.mark.asyncio
    async def test_download_success(self, tmp_path):
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"js content")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_response)

        dest = tmp_path / "test.js"
        result = await download_file(mock_session, "https://example.com/app.js", dest)

        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"js content"

    @pytest.mark.asyncio
    async def test_download_failure(self, tmp_path, fake_session):
        session = fake_session(raises=ConnectionRefusedError("refused"))

        dest = tmp_path / "test.js"
        result = await download_file(session, "https://example.com/app.js", dest)

        assert result is False
        assert not dest.exists()

    @pytest.mark.asyncio
    async def test_download_non_200_returns_false(self, tmp_path, fake_session):
        from tests.conftest import FakeResponse

        session = fake_session(
            routes={"https://example.com/app.js": FakeResponse(status=404, body="nope")}
        )

        dest = tmp_path / "test.js"
        result = await download_file(session, "https://example.com/app.js", dest)

        assert result is False
        assert not dest.exists()


class TestBuildTable:
    def test_build_table_done(self):
        rows = [
            {"idx": 1, "domain": "example.com", "status": "done", "bytes": 1024, "ms": 100, "js": 2}
        ]
        table = _build_table(rows)
        assert table is not None

    def test_build_table_error(self):
        rows = [
            {"idx": 1, "domain": "example.com", "status": "error", "bytes": 0, "ms": 50, "js": 0}
        ]
        table = _build_table(rows)
        assert table is not None

    def test_build_table_waiting(self):
        rows = [
            {
                "idx": 1,
                "domain": "example.com",
                "status": "waiting",
                "bytes": None,
                "ms": None,
                "js": None,
            }
        ]
        table = _build_table(rows)
        assert table is not None

    def test_build_table_fetching(self):
        rows = [
            {
                "idx": 1,
                "domain": "example.com",
                "status": "fetching",
                "bytes": None,
                "ms": None,
                "js": None,
            }
        ]
        table = _build_table(rows)
        assert table is not None

    def test_build_table_js_status(self):
        rows = [
            {"idx": 1, "domain": "example.com", "status": "js:5", "bytes": 512, "ms": 200, "js": 5}
        ]
        table = _build_table(rows)
        assert table is not None


class TestExtractJsLinksFromSoup:
    def test_finds_script_src(self):
        html = '<html><script src="/app.js"></script><script src="https://cdn.com/lib.js"></script></html>'
        soup = BeautifulSoup(html, "lxml")
        links = _extract_js_links(soup, "https://example.com")
        assert "https://example.com/app.js" in links
        assert "https://cdn.com/lib.js" in links

    def test_ignores_inline_scripts(self):
        html = '<html><script>console.log("hi")</script></html>'
        soup = BeautifulSoup(html, "lxml")
        links = _extract_js_links(soup, "https://example.com")
        assert links == []

    def test_deduplicates(self):
        html = '<html><script src="/app.js"></script><script src="/app.js"></script></html>'
        soup = BeautifulSoup(html, "lxml")
        links = _extract_js_links(soup, "https://example.com")
        assert len(links) == 1


class TestExtractTextFromSoup:
    def test_removes_script_style_nav_footer_header(self):
        html = """<html>
            <script>var x=1;</script>
            <style>.red{}</style>
            <nav>Nav</nav>
            <header>Header</header>
            <footer>Footer</footer>
            <main>Main content</main>
        </html>"""
        soup = BeautifulSoup(html, "lxml")
        text = _extract_text(soup)
        assert "var x=1" not in text
        assert "Main content" in text

    def test_empty_soup(self):
        soup = BeautifulSoup("<html></html>", "lxml")
        text = _extract_text(soup)
        assert text == ""


class TestScrapeMultiple:
    @patch("protor.scraper.console")
    def test_scrape_multiple_empty_urls(self, mock_console):
        result = scrape_multiple([], output_dir="data")
        assert result.endswith("sites_index.json")

    @patch("protor.scraper.console")
    def test_scrape_multiple_custom_output(self, mock_console):
        result = scrape_multiple([], output_dir="/tmp/protor_test_output")
        assert "protor_test_output" in result
