"""Additional tests for protor.scraper module - async fetch and scrape_multiple."""

import io
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bs4 import BeautifulSoup
from rich.console import Console

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

        session = fake_session(routes={"https://example.com": FakeResponse(status=304)})

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


class TestAPipedRunSaysWhatHappened:
    """
    A pipe used to get a header, silence, and a block at the end.

    Nothing at all was written while the work ran, so a ten-minute scrape logged a
    header, then ten minutes of nothing, then a summary — and which URLs had failed
    appeared nowhere until the run was over. The README has promised "one clean line
    per result" for exactly this case since before the promise was true.

    These drive `scrape_multiple` with `live=False`, which is what a redirect or a
    CI log gets whether or not stdout is a terminal.
    """

    def test_one_line_per_result_as_it_finishes(self, tmp_path, monkeypatch):
        import io

        from rich.console import Console

        import protor.engine as engine_mod
        import protor.scraper as scraper_mod
        from protor.fetcher import FetchResult

        pages = {
            "https://ex.com/": "<html><body><p>index</p></body></html>",
            "https://ex.com/a": "<html><body><p>a</p></body></html>",
        }
        from protor.exceptions import FetchError

        async def fake_fetch(session, url, **kwargs):
            if url == "https://ex.com/missing":
                # `fetch` raises on an error status rather than returning one, so
                # the stub has to as well — and it must not, or the retry backoff
                # turns a 404 into half a minute of test time.
                raise FetchError(url, "HTTP 404")
            return FetchResult(
                text=pages[url], nbytes=len(pages[url]), status=200, content_type="text/html"
            )

        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)

        async def robots_ok(url, session, user_agent=None):
            return True

        monkeypatch.setattr(engine_mod, "check_robots", robots_ok)

        buf = io.StringIO()
        # The result lines go to the console the *engine* holds, not the one the
        # scraper prints its own summary to.
        monkeypatch.setattr(
            engine_mod, "console", Console(file=buf, width=100, force_terminal=False)
        )
        monkeypatch.setattr(
            scraper_mod, "console", Console(file=buf, width=100, force_terminal=False)
        )

        # scrape_multiple is synchronous: it owns its own event loop.
        scraper_mod.scrape_multiple(
            ["https://ex.com/", "https://ex.com/a", "https://ex.com/missing"],
            str(tmp_path),
            live=False,
        )

        out = buf.getvalue()
        assert "\\x1b" not in out, f"a pipe must not receive escape codes:\n{out!r}"
        assert "✓ done" in out, out
        assert "404" in out, f"the failure and its reason are missing:\n{out}"
        # One line per result, and no table repeating them at the end.
        assert out.count("✓ done") == 2, out
        assert "Domain" not in out, "the table duplicates the lines"


class TestAnInterruptReportsWhatWasAlreadySaved:
    """
    Ctrl-C is a normal way for a batch to end, and it used to print nothing.

    `scrape_multiple` had no handler, so the interrupt reached the CLI, which
    printed one line — no page count, no output path, and no sign that every page
    already fetched was sitting on disk with no `sites_index.json` pointing at it.
    """

    def test_the_index_is_written_and_the_counts_reported(self, tmp_path, monkeypatch):
        import json

        import protor.engine as engine_mod
        import protor.scraper as scraper_mod
        from protor.models import SiteManifest, SiteMetadata

        monkeypatch.setattr(engine_mod, "check_robots", _always_allowed)

        real_engine = engine_mod.CrawlEngine

        def interrupting_run(self):
            # Stand in for a run that finished two pages and was then cut short.
            manifest = SiteManifest(
                url="https://ex.com/a",
                domain="ex.com",
                html_file=str(tmp_path / "a"),
                metadata=SiteMetadata(
                    title="A", description="", keywords=[], author="", og_tags={}
                ),
                text_content="a",
                markdown_content="a",
                js_files=[],
                js_count=0,
                bytes_received=29,
                elapsed_ms=3,
                timestamp="2024-01-01 00:00:00",
                success=True,
            )
            self.manifests.append(manifest)
            self.stats.scraped = 1
            self.stats.bytes_total = 29
            raise KeyboardInterrupt

        monkeypatch.setattr(engine_mod.CrawlEngine, "run", interrupting_run)
        monkeypatch.setattr(scraper_mod, "CrawlEngine", real_engine)

        buf = _console()
        monkeypatch.setattr(scraper_mod, "console", buf)
        monkeypatch.setattr(engine_mod, "console", buf)

        with pytest.raises(KeyboardInterrupt):
            scraper_mod.scrape_multiple(["https://ex.com/a"], str(tmp_path), live=False)

        index = tmp_path / "sites_index.json"
        assert index.exists(), "the fetched pages were left with no index pointing at them"
        assert len(json.loads(index.read_text())) == 1

        out = _plain(buf.file.getvalue())
        assert "stopped at" in out, out
        assert "1" in out, out
        assert str(index) in out, "the user is not told where the work went"

    def test_a_completed_run_is_unaffected(self, tmp_path, monkeypatch):
        """The control: an ordinary run must not claim to have been interrupted."""
        import protor.engine as engine_mod
        import protor.scraper as scraper_mod
        from protor.fetcher import FetchResult

        async def fake_fetch(session, url, **kwargs):
            return FetchResult(
                text="<html><body><p>x</p></body></html>",
                nbytes=29,
                status=200,
                content_type="text/html",
            )

        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)
        monkeypatch.setattr(engine_mod, "check_robots", _always_allowed)
        buf = _console()
        monkeypatch.setattr(scraper_mod, "console", buf)
        monkeypatch.setattr(engine_mod, "console", buf)

        index = scraper_mod.scrape_multiple(["https://ex.com/a"], str(tmp_path), live=False)

        assert index.endswith("sites_index.json")
        assert "stopped at" not in _plain(buf.file.getvalue())


async def _always_allowed(*_args, **_kwargs) -> bool:
    return True


def _console() -> Console:
    console = Console(file=None, width=100, force_terminal=False, legacy_windows=False)
    console.file = io.StringIO()
    return console


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestALiveRunDoesNotBuildTheLinesItWouldDiscard:
    """
    `line()` ignores its argument while animating, but the caller has already paid
    to build it.

    Formatting a result line costs 1.2us — two duration and size conversions and a
    join — and a live run discards every one, once per finished page: 48ms over a
    40,000-page crawl for nothing. The engine asks `wants_lines` first, which costs
    52ns, and the test below is on that side of the call rather than on the output.
    """

    def _engine_with(self, display):
        import protor.engine as engine_mod

        engine = engine_mod.CrawlEngine.__new__(engine_mod.CrawlEngine)
        engine._display = display
        # _emit fans out to the status hook too; this test is only about the line.
        engine._on_status = None
        return engine

    def _display(self, live):
        from rich.console import Console

        from protor.progress import LiveDisplay, Throttle

        console = Console(file=io.StringIO(), width=80, force_terminal=False)
        d = LiveDisplay(
            _render=lambda: "x", _live=live, _throttle=Throttle(0), _enabled=True, _console=console
        )
        return d

    def test_a_live_display_is_never_asked_to_format_a_line(self, monkeypatch):
        import protor.engine as engine_mod

        formatted: list[str] = []
        monkeypatch.setattr(
            engine_mod.CrawlEngine, "_result_line", lambda self, *a: formatted.append(a[0]) or "x"
        )

        engine = self._engine_with(self._display(live=object()))
        engine._emit("done", "https://ex.com/a", {"domain": "ex.com"})

        assert formatted == [], "a line was built for a display that discards it"

    def test_a_piped_display_is(self, monkeypatch):
        import protor.engine as engine_mod

        formatted: list[str] = []
        monkeypatch.setattr(
            engine_mod.CrawlEngine, "_result_line", lambda self, *a: formatted.append(a[0]) or "x"
        )

        engine = self._engine_with(self._display(live=None))
        engine._emit("done", "https://ex.com/a", {"domain": "ex.com"})

        assert formatted == ["done"], "the piped path stopped reporting results"

    def test_a_non_terminal_status_is_never_formatted(self, monkeypatch):
        """`fetching` fires for every attempt; only outcomes are worth a line."""
        import protor.engine as engine_mod

        formatted: list[str] = []
        monkeypatch.setattr(
            engine_mod.CrawlEngine, "_result_line", lambda self, *a: formatted.append(a[0]) or "x"
        )

        engine = self._engine_with(self._display(live=None))
        for status in ("fetching", "js:3", "ok"):
            engine._emit(status, "https://ex.com/a", {"domain": "ex.com"})

        assert formatted == [], formatted
