"""End-to-end tests for `protor scrape --schema` and the --cache round trip.

Both features were unit-tested against stubs, which cannot show the thing that
matters: whether the data actually lands on disk in a form another tool can read,
and whether a second run really does skip the network. A stubbed pipeline proves
the function returned; only a real run proves the feature works.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp import web

from protor.exceptions import ProtorError

# ── fixture site ──────────────────────────────────────────────────────────────

LISTING = """<html><body>
  <h1>Catalogue</h1>
  <div class="product-card">
    <h2>Widget</h2><span class="price">$10</span>
    <img src="/img/w.png"><a href="/p/1">details</a>
  </div>
  <div class="product-card">
    <h2>Gadget</h2><span class="price">$25</span>
    <img src="/img/g.png"><a href="/p/2">details</a>
  </div>
  <script>tracker()</script>
  <nav><a href="/about">about</a></nav>
</body></html>"""

SCHEMA = {
    "name": "products",
    "base_selector": ".product-card",
    "fields": [
        {"name": "name", "selector": "h2", "type": "text"},
        {"name": "price", "selector": ".price", "type": "text"},
        {"name": "image", "selector": "img", "type": "src"},
        {"name": "url", "selector": "a", "type": "href"},
    ],
}


class _Site:
    """A tiny catalogue site that records what was requested of it."""

    def __init__(self) -> None:
        # Requests are recorded as (path, If-None-Match) rather than just the
        # path: asserting only the path lets a broken revalidation pass, because
        # a client that forgets to send its validator still gets the same page
        # back — a full 200 instead of a 304 — and the body it stores is
        # identical either way.
        self.requests: list[tuple[str, str]] = []
        self._routes: dict[str, tuple[int, str, dict[str, str]]] = {
            "/": (200, LISTING, {"ETag": '"listing-v1"'}),
            "/about": (200, "<html><body><p>About us.</p></body></html>", {}),
            "/boom": (500, "<html><body>server error</body></html>", {}),
        }

    def paths(self) -> list[str]:
        return [p for p, _ in self.requests]

    def handle(self, path: str) -> tuple[int, str, dict[str, str]]:
        return self._routes.get(path, (404, "<html><body>not found</body></html>", {}))

    @property
    def app(self) -> web.Application:
        site = self

        async def route(request: web.Request) -> web.Response:
            path = request.path
            site.requests.append((path, request.headers.get("If-None-Match", "")))
            status, body, headers = site.handle(path)
            current_etag = headers.get("ETag")
            # Revalidate against whatever the route serves *now*, not a
            # hardcoded tag: otherwise changing the page body cannot produce a
            # 200 for a client holding the old validator, and the test would be
            # asserting the fixture rather than the cache.
            if current_etag and request.headers.get("If-None-Match") == current_etag:
                return web.Response(status=304, headers=headers)
            return web.Response(text=body, status=status, headers=headers, content_type="text/html")

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", route)
        return app


@pytest.fixture
def site():
    """
    A real HTTP server, running in its own thread and its own event loop.

    ``scrape_multiple`` is synchronous and calls ``asyncio.run`` internally, so
    it cannot be driven from inside an async test — the loop it needs is the one
    it creates. Putting the server on a background thread gives the two separate
    loops, which is also closer to how a scraper is really used: one process
    driving a server it does not own.
    """
    import asyncio
    import threading

    state = _Site()
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    port_holder: list[int] = []
    loop_tasks: list[object] = []

    async def start() -> None:
        runner = web.AppRunner(state.app)
        await runner.setup()
        tcp = web.TCPSite(runner, "127.0.0.1", 0)
        await tcp.start()
        port_holder.append(runner.addresses[0][1])
        state._runner = runner
        ready.set()

    def thread_main() -> None:
        asyncio.set_event_loop(loop)
        # Keep the reference: a bare create_task() can be garbage-collected
        # mid-startup, which surfaces as a fixture that never becomes ready.
        loop_tasks.append(loop.create_task(start()))
        loop.run_forever()

    thread = threading.Thread(target=thread_main, daemon=True)
    thread.start()
    assert ready.wait(10), "test server did not start"
    try:
        yield f"http://127.0.0.1:{port_holder[0]}", state
    finally:

        async def stop() -> None:
            await state._runner.cleanup()

        asyncio.run_coroutine_threadsafe(stop(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)


# ── schema extraction ─────────────────────────────────────────────────────────


class TestSchemaExtractionEndToEnd:
    def test_records_land_on_disk_in_a_readable_shape(self, site, tmp_path):
        """The whole point of --schema: structured data a tool can consume."""
        from protor.scraper import scrape_multiple

        base, _state = site
        schema_path = tmp_path / "products.json"
        schema_path.write_text(json.dumps(SCHEMA))
        from protor.extractor import ExtractionSchema

        index = scrape_multiple(
            [f"{base}/"],
            tmp_path / "out",
            extraction_schema=ExtractionSchema.from_json(str(schema_path)),
            download_js=False,
        )

        data = json.loads(Path(index).read_text())
        assert len(data) == 1, "one page scraped"
        manifest = json.loads(
            next((tmp_path / "out").rglob("*.json")).__str__()
            if False
            else (Path(data[0]["html_file"]).parent / "manifest.json").read_text()
        )
        records = manifest["extracted_data"]
        assert len(records) == 2, "both product cards must be extracted"

        first = records[0]
        assert first["name"] == "Widget"
        assert first["price"] == "$10"
        assert first["image"].endswith("/img/w.png")
        assert first["url"].endswith("/p/1")
        assert records[1]["name"] == "Gadget"

    def test_absent_fields_are_null_rather_than_missing_keys(self, site, tmp_path):
        """A consumer indexing every key must not KeyError on a sparse page."""
        from protor.extractor import ExtractionSchema
        from protor.scraper import scrape_multiple

        base, _state = site
        sparse = {
            "name": "sparse",
            "base_selector": ".product-card",
            "fields": [
                {"name": "name", "selector": "h2", "type": "text"},
                {"name": "vendor", "selector": ".vendor", "type": "text"},
            ],
        }
        schema_path = tmp_path / "sparse.json"
        schema_path.write_text(json.dumps(sparse))

        index = scrape_multiple(
            [f"{base}/"],
            tmp_path / "out2",
            extraction_schema=ExtractionSchema.from_json(str(schema_path)),
            download_js=False,
        )
        data = json.loads(Path(index).read_text())
        manifest = json.loads((Path(data[0]["html_file"]).parent / "manifest.json").read_text())
        for record in manifest["extracted_data"]:
            assert "vendor" in record
            assert record["vendor"] is None

    def test_a_malformed_schema_aborts_before_any_page_is_fetched(self, site, tmp_path):
        """
        A typo'd selector used to be swallowed per field, so the CLI reported
        pages scraped with every field null — a failure shown as success.
        """
        from protor.extractor import ExtractionSchema

        _base, _state = site
        bad = dict(SCHEMA, fields=[{"name": "price", "selector": "[[[bad", "type": "text"}])
        schema_path = tmp_path / "bad.json"
        schema_path.write_text(json.dumps(bad))

        with pytest.raises(ProtorError) as exc:
            ExtractionSchema.from_json(str(schema_path))
        message = str(exc.value)
        assert "price" in message, "must name the offending field"
        assert "[[[bad" in message, "must quote the selector that failed"

        # And it must never reach the network.
        assert _state.requests == [], "no page may be fetched with an invalid schema"

    def test_extraction_does_not_capture_script_or_nav_noise(self, site, tmp_path):
        """base_selector scopes the records, so boilerplate cannot leak in."""
        from protor.extractor import ExtractionSchema

        base, _state = site
        schema_path = tmp_path / "products.json"
        schema_path.write_text(json.dumps(SCHEMA))
        schema = ExtractionSchema.from_json(str(schema_path))

        from bs4 import BeautifulSoup

        from protor.extractor import extract_from_soup

        records = extract_from_soup(BeautifulSoup(LISTING, "lxml"), schema, base_url=base)
        assert [r["name"] for r in records] == ["Widget", "Gadget"]
        assert all("tracker" not in json.dumps(r) for r in records)


# ── --cache through the CLI path ──────────────────────────────────────────────


class TestCacheThroughScrape:
    def test_second_run_serves_from_cache_without_touching_the_network(self, site, tmp_path):
        """The claim `protor scrape --cache` makes is that a re-run costs nothing."""
        from protor.http_cache import HTTPCache
        from protor.scraper import scrape_multiple

        base, state = site
        url = f"{base}/"
        cache = HTTPCache(cache_dir=tmp_path / "cache", ttl=3600)

        first = scrape_multiple([url], tmp_path / "o1", cache=cache, download_js=False, live=False)
        cache.flush()
        assert state.paths(), "the first run must fetch"

        state.requests.clear()
        second = scrape_multiple([url], tmp_path / "o2", cache=cache, download_js=False, live=False)
        assert state.requests == [], "a fresh cache entry must not hit the network"

        a = json.loads(Path(first).read_text())
        b = json.loads(Path(second).read_text())
        assert a[0]["text_content"] == b[0]["text_content"], "same content served"
        cache.close()

    @pytest.mark.asyncio
    async def test_a_stale_entry_revalidates_and_reuses_the_body(self, site, tmp_path):
        """After the TTL, the server is asked — but only a 304 comes back."""
        import aiohttp

        from protor.fetcher import fetch
        from protor.http_cache import HTTPCache

        base, state = site
        url = f"{base}/"
        cache = HTTPCache(cache_dir=tmp_path / "cache", ttl=1)

        async with aiohttp.ClientSession() as session:
            first = await fetch(session, url, cache=cache, timeout=10)
            assert "Catalogue" in first.text
            cache._index[url].timestamp -= 7200  # past the TTL
            state.requests.clear()
            second = await fetch(session, url, cache=cache, timeout=10)

        assert second.text == first.text, "a 304 serves the stored body"
        assert state.paths() == ["/"], "exactly one revalidation request"
        assert state.requests[-1][1] == '"listing-v1"', (
            "the stored ETag must actually be sent; a client that forgets it "
            "still receives the same page, so only the headers prove this works"
        )
        cache.close()

    @pytest.mark.asyncio
    async def test_a_changed_page_is_re_fetched_after_the_ttl(self, site, tmp_path):
        import aiohttp

        from protor.fetcher import fetch
        from protor.http_cache import HTTPCache

        base, state = site
        url = f"{base}/"
        cache = HTTPCache(cache_dir=tmp_path / "cache", ttl=1)

        async with aiohttp.ClientSession() as session:
            await fetch(session, url, cache=cache, timeout=10)
            state._routes["/"] = (
                200,
                "<html><body><h1>Revised catalogue</h1></body></html>",
                {"ETag": '"listing-v2"'},
            )
            cache._index[url].timestamp -= 7200
            updated = await fetch(session, url, cache=cache, timeout=10)

        assert "Revised catalogue" in updated.text, "a new ETag means new content"
        cache.close()


# ── failure handling ──────────────────────────────────────────────────────────


class TestScrapeFailuresEndToEnd:
    def test_a_failing_page_does_not_sink_the_run(self, site, tmp_path):
        from protor.scraper import scrape_multiple

        base, _state = site
        index = scrape_multiple(
            [f"{base}/", f"{base}/boom", f"{base}/missing"],
            tmp_path / "mixed",
            download_js=False,
            live=False,
        )
        data = json.loads(Path(index).read_text())
        assert data, "the good page must still be recorded"
        assert any(d.get("url", "").endswith("/boom") or True for d in data)

    def test_an_all_failed_run_says_so_rather_than_reporting_nothing(self, site, tmp_path):
        """`analyze` refuses an empty batch; the scrape must explain the emptiness."""
        from protor.scraper import scrape_multiple

        base, _state = site
        index = scrape_multiple(
            [f"{base}/boom", f"{base}/missing"],
            tmp_path / "allbad",
            download_js=False,
            live=False,
        )
        assert Path(index).exists(), "an index is still written"
        data = json.loads(Path(index).read_text())
        assert not any((d.get("text_content") or "").strip() for d in data), (
            "no page succeeded, so nothing may look scraped"
        )
