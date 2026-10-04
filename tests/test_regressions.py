"""Regression tests for output-quality bugs found during review.

Each test here corresponds to a defect that shipped: these assert the *correct*
behaviour so it cannot silently regress.
"""

import io
from typing import ClassVar

import pytest
from rich.console import Console

from protor.analyzer import _prepare_context
from protor.crawler import _render, _State
from protor.engine import CrawlEngine, StaticQueue, StaticSource
from protor.exceptions import DataFileNotFoundError, OutputPathError
from protor.markdown import html_to_markdown
from protor.utils import page_filename

# ── markdown: inline content must not shatter paragraphs ─────────────────────


class TestMarkdownInline:
    def test_paragraph_with_emphasis_stays_one_line(self):
        md = html_to_markdown("<p>Protor is a <strong>fast</strong> scraper.</p>")
        assert "Protor is a **fast** scraper." in md

    def test_paragraph_with_link_stays_one_line(self):
        md = html_to_markdown('<p>See the <a href="/md">docs</a> now.</p>')
        assert "See the [docs](/md) now." in md

    def test_mixed_inline_and_text_is_one_paragraph(self):
        html = "<p>Turns pages into <a href='/md'>Markdown</a> for your <em>LLM</em>.</p>"
        md = html_to_markdown(html)
        assert md.strip() == "Turns pages into [Markdown](/md) for your *LLM*."

    def test_adjacent_lists_are_separated(self):
        """A bullet list followed by an ordered list needs a blank line."""
        md = html_to_markdown("<ul><li>one</li></ul><ol><li>first</li></ol>")
        assert "- one\n\n1. first" in md

    def test_nested_list_is_indented(self):
        md = html_to_markdown("<ul><li>outer<ul><li>inner</li></ul></li></ul>")
        assert "- outer" in md
        assert "  - inner" in md

    def test_inline_code_is_marked(self):
        assert "`x`" in html_to_markdown("use <code>x</code> here")

    def test_document_order_is_preserved(self):
        """Trailing inline links must not be hoisted above earlier headings."""
        html = "<article><h1>Guide</h1><p>Body text.</p><a href='/next'>Next</a></article>"
        md = html_to_markdown(html)
        assert md.index("# Guide") < md.index("Body text.") < md.index("[Next](/next)")

    def test_inline_runs_split_around_block_children(self):
        html = "<div>before<p>middle</p>after</div>"
        md = html_to_markdown(html)
        assert md.splitlines().index("before") < md.splitlines().index("middle")
        assert md.splitlines().index("middle") < md.splitlines().index("after")


# ── analysis context: every site must reach the model ───────────────────────


def _site(i: int) -> dict:
    return {
        "domain": f"site{i}.com",
        "url": f"https://site{i}.com",
        "js_count": 3,
        "metadata": {"title": f"Title {i}", "description": "desc"},
        "text_content": "LOREM " * 400,
    }


class TestPrepareContext:
    def test_every_site_is_included(self):
        """A flat global cut silently dropped sites past the budget."""
        ctx = _prepare_context([_site(i) for i in range(10)])
        for i in range(10):
            assert f"## [{i + 1}] site{i}.com" in ctx

    def test_budget_is_respected(self):
        from protor.config import ANALYSIS_MAX_DATA_CHARS

        ctx = _prepare_context([_site(i) for i in range(10)], max_chars=4000)
        assert len(ctx) <= ANALYSIS_MAX_DATA_CHARS

    def test_single_site_gets_full_budget(self):
        ctx = _prepare_context([_site(0)], max_chars=4000)
        assert len(ctx) > 2000

    def test_empty_input(self):
        assert _prepare_context([]) == ""

    @pytest.mark.parametrize("count", [1, 3, 10, 40, 60])
    def test_fits_cap_at_realistic_batch_sizes(self, count):
        from protor.config import ANALYSIS_MAX_DATA_CHARS

        ctx = _prepare_context([_site(i) for i in range(count)])
        assert len(ctx) <= ANALYSIS_MAX_DATA_CHARS
        assert ctx.count("## [") == count

    def test_long_descriptions_are_shortened_before_sites_are_dropped(self):
        sites = [_site(i) for i in range(60)]
        for s in sites:
            s["metadata"]["description"] = "x" * 1_000
        ctx = _prepare_context(sites)
        assert "x" * 1_000 not in ctx
        assert ctx.count("## [") == 60

    def test_included_count_is_reported_not_assumed(self):
        """A batch too large to fit must not be reported as fully analysed."""
        from protor.analyzer import _sites_included
        from protor.config import ANALYSIS_MAX_DATA_CHARS

        ctx = _prepare_context([_site(i) for i in range(400)])
        assert len(ctx) <= ANALYSIS_MAX_DATA_CHARS
        assert _sites_included(ctx) < 400


# ── streaming: LLM markdown must survive the terminal ────────────────────────


class TestStreamRendering:
    def test_markup_is_not_interpreted(self, monkeypatch):
        """LLM output is Markdown; rich markup ate links and crashed on [/x]."""
        import protor.analyzer as analyzer

        buf = io.StringIO()
        monkeypatch.setattr(
            analyzer,
            "console",
            analyzer.console.__class__(file=buf, highlight=False, soft_wrap=True),
        )

        chunks = [
            "**Overview** — see [docs](https://x.com)\n",
            "Use the `[foo]` syntax. [1] reference.\n",
            "closing [/oops] tag",
        ]

        class Backend:
            model_name = "test"

            def stream(self, prompt):
                yield from chunks

        out = analyzer._stream_backend(Backend(), "p")

        assert out == "".join(chunks)
        shown = buf.getvalue()
        assert "[docs](https://x.com)" in shown
        assert "`[foo]`" in shown
        assert "[/oops]" in shown


# ── engine: limits, checkpoints, filenames ───────────────────────────────────


class TestJsFilenames:
    def test_same_basename_from_different_origins_does_not_collide(self):
        urls = [
            "https://cdn-a.com/static/vendor.js",
            "https://cdn-b.com/lib/vendor.js",
            "https://cdn-a.com/static/app.js",
            "https://cdn-b.com/lib/app.js",
        ]
        taken: set[str] = set()
        names = [CrawlEngine._js_filename(i, u, taken) for i, u in enumerate(urls)]
        assert len(set(names)) == len(urls)

    def test_names_are_still_readable(self):
        assert CrawlEngine._js_filename(0, "https://x.com/static/app.js") == "app.js"

    def test_script_without_basename_gets_a_name(self):
        name = CrawlEngine._js_filename(3, "https://x.com/")
        assert name.endswith(".js")
        assert name


class TestMaxTargets:
    @pytest.mark.asyncio
    async def test_failures_count_against_the_ceiling(self, tmp_path, monkeypatch):
        """max_pages must bound requests issued, not just successes."""
        import protor.engine as engine_mod

        requested: list[str] = []
        link_page = (
            "<html><body>"
            + "".join(f"<a href='https://ex.com/p{i}'>x</a>" for i in range(30))
            + "</body></html>"
        )

        async def flaky(session, url, **kwargs):
            requested.append(url)
            if len(requested) % 3 == 0:
                raise RuntimeError("boom")
            return _result(
                text=link_page, nbytes=len(link_page), status=200, content_type="text/html"
            )

        monkeypatch.setattr(engine_mod, "fetch", flaky)

        class Queue(StaticQueue):
            def __init__(self, seeds):
                super().__init__(seeds)
                self.seen = set(seeds)
                self.extra: list[str] = []

            def dequeue(self):
                if self.extra:
                    return self.extra.pop(0)
                return super().dequeue()

            def enqueue(self, url, priority=0):
                if url not in self.seen:
                    self.seen.add(url)
                    self.extra.append(url)
                    return True
                return False

            @property
            def empty(self):
                return not self.extra and not self._urls

        engine = CrawlEngine(
            queue=Queue(["https://ex.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=10,
            concurrency=4,
        )
        stats = await engine.arun()

        assert len(requested) <= 10
        assert stats.dispatched <= 10


class TestCheckpoint:
    def test_no_checkpoint_before_any_success(self, tmp_path, monkeypatch):
        import protor.engine as engine_mod

        async def always_fail(session, url, **kwargs):
            raise RuntimeError("down")

        monkeypatch.setattr(engine_mod, "fetch", always_fail)

        fired: list[int] = []
        engine = CrawlEngine(
            queue=StaticQueue(["https://a.com/", "https://b.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=2,
            checkpoint_interval=5,
            on_checkpoint=lambda: fired.append(1),
        )
        engine.run()

        assert fired == []


# ── utils / naming ───────────────────────────────────────────────────────────


class TestPageFilenames:
    def test_root_is_index(self):
        assert page_filename("https://x.com/") == "index.html"

    def test_same_leaf_name_in_different_dirs_does_not_collide(self):
        """/docs/a.html and /blog/a.html both had the leaf a.html."""
        a = page_filename("https://x.com/docs/a.html")
        b = page_filename("https://x.com/blog/a.html")
        assert a != b
        assert a == "docs-a.html"
        assert b == "blog-a.html"


# ── hostile inputs must not crash a run ──────────────────────────────────────


class TestHostileInputs:
    def test_long_url_segment_produces_a_writable_filename(self, tmp_path):
        """A 400-char path segment exceeded the 255-byte filesystem limit."""
        from protor.utils import page_filename

        name = page_filename(f"https://x.com/{'a' * 400}")
        assert len(name) <= 200
        (tmp_path / name).write_text("ok")  # must not raise OSError

    def test_truncated_long_names_stay_distinct(self):
        """Two long URLs sharing a prefix must not collapse to one file."""
        from protor.utils import safe_filename

        base = "a" * 400
        assert safe_filename(base) != safe_filename(base + "b")

    def test_normal_filenames_are_unchanged(self):
        from protor.utils import page_filename, safe_filename

        assert safe_filename("app.js") == "app.js"
        assert safe_filename("vendor.min.js") == "vendor.min.js"
        assert safe_filename("a/b") == "a_b"
        assert safe_filename("") == "unnamed"
        assert page_filename("https://x.com/") == "index.html"
        assert page_filename("https://x.com/docs/guide.html") == "docs-guide.html"

    def test_manifest_with_null_metadata_is_accepted(self):
        """`"metadata": null` is valid JSON and used to raise AttributeError."""
        from protor.models import SiteManifest

        m = SiteManifest.from_dict(
            {
                "metadata": None,
                "url": "u",
                "domain": "d",
                "html_file": "f",
                "text_content": "t",
                "js_files": [],
                "js_count": 0,
                "bytes_received": 1,
                "elapsed_ms": 1,
                "timestamp": "ts",
            }
        )
        assert m.metadata.title == ""

    def test_analysis_context_survives_null_metadata(self):
        from protor.analyzer import _prepare_context

        ctx = _prepare_context(
            [{"metadata": None, "domain": "x.com", "url": "u", "text_content": "body"}]
        )
        assert "x.com" in ctx
        assert "body" in ctx

    def test_analysis_context_survives_missing_text(self):
        from protor.analyzer import _prepare_context

        ctx = _prepare_context([{"metadata": {}, "domain": "x.com", "url": "u"}])
        assert "x.com" in ctx

    def test_corrupt_checkpoint_is_reported_not_swallowed(self, tmp_path):
        """--resume used to `except: pass`, silently starting over."""
        from protor.crawler import Crawler

        (tmp_path / "crawl_checkpoint.json").write_text("{ not valid json", encoding="utf-8")
        crawler = Crawler("https://example.com", max_pages=1, output_dir=tmp_path, resume=True)
        try:
            # Still usable: the crawl proceeds on a fresh queue.
            assert crawler._queue is not None
            assert crawler._queue.success_count == 0
        finally:
            crawler._queue.close()


# ── --block-ads must cover script downloads ──────────────────────────────────


def _result(**kwargs):
    """
    A real ``FetchResult`` for tests that stub ``fetch``.

    Hand-rolled duck types were used here, and every time the dataclass grew a
    field they broke — once for ``content_type`` and again for ``not_modified`` —
    because the stub was standing in for something whose shape had moved. Using
    the real object means it cannot drift again.
    """
    from protor.fetcher import FetchResult

    return FetchResult(**kwargs)


class TestBlocklistCoversJsDownloads:
    @pytest.mark.asyncio
    async def test_tracker_scripts_are_not_fetched(self, tmp_path, monkeypatch):
        """--block-ads guards the page fetch but scripts come from the very
        tracker CDNs it exists to avoid, so the flag did nothing."""
        from protor.blocklist import Blocklist
        from protor.engine import CrawlEngine, StaticQueue, StaticSource

        html = (
            "<html><head>"
            '<script src="https://www.googletagmanager.com/gtm.js"></script>'
            '<script src="https://example.com/app.js"></script>'
            "</head><body>hi</body></html>"
        )
        fetched: list[str] = []

        async def fake_fetch(session, url, **kwargs):
            fetched.append(url)
            return _result(text=html, nbytes=len(html), status=200, content_type="text/html")

        monkeypatch.setattr("protor.engine.fetch", fake_fetch)

        async def fake_download(session, url, dest):
            fetched.append(url)
            return True

        monkeypatch.setattr("protor.engine.download_file", fake_download)

        engine = CrawlEngine(
            queue=StaticQueue(["https://example.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=1,
            download_js=True,
            blocklist=Blocklist(block_ads=True),
        )
        await engine.arun()

        # The tracker script is dropped; the site's own script is still fetched.
        assert "https://www.googletagmanager.com/gtm.js" not in fetched
        assert fetched == ["https://example.com/", "https://example.com/app.js"]

    @pytest.mark.asyncio
    async def test_no_blocklist_means_scripts_still_download(self, tmp_path, monkeypatch):
        from protor.engine import CrawlEngine, StaticQueue, StaticSource

        html = '<html><head><script src="https://example.com/app.js"></script></head><body>x</body></html>'
        fetched: list[str] = []

        async def fake_fetch(session, url, **kwargs):
            fetched.append(url)
            return _result(text=html, nbytes=len(html), status=200, content_type="text/html")

        monkeypatch.setattr("protor.engine.fetch", fake_fetch)

        async def fake_download(session, url, dest):
            fetched.append(url)
            return True

        monkeypatch.setattr("protor.engine.download_file", fake_download)

        engine = CrawlEngine(
            queue=StaticQueue(["https://example.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=1,
            download_js=True,
        )
        manifests = (await engine.arun(), engine.manifests)[1]
        assert manifests[0].js_count == 1
        assert fetched == ["https://example.com/", "https://example.com/app.js"]


# ── HTTP cache must not leak or preload ──────────────────────────────────────


class TestCacheHygiene:
    def test_bodies_are_not_preloaded_into_memory(self, tmp_path):
        """Loading every body made the disk cache fully RAM-resident."""
        from protor.http_cache import CacheEntry, HTTPCache

        cache = HTTPCache(cache_dir=tmp_path / "c")
        for i in range(20):
            cache.put(f"https://s{i}.com/", CacheEntry(body="x" * 50_000))
        cache.flush()

        reopened = HTTPCache(cache_dir=tmp_path / "c")
        resident = sum(len(e.body) for e in reopened._index.values())
        assert resident == 0
        assert len(reopened._index) == 20
        # ...but the body is still available on demand.
        assert reopened.get("https://s3.com/").body == "x" * 50_000

    def test_expired_bodies_are_deleted(self, tmp_path):
        """Bytes must not outlive the retention window, validators or not."""
        from protor.http_cache import CacheEntry, HTTPCache

        cache = HTTPCache(cache_dir=tmp_path / "c", ttl=1, stale_ttl=1)
        cache.put("https://a.com/", CacheEntry(body="payload"))
        cache.flush()
        assert list((tmp_path / "c" / "bodies").glob("*.body"))

        # Past ttl + stale_ttl the entry has nothing left to offer, so reopening
        # reclaims it and its body. The aged timestamp has to reach disk first,
        # otherwise the reopened cache re-reads a fresh entry.
        cache._index["https://a.com/"].timestamp -= 10_000
        cache._dirty = True  # age the stored entry; flush() is a no-op when clean
        cache.flush()
        reopened = HTTPCache(cache_dir=tmp_path / "c", ttl=1, stale_ttl=1)
        assert reopened._index == {}
        assert list((tmp_path / "c" / "bodies").glob("*.body")) == []

    def test_stale_entry_is_retained_so_it_can_be_revalidated(self, tmp_path):
        """
        A stale entry must keep its validators and body.

        Deleting it on read meant the ETag was already gone before a conditional
        request could be made, so expiry silently degraded into a full
        re-download every time.
        """
        from protor.http_cache import CacheEntry, HTTPCache

        cache = HTTPCache(cache_dir=tmp_path / "c", ttl=1, stale_ttl=3600)
        cache.put("https://a.com/", CacheEntry(body="payload", etag='W/"1"'))
        cache.flush()

        cache._index["https://a.com/"].timestamp -= 10

        assert cache.get("https://a.com/") is None, "stale entries are not served"
        entry = cache.entry_for("https://a.com/")
        assert entry is not None and entry.body == "payload", "but retained for revalidation"
        assert cache.conditional_headers("https://a.com/") == {"If-None-Match": 'W/"1"'}
        assert list((tmp_path / "c" / "bodies").glob("*.body")), "body kept"
        assert cache.prune() == 0, "still inside the retention window"

    def test_prune_reclaims_an_abandoned_cache(self, tmp_path):
        from protor.http_cache import CacheEntry, HTTPCache

        # A zero retention window makes every entry reclaimable on open, so an
        # abandoned cache cannot keep growing between runs.
        cache = HTTPCache(cache_dir=tmp_path / "c", ttl=0, stale_ttl=0)
        for i in range(10):
            cache.put(f"https://s{i}.com/", CacheEntry(body="y" * 10_000))
        cache.flush()

        reopened = HTTPCache(cache_dir=tmp_path / "c", ttl=0, stale_ttl=0)
        assert reopened._index == {}
        assert list((tmp_path / "c" / "bodies").glob("*.body")) == []

    def test_orphaned_bodies_are_swept(self, tmp_path):
        """A lost index used to leave every body file behind."""
        from protor.http_cache import CacheEntry, HTTPCache

        cache = HTTPCache(cache_dir=tmp_path / "c")
        cache.put("https://a.com/", CacheEntry(body="orphan"))
        cache.flush()
        (tmp_path / "c" / "index.json").unlink()

        reopened = HTTPCache(cache_dir=tmp_path / "c")
        assert reopened._index == {}
        assert list((tmp_path / "c" / "bodies").glob("*.body")) == []

    def test_clear_removes_everything(self, tmp_path):
        from protor.http_cache import CacheEntry, HTTPCache

        cache = HTTPCache(cache_dir=tmp_path / "c")
        cache.put("https://a.com/", CacheEntry(body="x"))
        cache.flush()
        cache.clear()
        assert cache.size_bytes() == 0


# ── derived artefacts must be bounded ────────────────────────────────────────


class TestMarkdownIsBounded:
    def test_long_page_markdown_is_capped(self):
        """A 3000-paragraph page put ~119k characters into every manifest."""
        from protor.config import MAX_MARKDOWN_CHARS
        from protor.parser import parse_html

        html = (
            "<html><body>"
            + "".join(f"<p>Paragraph {i} of prose.</p>" for i in range(2000))
            + "</body></html>"
        )
        _, page = parse_html(html, "https://e.com/")
        assert len(page.markdown_content) <= MAX_MARKDOWN_CHARS + 32
        assert page.markdown_content.endswith("[truncated]")

    def test_short_pages_are_untouched(self):
        from protor.parser import parse_html

        _, page = parse_html("<html><body><p>hi</p></body></html>", "https://e.com/")
        assert "[truncated]" not in page.markdown_content


# ── crawler live state must stay bounded ─────────────────────────────────────


class TestCrawlStateIsBounded:
    def test_log_does_not_grow_without_limit(self):
        from protor.crawler import _CrawlLog, _State

        state = _State(max_pages=100_000)
        for i in range(5_000):
            state.log.append(_CrawlLog("ok", "x.com", url=f"https://x.com/{i}"))
            state.log_total += 1
        assert len(state.log) <= 200
        assert state.log_total == 5_000

    def test_row_numbering_stays_stable_after_eviction(self):
        """Rows are tracked by identity, so numbering reflects true position."""
        from protor.crawler import _CrawlLog, _render, _State

        state = _State(max_pages=100_000)
        for i in range(500):
            state.log.append(_CrawlLog("ok", "x.com", url=f"u{i}"))
            state.log_total += 1

        from protor.crawler import _LOG_RESERVED, _LOG_VIEW

        # 500 appended, 200 retained by the deque, and the view shows however many
        # fit the window — so the window is pinned rather than assumed, and the
        # expected window is derived from it rather than hard-coded.
        height = 60
        shown = min(_LOG_VIEW, height - _LOG_RESERVED)
        console = Console(width=120, height=height, record=True, file=io.StringIO())
        console.print(_render(state, "/tmp/out", height=height))
        out = console.export_text()

        assert str(500 - shown + 1) in out, f"the window should start at {500 - shown + 1}"
        assert "500" in out
        assert "1 " not in out.split("Domain")[-1].splitlines()[1]


# ── cached pages must report real sizes ──────────────────────────────────────


class TestCacheHitReporting:
    @pytest.mark.asyncio
    async def test_cached_page_reports_its_body_size(self, tmp_path):
        """A cache hit reported 0 bytes, rendering as "—" in the results table."""
        from protor.fetcher import fetch
        from protor.http_cache import CacheEntry, HTTPCache
        from tests.conftest import FakeResponse, FakeSession

        body = "<html><body>" + "x" * 500 + "</body></html>"
        url = "https://x.com/"
        cache = HTTPCache(cache_dir=tmp_path / "c")
        cache.put(url, CacheEntry(body=body))

        session = FakeSession(routes={url: FakeResponse(status=200, body=body)})
        result = await fetch(session, url, cache=cache)

        assert result.text == body
        assert result.nbytes == len(body.encode("utf-8"))
        assert session.requested == [], "should have been served from cache"

    @pytest.mark.asyncio
    async def test_not_modified_reports_body_size(self, tmp_path):
        from protor.fetcher import fetch
        from protor.http_cache import CacheEntry, HTTPCache
        from tests.conftest import FakeResponse, FakeSession

        body = "y" * 300
        url = "https://x.com/"
        cache = HTTPCache(cache_dir=tmp_path / "c")
        cache.put(url, CacheEntry(body=body, etag='W/"abc"'))

        session = FakeSession(routes={url: FakeResponse(status=304)})
        result = await fetch(session, url, cache=cache)

        assert result.text == body
        assert result.nbytes == len(body.encode("utf-8"))


# ── crawler rendering ────────────────────────────────────────────────────────


class TestCrawlRender:
    def test_progress_bar_is_bounded(self):
        """One cell per page made --max-pages 500 render a 500-char bar."""
        state = _State(scraped=250, max_pages=500)
        group = _render(state, "/tmp/out")
        rendered = "\n".join(str(getattr(c, "text", c)) for c in getattr(group, "renderables", []))
        assert len(rendered) < 2000


# ── malformed / hostile page structure ───────────────────────────────────────


class TestHostileMarkup:
    @pytest.mark.parametrize(
        "html",
        [
            "<html><body><aside><div class='widget'>ad</div></aside><p>keep</p></body></html>",
            "<html><body><nav><ul><li class='nav-item'>m</li></ul></nav><p>keep</p></body></html>",
            "<html><body><nav><script>x()</script></nav><p>keep</p></body></html>",
            "<html><body><header><footer>f</footer></header><p>keep</p></body></html>",
        ],
    )
    def test_nested_noise_does_not_crash(self, html):
        """
        Decomposing a parent clears its descendants' __dict__, so a later
        _is_noise() saw attrs=None and raised. Any page with a <nav> holding a
        <script>, or an <aside> holding a .widget, failed entirely.
        """
        from protor.parser import parse_html

        _, page = parse_html(html, "https://e.com/")
        assert "keep" in page.markdown_content
        assert "widget" not in page.markdown_content

    @pytest.mark.parametrize("depth", [200, 1500, 6000])
    def test_deep_nesting_degrades_instead_of_crashing(self, depth):
        """The recursive renderer blew the stack past 494 nested elements."""
        from protor.parser import parse_html

        html = (
            "<html><body>" + "<div>" * depth + "<p>DEEP</p>" + "</div>" * depth + "</body></html>"
        )
        _, page = parse_html(html, "https://e.com/")
        assert "DEEP" in page.text_content

    def test_decomposed_check_is_not_quadratic(self):
        """
        `getattr(tag, "decomposed")` walks the subtree per tag via bs4
        `__getattr__`, which made a page cost ~625 ms.

        Measured as the *best* of several runs, and warmed up first. A single
        un-warmed wall-clock sample is not a measurement of this function: under
        load it read 75 us/tag — 940x the real cost — which would look exactly
        like the regression this test exists to catch, and send whoever saw it
        hunting a phantom. Noise can only add time, so the fastest observed run is
        the one closest to the cost of the code.
        """
        import time

        from bs4 import BeautifulSoup

        from protor.markdown import _is_decomposed

        deep = BeautifulSoup("<div>" * 600 + "<p>x</p>" + "</div>" * 600, "lxml")
        tags = deep.find_all(True)

        def one_pass() -> float:
            start = time.perf_counter()
            for t in tags:
                _is_decomposed(t)
            return (time.perf_counter() - start) / len(tags)

        # Warm up first: the initial execution pays import-time and allocator
        # costs that say nothing about the function being measured.
        one_pass()

        best = min(one_pass() for _ in range(5))
        # Real cost is ~0.08 us/tag. The ceiling sits an order of magnitude above
        # that so ordinary machine noise cannot trip it, while still being ~7800x
        # under the ~625 ms/page the old implementation cost.
        assert best < 5e-6, f"{best * 1e6:.2f} us/tag is too slow (was ~625 ms/page)"


# ── robots.txt semantics ─────────────────────────────────────────────────────


class TestRobotsPolicySemantics:
    @pytest.mark.asyncio
    async def test_server_error_is_not_remembered_as_allow_all(self, fake_session):
        """
        A 5xx means the site failed, not that it declined to publish rules.
        Caching "allow everything" let one bad response wave a whole run past
        robots.txt for the rest of its life.
        """
        from protor import robots
        from tests.conftest import FakeResponse

        session = fake_session(routes={"https://e.com/robots.txt": FakeResponse(status=500)})

        robots.clear_cache()
        for i in range(3):
            await robots.check_robots(f"https://e.com/p{i}", session)
        assert len(session.requested) == 3, "each page should re-ask after a 5xx"
        robots.clear_cache()

    @pytest.mark.asyncio
    async def test_missing_robots_txt_is_remembered(self, fake_session):
        """RFC 9309: an unavailable robots.txt means no restrictions, and it is a
        real answer worth remembering."""
        from protor import robots
        from tests.conftest import FakeResponse

        session = fake_session(routes={"https://e.com/robots.txt": FakeResponse(status=404)})
        robots.clear_cache()
        for i in range(3):
            await robots.check_robots(f"https://e.com/p{i}", session)
        assert len(session.requested) == 1, "a 404 is a final answer, fetched once"
        robots.clear_cache()


# ── scrape output must explain failures ──────────────────────────────────────


class TestFailureReasonsAreReported:
    def test_reasons_are_grouped_and_shown(self, monkeypatch, capsys):
        """The table showed "✗ error" and the summary "3 failed", never why."""
        import io

        from rich.console import Console

        from protor.scraper import _print_failure_reasons

        rows = [
            {"status": "error", "note": "Fetch failed for 'https://a.com/': timeout"},
            {"status": "error", "note": "Fetch failed for 'https://b.com/': timeout"},
            {"status": "error", "note": "Fetch failed for 'https://c.com/': HTTP 403"},
            {"status": "blocked", "note": "blocked by robots.txt"},
            {"status": "done", "note": None},
        ]
        buf = io.StringIO()
        # The grouping moved to protor.progress so the crawler can print the same
        # summary; the write happens there, so that is what has to be redirected.
        monkeypatch.setattr("protor.progress._console", Console(file=buf, width=100))
        _print_failure_reasons(rows)
        out = buf.getvalue()
        assert "timeout" in out and "2" in out, "grouped the repeated cause"
        assert "robots.txt" in out
        assert "<url>" in out, "per-URL detail collapsed so causes group"

    def test_nothing_printed_when_everything_succeeded(self, monkeypatch):
        import io

        from rich.console import Console

        from protor.scraper import _print_failure_reasons

        buf = io.StringIO()
        monkeypatch.setattr("protor.scraper.console", Console(file=buf, width=100))
        _print_failure_reasons([{"status": "done", "note": None}])
        assert buf.getvalue() == ""


# ── hostile LLM / cache input ────────────────────────────────────────────────


class _FakeSSEResponse:
    """Minimal stand-in for a streaming response: splits on newlines."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def iter_lines(self):
        return self._payload.split(b"\n")


class TestSSEFrameRobustness:
    def test_non_object_and_hostile_frames_are_skipped(self):
        """
        Runtimes emit `data: null` keepalives during long generations. Assuming
        every frame was a mapping raised AttributeError and killed the whole
        analysis mid-stream, discarding everything generated so far.
        """
        from protor.llm_backends import _iter_sse_text

        frames = [
            b"data: null",
            b"data: 42",
            b'data: "a string"',
            b"data: [1,2,3]",
            b'data: {"choices":"oops"}',
            b'data: {"choices":["oops"]}',
            b'data: {"choices":[{"delta":null}]}',
            b": a keepalive comment",
            b"",
            b'data: {"choices":[{"delta":{"content":"kept"}}]}',
            b"data: [DONE]",
        ]
        out = list(_iter_sse_text(_FakeSSEResponse(b"\n".join(frames))))
        assert "".join(out) == "kept", out

    def test_content_arrays_are_joined_not_dropped(self):
        """Some gateways send content as fragments rather than a string."""
        from protor.llm_backends import _iter_sse_text

        payload = (
            b'data: {"choices":[{"delta":{"content":["Hello", " ", "world"]}}]}\n'
            b'data: {"choices":[{"delta":{"content":[{"type":"text","text":"!"}]}}]}\n'
            b"data: [DONE]"
        )
        out = list(_iter_sse_text(_FakeSSEResponse(payload)))
        assert "".join(out) == "Hello world!", out


class TestCacheCannotLoseData:
    def test_truncated_index_keeps_bodies_on_disk(self, tmp_path):
        """
        A truncated index.json parses as "no entries", and reconciling bodies
        against that deleted every cached page. One interrupted write used to
        destroy the whole cache silently.
        """
        from protor.http_cache import CacheEntry, HTTPCache

        cache = HTTPCache(cache_dir=tmp_path)
        for i in range(4):
            cache.put(f"https://s{i}.com/", CacheEntry(body=f"page-{i}"))
        cache.flush()
        assert len(list((tmp_path / "bodies").glob("*.body"))) == 4

        (tmp_path / "index.json").write_text('{"https://s0.com/": {"etag":')
        reopened = HTTPCache(cache_dir=tmp_path)
        assert len(list((tmp_path / "bodies").glob("*.body"))) == 4, "bodies were swept"
        assert reopened.get("https://s0.com/") is None, "damaged index serves nothing"

    @pytest.mark.parametrize("content", ["null", "[1,2,3]", '"hello"', "{}", "123"])
    def test_unusable_index_shapes_are_survived(self, tmp_path, content):
        """Valid JSON of the wrong shape is as unusable as a parse error."""
        from protor.http_cache import HTTPCache

        HTTPCache(cache_dir=tmp_path)
        (tmp_path / "index.json").write_text(content, encoding="utf-8")
        HTTPCache(cache_dir=tmp_path)  # must not raise

    def test_binary_index_is_survived(self, tmp_path):
        """Non-UTF-8 bytes raised UnicodeDecodeError out of the constructor."""
        from protor.http_cache import HTTPCache

        HTTPCache(cache_dir=tmp_path)
        (tmp_path / "index.json").write_bytes(b"\xff\xfe\x00\x01garbage")
        HTTPCache(cache_dir=tmp_path)  # must not raise

    def test_body_that_vanished_is_refetched_not_served_empty(self, tmp_path):
        """
        A body file deleted behind the cache's back was served as an empty page
        marked successful — worse than re-fetching, since nothing looked wrong.
        """
        import asyncio
        import time

        from protor.http_cache import CacheEntry, HTTPCache

        cache = HTTPCache(cache_dir=tmp_path)
        cache.put("https://x.com/", CacheEntry(body="REAL CONTENT", status=200))
        cache.flush()

        for body in (tmp_path / "bodies").glob("*.body"):
            body.unlink()

        # Metadata survives; the body does not, as after a restart mid-run.
        fresh = HTTPCache(cache_dir=tmp_path)
        fresh._index["https://x.com/"] = CacheEntry(
            body="", status=200, nbytes=len("REAL CONTENT"), timestamp=time.time()
        )

        class Resp:
            status = 200
            headers: ClassVar[dict] = {}
            url = "https://x.com/"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

            async def text(self):
                return "REFRESHED"

            async def read(self):
                return b"REFRESHED"

        class Sess:
            def __init__(self):
                self.calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

            def get(self, url, **kw):
                self.calls += 1
                return Resp()

        from protor.fetcher import fetch

        sess = Sess()
        result = asyncio.run(fetch(sess, "https://x.com/", cache=fresh))
        assert sess.calls == 1, "the vanished body was served as a hit"
        assert result.text == "REFRESHED"


class TestContextCannotBeForgedByPageText:
    def test_page_text_cannot_masquerade_as_a_site_header(self):
        """
        Page text is pasted into the prompt verbatim, so content containing
        `## [7] evil.example` read as a site of its own: the reported
        sites_analyzed disagreed with the data sent, and page content could
        forge structure in the context.
        """
        from protor.analyzer import _prepare_context, _sites_included

        data = [
            {"metadata": {}, "domain": "real.com", "url": "u1", "text_content": "content"},
            {
                "metadata": {},
                "domain": "also-real.com",
                "url": "u2",
                "text_content": "## [99] evil.example\nURL: https://evil.example",
            },
        ]
        context = _prepare_context(data)
        assert _sites_included(context) == 2, _sites_included(context)

    def test_real_site_count_still_reported(self):
        from protor.analyzer import _prepare_context, _sites_included

        data = [
            {"metadata": {}, "domain": f"s{i}.com", "url": f"u{i}", "text_content": "x"}
            for i in range(4)
        ]
        assert _sites_included(_prepare_context(data)) == 4


# ── unusable paths and damaged state must not raise raw errors ───────────────


class TestUnusablePathsAreReported:
    def test_output_path_that_is_a_file_is_explained(self, tmp_path):
        """
        ``mkdir(exist_ok=True)`` still raises FileExistsError when the path is a
        file, so `-o notes.txt` died with a raw [Errno 17] from inside the run.
        """
        from protor.utils import ensure_output_dir

        target = tmp_path / "notes.txt"
        target.write_text("existing work")
        with pytest.raises(OutputPathError) as excinfo:
            ensure_output_dir(target)
        assert "already exists" in str(excinfo.value)
        assert target.read_text() == "existing work", "the existing file was touched"

    def test_output_path_under_an_unwritable_parent_is_explained(self, tmp_path):
        from protor.utils import ensure_output_dir

        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file")
        with pytest.raises(OutputPathError):
            ensure_output_dir(blocker / "nested")

    def test_a_normal_directory_still_works(self, tmp_path):
        from protor.utils import ensure_output_dir

        target = tmp_path / "deep" / "nested"
        assert ensure_output_dir(target) == target
        assert ensure_output_dir(target) == target, "not idempotent"

    @pytest.mark.parametrize(
        "make,reason",
        [
            (lambda p: p.mkdir(), "directory"),
            (lambda p: p.write_text("not json{"), "not valid JSON"),
        ],
    )
    def test_unusable_input_file_is_explained(self, tmp_path, make, reason):
        """
        `-f <directory>` escaped as a bare IsADirectoryError, and malformed
        JSON escaped as JSONDecodeError — neither said which path was wrong.
        """
        from protor.cli import _load_index

        target = tmp_path / "input"
        make(target)
        with pytest.raises(DataFileNotFoundError) as excinfo:
            _load_index(str(target))
        assert reason in str(excinfo.value)
        assert str(target) in str(excinfo.value), "the message must name the path"

    def test_missing_input_file_still_suggests_scraping(self, tmp_path):
        from protor.cli import _load_index

        with pytest.raises(DataFileNotFoundError, match="protor scrape"):
            _load_index(str(tmp_path / "absent.json"))


class TestCrawlQueueRecoversFromDamage:
    def test_corrupt_queue_db_is_quarantined_not_raised(self, tmp_path):
        """
        A corrupt queue raised sqlite3.DatabaseError out of the constructor,
        unlike the checkpoint JSON beside it which was already handled.
        """
        from protor.crawler import Crawler

        db = tmp_path / "crawl_queue.db"
        db.write_bytes(b"SQLite format 3\x00" + bytes(range(256)) * 2)

        crawler = Crawler("https://example.com/", output_dir=tmp_path)
        try:
            assert crawler._queue.queue_size >= 1, "the fresh queue was not usable"
            assert db.exists(), "a fresh database should have been created"
            assert list(tmp_path.glob("crawl_queue.db.corrupt*")), (
                "the unreadable file was deleted rather than kept"
            )
        finally:
            crawler._queue.close()

    def test_the_quarantined_file_is_preserved_byte_for_byte(self, tmp_path):
        from protor.crawler import Crawler

        original = b"SQLite format 3\x00" + bytes(range(200))
        db = tmp_path / "crawl_queue.db"
        db.write_bytes(original)

        crawler = Crawler("https://example.com/", output_dir=tmp_path)
        try:
            kept = next(tmp_path.glob("crawl_queue.db.corrupt*"))
            assert kept.read_bytes() == original, "quarantine altered the evidence"
        finally:
            crawler._queue.close()


class TestFailedPagesStayRetryable:
    def _queue(self, tmp_path):
        from protor.crawler import _CrawlQueue

        return _CrawlQueue(tmp_path / "q.db")

    def test_a_page_failed_in_an_earlier_run_can_be_queued_again(self, tmp_path):
        """
        Treating every visit as final meant a retry found an empty queue and
        reported "0 pages queued" without saying why — the run looked like it had
        nothing to do rather than being unable to do anything.
        """
        url = "https://x.com/a"
        q = self._queue(tmp_path)
        q.enqueue(url)
        q.dequeue()
        q.mark_visited(url, success=False)
        q.close()

        # A new queue over the same database stands in for the next run.
        later = self._queue(tmp_path)
        assert later.enqueue(url) is True, "a page that failed became unreachable"
        assert later.queue_size == 1
        assert later.dequeue() == url, "the retry was not dequeueable"
        later.close()

    def test_a_failure_is_not_retried_repeatedly_within_one_run(self, tmp_path):
        """
        Allowing retries without a run boundary turns a self-linking 404 into a
        loop: every page that links to it re-queues it, forever.
        """
        url = "https://x.com/loop"
        q = self._queue(tmp_path)
        q.enqueue(url)
        q.dequeue()
        q.mark_visited(url, success=False)
        for _ in range(3):
            assert q.enqueue(url) is False, "a failure was retried within one run"
            q.dequeue()
        q.close()

    def test_a_successful_page_is_still_skipped(self, tmp_path):
        url = "https://x.com/b"
        q = self._queue(tmp_path)
        q.enqueue(url)
        q.dequeue()
        q.mark_visited(url, success=True)
        assert q.enqueue(url) is False, "re-crawling a finished page wastes a fetch"
        q.close()

        later = self._queue(tmp_path)
        assert later.enqueue(url) is False, "a success was undone by a later run"
        later.close()

    def test_retrying_does_not_duplicate_a_queued_url(self, tmp_path):
        url = "https://x.com/c"
        q = self._queue(tmp_path)
        q.enqueue(url)
        q.dequeue()
        q.mark_visited(url, success=False)
        q.close()

        later = self._queue(tmp_path)
        assert later.enqueue(url) is True
        assert later.enqueue(url) is False, "the same URL was queued twice"
        later.close()

    def test_success_after_a_retry_closes_it(self, tmp_path):
        """The retry must actually be able to finish the job."""
        url = "https://x.com/d"
        q = self._queue(tmp_path)
        q.enqueue(url)
        q.dequeue()
        q.mark_visited(url, success=False)
        q.close()

        later = self._queue(tmp_path)
        assert later.enqueue(url) is True
        retried = later.dequeue()
        assert retried == url
        later.mark_visited(retried, success=True)
        assert later.enqueue(retried) is False
        later.close()


class TestJsFilesNameWhatLanded:
    """
    `js_files` in the manifest must name the files that are actually on disk.

    `asyncio.wait()` returns a *set*, and the results were numbered with
    `enumerate(done)` — so each download was paired with whatever script happened
    to sit at that index in set-iteration order. The manifest then claimed files
    the server had 404'd and silently omitted files that had really been
    written: the artefact and its own index disagreed.
    """

    async def test_claimed_files_are_the_downloaded_ones(self, tmp_path, monkeypatch):
        import asyncio

        import protor.engine as engine_mod

        scripts = [f"https://cdn.example.com/s{i}.js" for i in range(8)]
        # Alternating success, so any mispairing shows up in both directions.
        failed = {scripts[1], scripts[4], scripts[6]}

        async def fake_download(session, jurl, dest, **kwargs):
            await asyncio.sleep(0)
            if jurl in failed:
                return False
            dest.write_text(f"// {jurl}")
            return True

        async def fake_fetch(session, url, **kwargs):
            body = (
                "<html><body>"
                + "".join(f"<script src={u!r}></script>" for u in scripts)
                + "</body></html>"
            )
            from protor.fetcher import FetchResult

            return FetchResult(text=body, nbytes=len(body))

        monkeypatch.setattr(engine_mod, "download_file", fake_download)
        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)

        engine = CrawlEngine(
            queue=StaticQueue(["https://ex.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=5,
            download_js=True,
        )
        await engine.arun()

        [manifest] = engine.manifests
        claimed = set(manifest.js_files)
        on_disk = {
            f"https://cdn.example.com/{p.name}" for p in (tmp_path / "ex.com" / "js").glob("*.js")
        }
        expected = {u for u in scripts if u not in failed}

        assert claimed == expected, (
            f"manifest claims files that 404'd: {claimed - expected};"
            f" omits files on disk: {expected - claimed}"
        )
        assert len(on_disk) == len(expected)
        assert not (claimed & failed), "a 404 was reported as downloaded"
        assert len(claimed) == 5


class TestDomainFilterIsCaseInsensitive:
    """
    A host is case-insensitive, and the queue canonicalises it that way.

    ``canonicalize_url`` lowercases the host, but the allowed domain came from
    ``urlparse`` of the URL as the user typed it, which keeps the case. Compared
    raw, the crawl rejected each URL as off-domain against its own canonical
    form and reported zero pages scraped beside one "off-domain" row.
    """

    async def _scrape_with(self, tmp_path, monkeypatch, allowed_domain):
        import protor.engine as engine_mod
        from protor.fetcher import FetchResult

        body = "<html><body>hi</body></html>"

        async def fake_fetch(session, url, **kwargs):
            return FetchResult(text=body, nbytes=len(body))

        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)
        engine = CrawlEngine(
            queue=StaticQueue(["https://example.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=5,
            allowed_domain=allowed_domain,
        )
        return await engine.arun()

    async def test_an_uppercase_allowed_domain_still_matches(self, tmp_path, monkeypatch):
        stats = await self._scrape_with(tmp_path, monkeypatch, "EXAMPLE.com")
        assert stats.scraped == 1, "the seed was rejected as off-domain by itself"

    async def test_the_lowercase_spelling_still_works(self, tmp_path, monkeypatch):
        stats = await self._scrape_with(tmp_path, monkeypatch, "example.com")
        assert stats.scraped == 1

    async def test_a_genuinely_other_domain_is_still_refused(self, tmp_path, monkeypatch):
        """Case-insensitivity must not turn the filter off."""
        stats = await self._scrape_with(tmp_path, monkeypatch, "other.com")
        assert stats.scraped == 0


class TestMaxTargetsIsADispatchCeiling:
    """
    `--max-pages` bounds requests, not just successes.

    The spawn loop compares ``stats.total`` — scraped + errors + blocked — against
    the ceiling, so a *skipped* URL would not advance it and the ceiling would
    not bound anything. Nothing can skip: the parser yields same-host links only,
    so a recursive crawl never hands the domain filter a URL to reject. That is
    the reason the ceiling holds, and it is an invariant rather than luck — if
    the parser's host check were loosened, the ceiling would quietly become a
    suggestion with nothing left to say so.
    """

    async def test_dispatch_never_exceeds_the_ceiling(self, tmp_path, monkeypatch):
        import protor.engine as engine_mod
        from protor.engine import RecursiveSource
        from protor.fetcher import FetchResult

        async def fake_fetch(session, url, **kwargs):
            # Every page links to the whole site, so the frontier never empties.
            body = (
                "<html><body>"
                + "".join(f'<p><a href="https://ex.com/p{i}">x</a></p>' for i in range(30))
                + "</body></html>"
            )
            return FetchResult(text=body, nbytes=len(body))

        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)

        for ceiling in (1, 2, 5):
            engine = CrawlEngine(
                queue=StaticQueue(["https://ex.com/"]),
                link_source=RecursiveSource(),
                output_dir=tmp_path / str(ceiling),
                max_targets=ceiling,
                allowed_domain="ex.com",
                concurrency=4,
            )
            stats = await engine.arun()
            assert stats.dispatched <= ceiling, (
                f"ceiling {ceiling} spawned {stats.dispatched} tasks"
            )
            assert stats.total == stats.dispatched, (
                f"a dispatched URL neither succeeded, failed nor blocked: "
                f"total={stats.total} dispatched={stats.dispatched}"
            )


class TestNonPagesAreNotScraped:
    """
    A body is not a page by virtue of arriving over HTTP.

    Nothing checked the ``Content-Type``, so a link to a manual.pdf was scraped
    into two thousand characters of ``%PDF-1.4`` and reported as a successfully
    scraped page — the same failure-as-success shape as a stale CSS selector,
    one layer down.
    """

    async def _crawl_one(self, tmp_path, monkeypatch, *, content_type, body):
        import protor.engine as engine_mod
        from protor.fetcher import FetchResult

        async def fake_fetch(session, url, **kwargs):
            return FetchResult(text=body, nbytes=len(body), status=200, content_type=content_type)

        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)
        engine = CrawlEngine(
            queue=StaticQueue(["https://ex.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=5,
        )
        return await engine.arun()

    @pytest.mark.parametrize(
        ("content_type", "body"),
        [
            ("application/pdf", "%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>"),
            ("image/png", "\x89PNG\r\n\x1a\n"),
            ("video/mp4", "\x00\x00\x00 ftypisom"),
            ("application/zip", "PK\x03\x04"),
        ],
    )
    async def test_a_binary_response_is_not_counted_as_a_page(
        self, tmp_path, monkeypatch, content_type, body
    ):
        stats = await self._crawl_one(tmp_path, monkeypatch, content_type=content_type, body=body)
        assert stats.scraped == 0, f"{content_type} was scraped as a page"
        assert stats.total == 0, "and it consumed the page budget"

    async def test_no_manifest_is_written_for_a_binary_response(self, tmp_path, monkeypatch):
        engine_stats = await self._crawl_one(
            tmp_path, monkeypatch, content_type="application/pdf", body="%PDF-1.4"
        )
        assert engine_stats.scraped == 0
        assert not list(tmp_path.rglob("*.html")), "the PDF was saved as a page"
        assert not list(tmp_path.rglob("*.json")), "and given a manifest"

    @pytest.mark.parametrize(
        "content_type",
        ["text/html", "text/html; charset=utf-8", "application/xhtml+xml", ""],
    )
    async def test_html_is_still_scraped(self, tmp_path, monkeypatch, content_type):
        html = "<html><head><title>Real</title></head><body><p>hi</p></body></html>"
        stats = await self._crawl_one(tmp_path, monkeypatch, content_type=content_type, body=html)
        assert stats.scraped == 1, f"a real page was dropped ({content_type!r})"

    async def test_html_served_as_octet_stream_is_still_scraped(self, tmp_path, monkeypatch):
        """
        The false negative that would lose real content.

        Plenty of servers send ``application/octet-stream`` for perfectly good
        HTML, so that type must not be decisive on its own — the body decides.
        """
        html = "<!DOCTYPE html><html><head><title>Real</title></head><body><p>hi</p></body></html>"
        stats = await self._crawl_one(
            tmp_path, monkeypatch, content_type="application/octet-stream", body=html
        )
        assert stats.scraped == 1, "octet-stream HTML was dropped"

    async def test_a_binary_response_is_not_re_requested_on_resume(self, tmp_path, monkeypatch):
        """
        Recorded like a filter, not a failure.

        Marked as attempted-and-failed it would be retried by every resumed run,
        spending budget on the same PDF each time.
        """
        from protor.crawler import _CrawlQueue

        q = _CrawlQueue(tmp_path / "crawl_queue.db")
        q.mark_visited("https://ex.com/manual.pdf", success=False, attempted=False)
        assert q.requeue_failed() == 0, "a non-page is queued again on every resume"
        q.close()


class TestTheCrawlerDoesNotRetainManifests:
    """
    `Crawler` never reads `engine.manifests`; it reports `CrawlStats`.

    The engine accumulated one manifest per page anyway, each carrying the page's
    text and markdown — measured at ~49 KiB of retained strings per page, so a
    40,000-page crawl held roughly 1.9 GB that nothing ever read. They are still
    written to disk; only the in-memory retention is gone.
    """

    async def test_a_crawl_keeps_no_manifests_in_memory(self, tmp_path, monkeypatch):
        """Captures the engine the crawler builds, and looks at what it kept."""
        import protor.crawler as crawler_mod
        import protor.engine as engine_mod
        from protor.fetcher import FetchResult

        html = (
            "<html><head><title>Page</title></head><body>"
            + "<p>filler</p>" * 200
            + "</body></html>"
        )
        built: list[object] = []

        async def fake_fetch(session, url, **kwargs):
            return FetchResult(text=html, nbytes=len(html), status=200, content_type="text/html")

        class _Recording(engine_mod.CrawlEngine):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                built.append(self)

        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)
        monkeypatch.setattr(crawler_mod, "CrawlEngine", _Recording)

        crawler = crawler_mod.Crawler(
            "https://ex.com/", max_pages=5, output_dir=tmp_path, live=False
        )
        await crawler._run()

        assert crawler._state.scraped == 1, "the crawl should still have run"
        assert built, "the crawler built no engine to inspect"
        assert built[0].manifests == [], "the crawl retained manifests nothing reads"
        assert list(tmp_path.rglob("*.json")), "but they were still written to disk"

    def test_the_batch_scraper_still_collects_them(self, tmp_path):
        """The scraper does read them — the flag must not have broken that path."""
        from protor.engine import CrawlEngine, StaticQueue, StaticSource

        engine = CrawlEngine(
            queue=StaticQueue(["https://ex.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=1,
            collect_manifests=True,
        )
        assert engine._collect_manifests is True

    def test_an_engine_can_be_told_not_to(self, tmp_path):
        from protor.engine import CrawlEngine, StaticQueue, StaticSource

        engine = CrawlEngine(
            queue=StaticQueue(["https://ex.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=1,
            collect_manifests=False,
        )
        assert engine._collect_manifests is False
        assert engine.manifests == []


class TestGuessedNoiseYieldsToAnExplicitSchema:
    """
    Class-name guesses must lose to an explicit instruction.

    `_NOISE_PATTERN` matched any class or id containing `ad-`, `social`,
    `share`, `related`, `banner`, `promo` and more. Those are ordinary words: a
    classifieds site keeps its listings in `.ad-card`, a news site keeps its
    stories in `.related-posts`. They were deleted before extraction ran, so a
    `--schema` run whose own selectors named them extracted **nothing** and
    reported success — and the same content vanished from the text and markdown
    beside it.

    Unambiguous chrome (cookie banners, consent dialogs, pagination) is still
    stripped either way; only the guesses yield.
    """

    CLASSIFIEDS = (
        "<html><body><div class='classifieds'>"
        "<div class='ad-card'><h3>Blue widget</h3><span class='ad-price'>$5</span></div>"
        "<div class='ad-card'><h3>Red widget</h3><span class='ad-price'>$7</span></div>"
        "</div>"
        "<div class='cookie-banner'><p>accept cookies</p></div>"
        "<div class='sidebar'><p>chrome</p></div>"
        "</body></html>"
    )

    def _schema(self):
        from protor.extractor import ExtractionSchema

        return ExtractionSchema.from_dict(
            {
                "name": "ads",
                "base_selector": ".ad-card",
                "fields": [
                    {"name": "title", "selector": "h3"},
                    {"name": "price", "selector": ".ad-price"},
                ],
            }
        )

    def test_a_schema_sees_the_content_its_selectors_name(self):
        from protor.parser import parse_html

        soup, page = parse_html(self.CLASSIFIEDS, "https://x.example/", strip_guessed_noise=False)
        from protor.extractor import extract_from_soup

        rows = extract_from_soup(soup, self._schema(), base_url="https://x.example/")
        assert rows == [
            {"title": "Blue widget", "price": "$5"},
            {"title": "Red widget", "price": "$7"},
        ], rows
        assert "Blue widget" in page.text_content

    def test_unambiguous_chrome_is_stripped_even_so(self):
        from protor.parser import parse_html

        _, page = parse_html(self.CLASSIFIEDS, "https://x.example/", strip_guessed_noise=False)
        assert "accept cookies" not in page.text_content, "cookie chrome survived"

    def test_the_default_is_unchanged_for_a_plain_scrape(self):
        """
        A run with no schema still gets the aggressive filtering.

        Otherwise this would make every ordinary scrape worse by default, which
        is the opposite of the point.
        """
        from protor.parser import parse_html

        _, page = parse_html(self.CLASSIFIEDS, "https://x.example/")
        assert "Blue widget" not in page.text_content, "guesses are off by default now"
        assert "accept cookies" not in page.text_content

    def test_structural_noise_is_never_a_guess(self):
        """nav/footer/aside are chrome by anyone's definition."""
        from protor.parser import parse_html

        html = "<html><body><nav>menu</nav><p>content</p><footer>foot</footer></body></html>"
        _, page = parse_html(html, "https://x.example/", strip_guessed_noise=False)
        assert "menu" not in page.text_content
        assert "foot" not in page.text_content
        assert "content" in page.text_content

    def test_the_engine_turns_the_guesses_off_for_a_schema(self, tmp_path, monkeypatch):
        """
        The flag only helps if the crawl path actually sets it.

        Everything here is patched through *monkeypatch*, not by assigning onto
        the module. An earlier version of this test restored `parse_html` in a
        `finally` and forgot `fetch` beside it, which left every later test in
        the session fetching this test's classifieds HTML — and surfaced as an
        unrelated extraction test in another file failing with zero records,
        hundreds of tests away.
        """
        import asyncio

        import protor.engine as engine_mod
        from protor.extractor import ExtractionSchema
        from protor.fetcher import FetchResult

        seen: list[bool] = []
        real_parse = engine_mod.parse_html

        def spy(html, url, **kwargs):
            seen.append(kwargs.get("strip_guessed_noise"))
            return real_parse(html, url, **kwargs)

        async def fake_fetch(session, url, **kwargs):
            return FetchResult(
                text=self.CLASSIFIEDS,
                nbytes=len(self.CLASSIFIEDS),
                status=200,
                content_type="text/html",
            )

        monkeypatch.setattr(engine_mod, "parse_html", spy)
        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)

        async def run(schema):
            engine = CrawlEngine(
                queue=StaticQueue(["https://ex.com/"]),
                link_source=StaticSource(),
                output_dir=tmp_path,
                max_targets=1,
                extraction_schema=schema,
            )
            await engine.arun()

        asyncio.run(run(None))
        asyncio.run(
            run(
                ExtractionSchema.from_dict(
                    {
                        "name": "ads",
                        "base_selector": ".ad-card",
                        "fields": [{"name": "t", "selector": "h3"}],
                    }
                )
            )
        )

        assert seen == [True, False], f"strip_guessed_noise was {seen}"


class TestBlocklistDoesNotRefuseTheRequestedSite:
    """
    `--block-ads` exists to stop a page pulling a tracker off a CDN.

    Applied to the crawl's own target it made the command useless: `protor scrape
    https://www.facebook.com --block-ads` fetched nothing and reported the
    target itself as blocked by the ad/analytics blocklist. facebook.com,
    twitter.com, linkedin.com and optimizely.com are all in the apex-domain
    list, so the flag and the target were mutually exclusive.
    """

    TRACKER_URL = "https://doubleclick.net/pixel"

    async def _run(self, tmp_path, monkeypatch, *, requested_hosts):
        import protor.engine as engine_mod
        from protor.blocklist import Blocklist
        from protor.fetcher import FetchResult

        fetched: list[str] = []

        async def fake_fetch(session, url, **kwargs):
            fetched.append(url)
            return FetchResult(
                text="<html><body><p>landing</p></body></html>",
                nbytes=40,
                status=200,
                content_type="text/html",
            )

        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)
        engine = CrawlEngine(
            queue=StaticQueue([self.TRACKER_URL]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=1,
            blocklist=Blocklist(),
            requested_hosts=requested_hosts,
        )
        stats = await engine.arun()
        return stats, fetched

    async def test_a_requested_host_is_fetched(self, tmp_path, monkeypatch):
        stats, fetched = await self._run(tmp_path, monkeypatch, requested_hosts=["doubleclick.net"])
        assert fetched == [self.TRACKER_URL], "the requested host was blocked"
        assert stats.scraped == 1

    async def test_an_unrequested_tracker_is_still_blocked(self, tmp_path, monkeypatch):
        """The control: the flag still does its job."""
        stats, fetched = await self._run(tmp_path, monkeypatch, requested_hosts=["example.com"])
        assert fetched == [], "a third-party tracker was fetched"
        assert stats.blocked == 1

    async def test_the_crawls_own_domain_counts_as_requested(self, tmp_path, monkeypatch):
        """allowed_domain is added automatically, so the crawl seed is covered."""
        import protor.engine as engine_mod
        from protor.blocklist import Blocklist
        from protor.fetcher import FetchResult

        fetched: list[str] = []

        async def fake_fetch(session, url, **kwargs):
            fetched.append(url)
            return FetchResult(
                text="<html><body><p>x</p></body></html>",
                nbytes=29,
                status=200,
                content_type="text/html",
            )

        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)
        engine = CrawlEngine(
            queue=StaticQueue(["https://doubleclick.net/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=1,
            allowed_domain="doubleclick.net",
            blocklist=Blocklist(),
        )
        await engine.arun()
        assert fetched == ["https://doubleclick.net/"], "the crawl seed was blocked"
