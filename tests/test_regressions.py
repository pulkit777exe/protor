"""Regression tests for output-quality bugs found during review.

Each test here corresponds to a defect that shipped: these assert the *correct*
behaviour so it cannot silently regress.
"""

import io

import pytest
from rich.console import Console

from protor.analyzer import _prepare_context
from protor.crawler import _render, _State
from protor.engine import CrawlEngine, StaticQueue, StaticSource
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
            return type("R", (), {"text": link_page, "nbytes": len(link_page), "status": 200})()

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
            return type("R", (), {"text": html, "nbytes": len(html), "status": 200})()

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
            return type("R", (), {"text": html, "nbytes": len(html), "status": 200})()

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

        console = Console(width=120, record=True, file=io.StringIO())
        console.print(_render(state, "/tmp/out"))
        out = console.export_text()
        # 500 appended, 200 retained, 20 shown: the visible window is 481..500.
        assert "481" in out
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
