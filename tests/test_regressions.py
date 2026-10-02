"""Regression tests for output-quality bugs found during review.

Each test here corresponds to a defect that shipped: these assert the *correct*
behaviour so it cannot silently regress.
"""

import io

import pytest

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


# ── crawler rendering ────────────────────────────────────────────────────────


class TestCrawlRender:
    def test_progress_bar_is_bounded(self):
        """One cell per page made --max-pages 500 render a 500-char bar."""
        state = _State(scraped=250, max_pages=500)
        group = _render(state, "/tmp/out")
        rendered = "\n".join(str(getattr(c, "text", c)) for c in getattr(group, "renderables", []))
        assert len(rendered) < 2000
