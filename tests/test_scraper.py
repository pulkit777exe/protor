"""Tests for protor.scraper HTML-parsing helpers and the site-index write."""

from __future__ import annotations

import json
import os
import time
import tracemalloc
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from protor.engine import CrawlStats
from protor.http_cache import _SWEEP_MARKER, CacheEntry, HTTPCache
from protor.markdown import clean_soup
from protor.models import SiteManifest, SiteMetadata
from protor.parser import (
    _extract_js_links,
    _extract_metadata,
    _extract_text,
    extract_links,
    parse_html,
)
from protor.scraper import _write_manifest_index
from protor.utils import save_json
from tests.conftest import EMPTY_HTML, SIMPLE_HTML


class TestExtractMetadata:
    def test_title(self):
        soup = BeautifulSoup(SIMPLE_HTML, "lxml")
        m = _extract_metadata(soup)
        assert m.title == "Test Site"

    def test_description(self):
        soup = BeautifulSoup(SIMPLE_HTML, "lxml")
        assert _extract_metadata(soup).description == "A test description."

    def test_keywords_split(self):
        soup = BeautifulSoup(SIMPLE_HTML, "lxml")
        kw = _extract_metadata(soup).keywords
        assert "test" in kw
        assert "python" in kw
        assert "scraper" in kw

    def test_author(self):
        soup = BeautifulSoup(SIMPLE_HTML, "lxml")
        assert _extract_metadata(soup).author == "Pulkit"

    def test_og_tag(self):
        soup = BeautifulSoup(SIMPLE_HTML, "lxml")
        og = _extract_metadata(soup).og_tags
        assert og.get("og:title") == "Test OG Title"

    def test_empty_html(self):
        soup = BeautifulSoup(EMPTY_HTML, "lxml")
        m = _extract_metadata(soup)
        assert m.title == ""
        assert m.keywords == []

    def test_missing_title_tag(self):
        html = "<html><body><p>No title here.</p></body></html>"
        soup = BeautifulSoup(html, "lxml")
        assert _extract_metadata(soup).title == ""


class TestExtractJsLinks:
    def test_finds_relative_and_absolute(self):
        soup = BeautifulSoup(SIMPLE_HTML, "lxml")
        links = _extract_js_links(soup, "https://example.com")
        assert "https://example.com/static/app.js" in links
        assert "https://cdn.example.com/lib.js" in links

    def test_deduplicates(self):
        html = '<script src="/a.js"></script><script src="/a.js"></script>'
        soup = BeautifulSoup(html, "lxml")
        links = _extract_js_links(soup, "https://example.com")
        assert links.count("https://example.com/a.js") == 1

    def test_empty(self):
        soup = BeautifulSoup(EMPTY_HTML, "lxml")
        assert _extract_js_links(soup, "https://example.com") == []

    def test_ignores_inline_scripts(self):
        html = "<script>console.log('inline')</script>"
        soup = BeautifulSoup(html, "lxml")
        assert _extract_js_links(soup, "https://example.com") == []


class TestExtractLinks:
    def test_returns_internal_links(self):
        links = extract_links(SIMPLE_HTML, "https://example.com")
        assert "https://example.com/about" in links
        assert "https://example.com/contact" in links

    def test_excludes_external_links(self):
        links = extract_links(SIMPLE_HTML, "https://example.com")
        assert not any("external.com" in link for link in links)

    def test_deduplicates(self):
        html = '<a href="/page">A</a><a href="/page">B</a>'
        links = extract_links(html, "https://example.com")
        assert links.count("https://example.com/page") == 1

    def test_strips_fragments(self):
        html = '<a href="/page#section">Link</a>'
        links = extract_links(html, "https://example.com")
        assert "https://example.com/page" in links
        assert not any("#" in link for link in links)

    def test_empty_html(self):
        assert extract_links(EMPTY_HTML, "https://example.com") == []


class TestExtractText:
    def _clean(self, html):
        """_extract_text reads an already-filtered tree; parse_soup cleans once."""
        soup = BeautifulSoup(html, "lxml")
        clean_soup(soup)
        return soup

    def test_removes_nav_footer_scripts(self):
        text = _extract_text(self._clean(SIMPLE_HTML))
        assert "Navigation" not in text
        assert "Footer text" not in text
        assert "console.log" not in text

    def test_includes_main_content(self):
        text = _extract_text(self._clean(SIMPLE_HTML))
        assert "Hello World" in text
        assert "main content" in text

    def test_truncates_long_content(self):
        long_html = "<p>" + ("x " * 10_000) + "</p>"
        text = _extract_text(self._clean(long_html))
        assert len(text) <= 10_000

    def test_empty_html_returns_empty(self):
        text = _extract_text(self._clean(EMPTY_HTML))
        assert text.strip() == ""


class TestOneWalkPerPage:
    """
    Links, JS references and metadata used to be four separate searches.

    Each find_all builds and runs its own SoupStrainer, which cost more than the
    tags it selected: instrumenting one page showed the selectors
    {'a': 1, 'script': 1, True: 1, 'meta': 1} plus soup.title, and
    _extract_internal_links alone was 4.0 ms of the 30 ms a page took.
    """

    @pytest.fixture
    def counted(self, monkeypatch):
        """Record every find_all selector parse_html triggers."""
        calls: list[tuple] = []
        real = BeautifulSoup.find_all

        def spy(self, name=None, attrs=None, **kwargs):
            calls.append((name, attrs, kwargs.get("recursive", True)))
            return real(self, name, attrs, **kwargs)

        monkeypatch.setattr(BeautifulSoup, "find_all", spy)
        return calls

    def test_parse_html_makes_no_tag_searches(self, counted):
        parse_html(SIMPLE_HTML, "https://example.com/")
        assert counted == [], f"expected a single walk, got selectors {counted}"

    def test_page_artefacts_are_all_collected(self):
        """One walk must still find everything the four separate ones did."""
        _, page = parse_html(SIMPLE_HTML, "https://example.com/")

        assert page.metadata.title == "Test Site"
        assert page.metadata.description == "A test description."
        assert page.metadata.author == "Pulkit"
        assert page.metadata.og_tags["og:title"] == "Test OG Title"
        assert "test" in page.metadata.keywords

        assert page.links == ["https://example.com/about", "https://example.com/contact"]
        assert page.js_links == [
            "https://example.com/static/app.js",
            "https://cdn.example.com/lib.js",
        ]

    def test_navigation_links_are_collected_before_noise_is_removed(self):
        """
        The single pass runs over the raw tree on purpose: clean_soup deletes
        scripts and navigational markup, and the crawl frontier is built from
        exactly those links.
        """
        html = (
            "<html><body><nav><a href='/menu'>m</a></nav>"
            '<script src="/app.js"></script>'
            '<div class="cookie-consent"><a href="/cookie">c</a></div>'
            '<main><a href="/content">k</a></main>'
            "</body></html>"
        )
        _, page = parse_html(html, "https://example.com/")
        assert page.links == [
            "https://example.com/menu",
            "https://example.com/cookie",
            "https://example.com/content",
        ]
        assert page.js_links == ["https://example.com/app.js"]
        # The same links feed the frontier, but none of the noise reaches the output.
        assert "cookie" not in page.markdown_content
        assert "menu" not in page.markdown_content
        assert "[k](https://example.com/content)" in page.markdown_content

    def test_document_order_is_preserved(self):
        """Links and scripts keep page order, which the queue depends on."""
        html = (
            "<html><body>"
            "<a href='/1'>1</a><script src='/1.js'></script>"
            "<a href='/2'>2</a><script src='/2.js'></script>"
            "</body></html>"
        )
        _, page = parse_html(html, "https://example.com/")
        assert page.links == ["https://example.com/1", "https://example.com/2"]
        assert page.js_links == ["https://example.com/1.js", "https://example.com/2.js"]


class TestExtractHelpersOnRawTrees:
    """The standalone helpers must keep working on a tree nothing has cleaned."""

    def test_metadata_from_an_unfiltered_tree(self):
        soup = BeautifulSoup(SIMPLE_HTML, "lxml")
        assert _extract_metadata(soup).title == "Test Site"

    def test_first_title_wins(self):
        soup = BeautifulSoup(
            "<html><head><title>First</title><title>Second</title></head></html>", "lxml"
        )
        assert _extract_metadata(soup).title == "First"

    def test_missing_title_leaves_default(self):
        soup = BeautifulSoup("<html><body><p>x</p></body></html>", "lxml")
        assert _extract_metadata(soup).title == ""

    def test_blank_title_is_ignored(self):
        soup = BeautifulSoup("<html><head><title>   </title></head></html>", "lxml")
        assert _extract_metadata(soup).title == ""

    def test_non_http_base_url_yields_no_links(self):
        assert extract_links("<a href='/a'>a</a>", "ftp://example.com/") == []

    def test_relative_base_url_matches_relative_links(self):
        links = extract_links("<a href='page'>p</a>", "https://example.com/dir/")
        assert links == ["https://example.com/dir/page"]

    def test_protocol_relative_script_is_resolved(self):
        soup = BeautifulSoup("<script src='//cdn.example.com/a.js'></script>", "lxml")
        assert _extract_js_links(soup, "https://example.com/") == ["https://cdn.example.com/a.js"]

    def test_javascript_url_script_is_rejected(self):
        soup = BeautifulSoup("<script src='javascript:void(0)'></script>", "lxml")
        assert _extract_js_links(soup, "https://example.com/") == []


# ── site index write ─────────────────────────────────────────────────────────


def _manifest(i: int, body_bytes: int = 24_000) -> SiteManifest:
    """One page's worth of manifest: text plus markdown, as a real run records."""
    body = "lorem ipsum dolor sit amet " * (body_bytes // 27)
    return SiteManifest(
        url=f"https://site{i}.example.com/page/{i}",
        domain=f"site{i}.example.com",
        html_file=f"/tmp/out/site{i}/index.html",
        metadata=SiteMetadata(title=f"Site {i}", description="d" * 160, author="Pulkit"),
        text_content=body,
        js_files=[f"https://cdn.example.com/app{i}.js"],
        js_count=1,
        markdown_content=f"## heading\n\n{body}",
        bytes_received=51_234,
        elapsed_ms=120 + i,
        timestamp="2024-01-01 00:00:00",
        success=True,
    )


@pytest.fixture
def fake_engine(monkeypatch):
    """
    Run ``scrape_multiple`` end to end with the crawl engine stubbed out.

    The index write is one line of ``scrape_multiple``, so a test aimed at that
    line has to reach it through the real function — calling the writer
    directly would pass even if the orchestrator kept using the old serialiser.
    """

    def run_with(output_dir, manifests: list[SiteManifest]) -> str:
        import protor.scraper as scraper_mod

        class StubEngine:
            def __init__(self, **_kwargs):
                pass

            def run(self):
                return CrawlStats(
                    scraped=len(manifests),
                    bytes_total=sum(m.bytes_received for m in manifests),
                )

            @property
            def manifests(self):
                return manifests

        monkeypatch.setattr(scraper_mod, "CrawlEngine", StubEngine)
        monkeypatch.setattr(scraper_mod.console, "print", lambda *_a, **_k: None)
        return scraper_mod.scrape_multiple(
            [m.url for m in manifests],
            output_dir=str(output_dir),
            live=False,
            download_js=False,
        )

    return run_with


class TestSiteIndexWrite:
    """
    ``sites_index.json`` must cost one manifest of memory, not one whole index.

    ``save_json`` serialised the full list to a ``str`` and then let ``write_text``
    encode it again, so a batch run held two complete copies of every scraped
    page at once: 201 MiB of peak allocation for a 2,000-site index, and no
    ceiling on it — ~50 KB of manifest per page means ~50 MB per 1,000 pages.
    """

    def test_round_trips_to_the_previous_serialisation(self, tmp_path):
        """Whatever it writes must parse back to exactly the old document."""
        manifests = [_manifest(i) for i in range(6)]
        index = tmp_path / "sites_index.json"

        _write_manifest_index(manifests, index)

        expected = json.loads(
            json.dumps([m.to_dict() for m in manifests], indent=2, ensure_ascii=False)
        )
        assert json.loads(index.read_text(encoding="utf-8")) == expected

    def test_output_is_byte_identical_to_save_json(self, tmp_path):
        """Same bytes means `protor analyze` and every reader see no change."""
        manifests = [_manifest(i, body_bytes=4_000) for i in range(5)]

        streamed = tmp_path / "streamed.json"
        _write_manifest_index(manifests, streamed)
        reference = tmp_path / "reference.json"
        save_json([m.to_dict() for m in manifests], reference)

        assert streamed.read_bytes() == reference.read_bytes()

    def test_empty_run_writes_an_empty_array(self, tmp_path):
        """A run that fetched nothing still has to leave a loadable index."""
        index = tmp_path / "sites_index.json"

        _write_manifest_index([], index)

        assert json.loads(index.read_text(encoding="utf-8")) == []

    def test_missing_parent_directory_is_created(self, tmp_path):
        index = tmp_path / "nested" / "deeper" / "sites_index.json"
        _write_manifest_index([_manifest(0, body_bytes=100)], index)
        assert len(json.loads(index.read_text(encoding="utf-8"))) == 1

    def test_peak_memory_is_bounded_by_one_manifest(self, tmp_path, fake_engine):
        """
        The whole point: peak must not scale with the size of the index.

        200 manifests of ~50 KB reproduce the reported shape at a fraction of the
        runtime, and it goes through ``scrape_multiple`` rather than the writer
        directly — a streaming writer the orchestrator does not call fixes
        nothing. Serialising the whole list peaks at roughly two copies of the
        document (~20 MB here); streaming peaks at a single manifest (< 1 MB).
        """
        manifests = [_manifest(i) for i in range(200)]

        tracemalloc.start()
        try:
            index = Path(fake_engine(tmp_path, manifests))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        doc_bytes = index.stat().st_size
        assert doc_bytes > 5_000_000, f"fixture too small to be meaningful: {doc_bytes} B"
        # One manifest is ~50 KB; even generous headroom stays far below a tenth
        # of the document, which the old path exceeded by 2x.
        assert peak < doc_bytes // 10, (
            f"peak {peak / 1e6:.1f} MB for a {doc_bytes / 1e6:.1f} MB index"
        )

    def test_scrape_multiple_writes_every_manifest(self, tmp_path, fake_engine):
        """The streaming write must not drop or reorder anything on the way out."""
        manifests = [_manifest(i, body_bytes=2_000) for i in range(30)]

        index = Path(fake_engine(tmp_path, manifests))

        parsed = json.loads(index.read_text(encoding="utf-8"))
        assert len(parsed) == len(manifests)
        assert [r["url"] for r in parsed] == [m.url for m in manifests]

    def test_non_ascii_content_survives(self, tmp_path):
        """`ensure_ascii=False` was deliberate — keep it, or the index bloats."""
        manifests = [_manifest(0, body_bytes=100)]
        manifests[0].text_content = "café — naïve 日本語"
        index = tmp_path / "sites_index.json"

        _write_manifest_index(manifests, index)

        raw = index.read_text(encoding="utf-8")
        assert "café — naïve 日本語" in raw, "escaped instead of written as UTF-8"
        assert json.loads(raw)[0]["text_content"] == "café — naïve 日本語"

    def test_scrape_multiple_index_still_parses(self, tmp_path):
        """The orchestrator's own output path must stay a valid JSON array."""
        from protor.scraper import scrape_multiple

        index = Path(scrape_multiple([], output_dir=str(tmp_path), live=False))

        assert json.loads(index.read_text(encoding="utf-8")) == []


# ── cache orphan sweep ────────────────────────────────────────────────────────


@pytest.fixture
def counting_sweep(monkeypatch):
    """Count how many body-directory sweeps a run of constructions triggers."""
    calls: list[int] = []
    real = HTTPCache._sweep_orphan_bodies

    def spy(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(HTTPCache, "_sweep_orphan_bodies", spy)
    return calls


class TestCacheOrphanSweepIsNotPerConstruction:
    """
    Opening a cache must not walk every body file it holds.

    The constructor reconciled the bodies directory on every single open, so the
    cost grew with the cache — 2.1 ms for 200 bodies, 23 ms for 2,000, all of it
    before the run had done any work. Orphans only appear when a run died between
    writing a body and flushing the index, so a timer bounds the leak just as
    tightly as a sweep per open, at a fixed share of the cost.
    """

    @pytest.fixture
    def populated(self, tmp_path):
        """A cache with a real index and 50 bodies on disk."""
        cache = HTTPCache(cache_dir=tmp_path / "c", sweep_interval=300)
        for i in range(50):
            cache.put(f"https://s{i}.example.com/", CacheEntry(body="x" * 500))
        cache.flush()
        return tmp_path / "c"

    def test_reopening_repeatedly_sweeps_at_most_once(self, populated, counting_sweep):
        # The fixture already swept once and left a marker behind.
        for _ in range(10):
            HTTPCache(cache_dir=populated)

        assert counting_sweep == [], f"swept {len(counting_sweep)} times in 10 reopens"

    def test_a_live_body_is_never_touched_by_a_skipped_sweep(self, populated):
        before = {p.name for p in (populated / "bodies").glob("*.body")}
        HTTPCache(cache_dir=populated)
        assert {p.name for p in (populated / "bodies").glob("*.body")} == before

    def test_the_first_open_of_a_fresh_cache_still_sweeps(self, populated, counting_sweep):
        """
        No marker means no sweep has ever happened, so the interval must not be
        treated as already satisfied — otherwise a cache directory that predates
        this marker would never be reconciled.
        """
        (populated / _SWEEP_MARKER).unlink()

        HTTPCache(cache_dir=populated)

        assert len(counting_sweep) == 1

    def test_an_orphan_is_swept_once_the_interval_elapses(self, populated, counting_sweep):
        orphan = populated / "bodies" / "orphan.body"
        orphan.write_text("left by a run that died before flush", encoding="utf-8")

        HTTPCache(cache_dir=populated)
        assert orphan.exists(), "swept inside the interval, defeating the point"

        # Backdate the marker past the interval rather than sleeping through it.
        old = time.time() - 10_000
        os.utime(populated / _SWEEP_MARKER, (old, old))
        HTTPCache(cache_dir=populated)

        assert len(counting_sweep) == 1
        assert not orphan.exists(), "an orphan outlived its interval"

    def test_a_zero_interval_sweeps_on_every_open(self, populated, counting_sweep):
        """sweep_interval=0 restores the old behaviour for callers that want it."""
        for _ in range(4):
            HTTPCache(cache_dir=populated, sweep_interval=0)

        assert len(counting_sweep) == 4

    def test_explicit_prune_forces_the_sweep(self, populated, counting_sweep):
        cache = HTTPCache(cache_dir=populated)
        assert counting_sweep == []  # inside the interval, so the open skipped it

        cache.prune()  # ...but the caller asked for a sweep and gets one

        assert len(counting_sweep) == 1

    def test_prune_can_skip_the_sweep_entirely(self, populated, counting_sweep):
        cache = HTTPCache(cache_dir=populated)
        cache.prune(sweep_orphans=False)
        assert counting_sweep == []


class TestCacheSweepCannotLoseData:
    """The sweep is now rarer, so its failure modes have to be pinned down."""

    @pytest.fixture
    def populated(self, tmp_path):
        cache = HTTPCache(cache_dir=tmp_path / "c")
        for i in range(20):
            cache.put(f"https://s{i}.example.com/", CacheEntry(body=f"body-{i}"))
        cache.flush()
        return tmp_path / "c"

    def test_an_unreadable_index_is_never_swept_against(self, populated):
        """
        A damaged index parses as "no entries". Reconciling against that deleted
        every body file, so a skipped sweep here is the safe direction.
        """
        (populated / "index.json").write_text(
            '{"https://s0.example.com/": {"etag":', encoding="utf-8"
        )

        for _ in range(3):
            HTTPCache(cache_dir=populated)

        assert len(list((populated / "bodies").glob("*.body"))) == 20

    def test_bodies_without_an_index_are_swept_immediately(self, populated):
        """
        A lost index is the worst case: every body is unreachable. That is swept
        whatever the interval says, so the bytes cannot outlive the cache dir.
        """
        HTTPCache(cache_dir=populated)  # marker now exists
        (populated / "index.json").unlink()

        HTTPCache(cache_dir=populated)

        assert list((populated / "bodies").glob("*.body")) == []

    def test_expired_entries_are_still_reclaimed_on_every_open(self, tmp_path):
        """
        Entry expiry is an in-memory walk, not a directory sweep, so it must not
        be put on the interval — an abandoned cache would keep its bytes forever.
        """
        cache = HTTPCache(cache_dir=tmp_path / "c", ttl=1, stale_ttl=1)
        cache.put("https://old.example.com/", CacheEntry(body="aged"))
        cache.put("https://new.example.com/", CacheEntry(body="fresh"))
        cache.flush()
        # Age one entry on disk, so the reopen sees it rather than a live cache.
        cache._index["https://old.example.com/"].timestamp -= 10_000
        cache._dirty = True  # flush() is a no-op when clean
        cache.flush()
        HTTPCache(cache_dir=tmp_path / "c", ttl=1, stale_ttl=1)  # sets the marker

        # Inside the sweep interval, twice over: entry expiry must still happen.
        for _ in range(2):
            reopened = HTTPCache(cache_dir=tmp_path / "c", ttl=1, stale_ttl=1)

        assert "https://old.example.com/" not in reopened._index
        assert "https://new.example.com/" in reopened._index

    def test_stale_entries_keep_their_validators(self, populated):
        """The retention window must survive the sweep being made rarer."""
        cache = HTTPCache(cache_dir=populated, ttl=1, stale_ttl=3600)
        cache.put("https://a.example.com/", CacheEntry(body="payload", etag='W/"1"'))
        cache.flush()
        cache._index["https://a.example.com/"].timestamp -= 10

        assert cache.get("https://a.example.com/") is None, "stale entries are not served"
        assert cache.prune() == 0, "still inside the retention window"
        assert cache.conditional_headers("https://a.example.com/") == {"If-None-Match": 'W/"1"'}
        assert cache._body_path("https://a.example.com/").exists()

    def test_clear_does_not_leave_a_stale_sweep_marker_behind(self, populated):
        """
        After a clear the directory is empty; a marker claiming "already swept"
        is accurate, but a fresh open must not go on trusting a verdict about
        bodies it has never seen.
        """
        cache = HTTPCache(cache_dir=populated)
        cache.clear()
        assert cache.size_bytes() == 0

        cache.put("https://new.example.com/", CacheEntry(body="after clear"))
        cache.flush()

        reopened = HTTPCache(cache_dir=populated)
        assert reopened.get("https://new.example.com/") is not None
