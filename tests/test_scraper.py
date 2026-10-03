"""Tests for protor.scraper HTML-parsing helpers."""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from protor.markdown import clean_soup
from protor.parser import (
    _extract_js_links,
    _extract_metadata,
    _extract_text,
    extract_links,
    parse_html,
)
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
