"""Tests for protor.scraper HTML-parsing helpers."""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from protor.scraper import (
    _extract_js_links,
    _extract_metadata,
    _extract_text,
    extract_links,
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
        links = _extract_js_links(SIMPLE_HTML, "https://example.com")
        assert "https://example.com/static/app.js" in links
        assert "https://cdn.example.com/lib.js" in links

    def test_deduplicates(self):
        html = '<script src="/a.js"></script><script src="/a.js"></script>'
        links = _extract_js_links(html, "https://example.com")
        assert links.count("https://example.com/a.js") == 1

    def test_empty(self):
        assert _extract_js_links(EMPTY_HTML, "https://example.com") == []

    def test_ignores_inline_scripts(self):
        html = "<script>console.log('inline')</script>"
        assert _extract_js_links(html, "https://example.com") == []


class TestExtractLinks:
    def test_returns_internal_links(self):
        links = extract_links(SIMPLE_HTML, "https://example.com")
        assert "https://example.com/about" in links
        assert "https://example.com/contact" in links

    def test_excludes_external_links(self):
        links = extract_links(SIMPLE_HTML, "https://example.com")
        assert not any("external.com" in l for l in links)

    def test_deduplicates(self):
        html = '<a href="/page">A</a><a href="/page">B</a>'
        links = extract_links(html, "https://example.com")
        assert links.count("https://example.com/page") == 1

    def test_strips_fragments(self):
        html = '<a href="/page#section">Link</a>'
        links = extract_links(html, "https://example.com")
        assert "https://example.com/page" in links
        assert not any("#" in l for l in links)

    def test_empty_html(self):
        assert extract_links(EMPTY_HTML, "https://example.com") == []


class TestExtractText:
    def test_removes_nav_footer_scripts(self):
        text = _extract_text(SIMPLE_HTML)
        assert "Navigation" not in text
        assert "Footer text" not in text
        assert "console.log" not in text

    def test_includes_main_content(self):
        text = _extract_text(SIMPLE_HTML)
        assert "Hello World" in text
        assert "main content" in text

    def test_truncates_long_content(self):
        long_html = "<p>" + ("x " * 10_000) + "</p>"
        text = _extract_text(long_html)
        assert len(text) <= 10_000

    def test_empty_html_returns_empty(self):
        text = _extract_text(EMPTY_HTML)
        assert text.strip() == ""