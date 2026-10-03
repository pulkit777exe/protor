"""
The plain-text preview is the analyser's prompt body, so decoration is waste.

`_extract_text` produces the `text_content` field, and the analyser sends that
field to the model. Every character in it is therefore charged against the
context window that the character budget exists to protect.

Real pages put their table structure in the document as *text*. Hacker News
separates every column with a literal `|` and wraps each link's domain in bare
parentheses, which put 132 punctuation-only strings into a 420-string preview of
its front page — 31% of the prompt spent on `|`, `(` and `)`. The same page's
Markdown, rendered separately, was fine; this was the plain-text path only.
"""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from protor.markdown import clean_soup
from protor.parser import _extract_text, parse_html


def _text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    clean_soup(soup)
    return _extract_text(soup)


class TestPunctuationIsDropped:
    def test_table_separator_pipes_are_dropped(self):
        """The Hacker News shape: a literal pipe between every pair of cells."""
        html = "<table><tr><td>Title</td><td class='sep'>|</td><td>17 points</td></tr></table>"
        out = _text(html)
        assert "Title" in out and "17 points" in out
        assert "|" not in out, f"table separators leaked into the preview: {out!r}"

    def test_bare_parentheses_around_a_link_are_dropped(self):
        html = "<span class='comhead'>(<a href='http://x.com'>x.com</a>)</span>"
        out = _text(html)
        assert "x.com" in out, "the domain itself is content"
        assert "(" not in out and ")" not in out, f"bare parens leaked: {out!r}"

    @pytest.mark.parametrize("decoration", ["|", "(", ")", "[", "]", "•", "·", "—", "…", "/", "-"])
    def test_single_punctuation_marks_are_dropped(self, decoration):
        assert _text(f"<p>Real text</p><span>{decoration}</span>") == "Real text"

    def test_no_line_is_punctuation_only(self):
        """The property that matters, checked on a layout built to violate it."""
        rows = "".join(
            f"<tr><td>Story {i}</td><td class='sep'>|</td>"
            f"<td>{i} points</td><td class='sep'>|</td></tr>"
            for i in range(20)
        )
        lines = [ln for ln in _text(f"<table>{rows}</table>").splitlines() if ln.strip()]
        assert lines, "content was lost entirely"
        assert all(any(c.isalnum() for c in ln) for ln in lines), (
            f"{sum(1 for ln in lines if not any(c.isalnum() for c in ln))} punctuation-only lines survive"
        )


class TestContentIsKept:
    def test_ordinary_text_is_untouched(self):
        assert _text("<p>The quick brown fox.</p>") == "The quick brown fox."

    def test_numbers_count_as_content(self):
        """A price or a count is meaning; it must not be filtered as decoration."""
        assert _text("<p>Only 17 left</p>") == "Only 17 left"
        assert _text("<p>2024</p>") == "2024"

    def test_non_latin_scripts_count_as_content(self):
        """`\\w` matches CJK, but the filter must not be ASCII-only."""
        for text in ("日本語のテキスト", "Ελληνικά", "Привет", "العربية"):
            assert _text(f"<p>{text}</p>") == text, f"{text!r} was dropped"

    def test_a_mixed_line_keeps_its_punctuation(self):
        """Only the *whole* string must lack letters; punctuation inside is fine."""
        out = _text("<p>Revenue: $4.2M (up 12%)</p>")
        assert out == "Revenue: $4.2M (up 12%)"

    def test_underscore_only_content_is_dropped(self):
        """`_` is a word character to `\\w`, so the filter excludes it explicitly."""
        assert _text("<p>___</p>") == ""


class TestBudgetInteraction:
    def test_dropping_decoration_leaves_more_room_for_content(self):
        """The point of the change: the same budget now buys more real content."""
        noise = "".join("<td class='sep'>|</td>" for _ in range(400))
        content = "".join(f"<p>Story number {i}</p>" for i in range(60))

        html_noisy = f"<div>{noise}</div><div>{content}</div>"
        soup = BeautifulSoup(html_noisy, "lxml")
        clean_soup(soup)

        # Without the filter, the 400 separators are walked first and consume the
        # whole 200-character budget: the site itself is never reached. With it,
        # the budget is spent on the site.
        kept = _extract_text(soup, max_chars=200)
        assert "Story number" in kept, f"no content survived: {kept!r}"
        assert len(kept) > 150, f"only {len(kept)} of 200 characters spent on content"
        assert "|" not in kept

    def test_the_budget_is_still_respected(self):
        soup = BeautifulSoup("<p>" + ("word " * 5000) + "</p>", "lxml")
        clean_soup(soup)
        assert len(_extract_text(soup, max_chars=100)) <= 100


class TestEndToEnd:
    def test_parse_html_reports_clean_text_content(self):
        html = (
            "<html><body><table><tr>"
            "<td><a href='/a'>Real story</a></td><td class='sep'>|</td>"
            "<td>42 points</td>"
            "</tr></table></body></html>"
        )
        _, page = parse_html(html, "https://news.example/")
        assert "Real story" in page.text_content
        assert "42 points" in page.text_content
        assert "|" not in page.text_content, page.text_content
