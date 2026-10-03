"""
Tables and unlisted container tags must not swallow a page.

Every one of these was found by scraping a real site rather than by reading the
code: Hacker News produced an **empty** Markdown document while its plain-text
extraction worked perfectly, so the failure was invisible in the tests that
existed and fatal to the feature. Markdown is what the analyser reads, so an
empty document there means the model is asked to analyse nothing.

The theme is the same in all three cases. Rendering was driven by an *allowlist*
of tag names (``_BLOCK_TAGS``), and anything not on it was treated as inline
text. Two rules then combined: the block renderer declined to descend into a tag
it did not recognise, and the inline renderer skipped block descendants. Between
them, an unlisted wrapper erased everything inside it.
"""

from __future__ import annotations

import pytest

from protor.markdown import html_to_markdown, soup_to_markdown


class TestUnlistedContainers:
    """A tag absent from the allowlist must not hide the content inside it."""

    @pytest.mark.parametrize(
        "wrapper,html,expected",
        [
            # Hacker News is built on this one. A <table> is a block, so the
            # inline renderer skipped it while the block renderer never entered
            # the <center> around it: the page rendered as "".
            ("center", "<center><p>Real paragraph.</p></center>", "Real paragraph."),
            (
                "center>table",
                "<center><table><tr><td>Cell text</td></tr></table></center>",
                "Cell text",
            ),
            # Legacy markup, still in wide use.
            ("font", '<font size="2"><p>Real paragraph.</p></font>', "Real paragraph."),
            ("center>div>p", "<center><div><p>Nested.</p></div></center>", "Nested."),
            # Nesting deeper than one unlisted level.
            ("span>span>p", "<span><span><p>Deep.</p></span></span>", "Deep."),
            ("center>font>p", "<center><font><p>Deeper.</p></font></center>", "Deeper."),
        ],
    )
    def test_content_inside_an_unlisted_tag_survives(self, wrapper, html, expected):
        got = html_to_markdown(html, "https://x.com")
        assert got.strip(), f"<{wrapper}> produced an empty document"
        assert expected in got

    def test_a_list_wrapped_in_an_unlisted_tag_survives(self):
        md = html_to_markdown(
            "<center><ul><li>First</li><li>Second</li></ul></center>", "https://x.com"
        )
        assert "First" in md and "Second" in md, md

    def test_inline_formatting_is_not_regressed(self):
        """The fix must not turn inline tags into block boundaries."""
        md = html_to_markdown(
            '<p>Some <b>bold</b> and <a href="/l">a link</a> plus <em>emphasis</em>.</p>',
            "https://x.com",
        )
        assert md == "Some **bold** and [a link](https://x.com/l) plus *emphasis*."

    def test_a_deeply_nested_unlisted_wrapper_terminates(self):
        """Unbounded wrapping must not turn into unbounded recursion."""
        html = "<span>" * 400 + "<p>Deep enough.</p>" + "</span>" * 400
        md = html_to_markdown(html, "https://x.com")
        assert "Deep enough." in md


class TestTableSize:
    """Padding a table out to its column widths is waste, and it is large waste."""

    def test_a_table_is_not_padded_to_its_widest_cell(self):
        html = (
            "<table><tr><th>short</th><th>other</th></tr>"
            "<tr><td>" + "x" * 4000 + "</td><td>b</td></tr></table>"
        )
        md = html_to_markdown(html, "https://x.com")
        assert "x" * 4000 in md, "the cell content must survive"
        # The old renderer padded every cell in that column to 4000 characters.
        assert len(md) < 4200, f"padding inflated the table to {len(md)} chars"

    def test_wide_and_narrow_rows_do_not_produce_runs_of_empty_pipes(self):
        html = (
            "<table><tr><td>a</td></tr>"
            "<tr>" + "".join(f"<td>c{i}</td>" for i in range(20)) + "</tr></table>"
        )
        md = html_to_markdown(html, "https://x.com")
        assert "|  |" not in md, f"empty padding cells in:\n{md}"

    def test_the_separator_matches_the_header_width(self):
        html = (
            "<table><tr><td>a</td><td>b</td></tr>"
            "<tr>" + "".join(f"<td>c{i}</td>" for i in range(12)) + "</tr></table>"
        )
        md = html_to_markdown(html, "https://x.com")
        separator = [ln for ln in md.splitlines() if set(ln) <= set("| -") and ln.strip()]
        assert len(separator) == 1, md
        assert separator[0].count("---") == 2, f"separator sized from the wrong row: {separator[0]}"


class TestNestedTables:
    """A layout table frames content; it must neither absorb nor hide it."""

    def test_content_inside_a_nested_table_appears_once_not_thrice(self):
        title = "A distinctive story title"
        html = f"<table><tr><td><table><tr><td>{title}</td></tr></table></td></tr></table>"
        md = html_to_markdown(html, "https://x.com")
        assert md.count(title) == 1, f"duplicated content:\n{md}"

    def test_a_layout_table_does_not_restate_its_nested_table(self):
        """The outer cell is a frame. Its text must not restate the frame's content."""
        html = (
            "<table><tr><td>"
            "<table><tr><td>inner one</td></tr><tr><td>inner two</td></tr></table>"
            "</td></tr></table>"
        )
        md = html_to_markdown(html, "https://x.com")
        assert md.count("inner one") == 1, md
        assert md.count("inner two") == 1, md

    def test_an_outer_row_does_not_harvest_the_inner_rows(self):
        """`row.find_all("td")` descends into the nested table; it must not."""
        html = (
            "<table><tr><td>"
            "<table>" + "".join(f"<tr><td>story {i}</td></tr>" for i in range(5)) + "</table>"
            "</td></tr></table>"
        )
        md = html_to_markdown(html, "https://x.com")
        for i in range(5):
            assert md.count(f"story {i}") == 1, f"story {i} appeared more than once:\n{md}"

    def test_deeply_nested_tables_terminate(self):
        html = "<table><tr><td>" * 30 + "core" + "</td></tr></table>" * 30
        md = html_to_markdown(html, "https://x.com")
        assert "core" in md


class TestEmptyAndSpacerRows:
    def test_a_table_of_only_empty_cells_is_dropped(self):
        md = html_to_markdown("<table><tr><td></td><td></td></tr></table>", "https://x.com")
        assert md.strip() == "", f"a content-free table should not be emitted: {md!r}"

    def test_a_rowless_table_is_dropped(self):
        assert html_to_markdown("<table></table>", "https://x.com").strip() == ""

    def test_a_real_table_still_renders(self):
        md = html_to_markdown(
            "<table><tr><th>Name</th><th>Price</th></tr><tr><td>Widget</td><td>10</td></tr></table>",
            "https://x.com",
        )
        assert "| Name | Price |" in md
        assert "| --- | --- |" in md
        assert "| Widget | 10 |" in md


class TestRegressionShape:
    """The end-to-end shape: a page whose whole body is one unlisted wrapper."""

    def test_a_table_based_page_renders_its_stories(self):
        """The Hacker News shape, reduced to its minimum."""
        page = (
            "<html><body><center><table class='itemlist'>"
            + "".join(
                f"<tr><td><span class='rank'>{i}.</span></td>"
                f"<td><a href='/item?id={i}'>Story number {i}</a></td></tr>"
                for i in range(1, 6)
            )
            + "</table></center></body></html>"
        )
        md = html_to_markdown(page, "https://news.ycombinator.com")
        assert md.strip(), "a table-based page rendered empty"
        for i in range(1, 6):
            assert md.count(f"Story number {i}") == 1, f"story {i} lost or duplicated:\n{md}"

    def test_output_size_is_proportional_to_content_not_to_layout(self):
        """
        The regression that made this worth fixing: a 34 KB page produced 715 KB
        of Markdown. Scaling the page should scale the output roughly linearly,
        not quadratically in the widest cell.
        """

        def page(n: int) -> str:
            return (
                "<center><table><tr><th>h</th></tr>"
                + "".join(f"<tr><td>{'w' * 400}-row{i}</td></tr>" for i in range(n))
                + "</table></center>"
            )

        from bs4 import BeautifulSoup

        from protor.markdown import clean_soup

        sizes = []
        for n in (10, 40):
            soup = BeautifulSoup(page(n), "lxml")
            clean_soup(soup)
            sizes.append(len(soup_to_markdown(soup, "https://x.com")))

        ratio = sizes[1] / sizes[0]
        assert ratio < 8, f"4x the rows produced {ratio:.1f}x the output: {sizes}"
