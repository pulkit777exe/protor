"""
One visual language across every command.

The audit behind this file found the same fact drawn five ways: a *success*
headline sat in the default colour while warnings were yellow and errors red, a
permanently-skipped page was drawn with the in-progress glyph in yellow, and the
progress bar and its own percentage disagreed about how far along the crawl was.
None of that is a crash, which is exactly why it survives — the output looks
plausible in a screenshot and is wrong in the terminal.
"""

from __future__ import annotations

from io import StringIO
from typing import ClassVar

import pytest
from rich.console import Console

from protor import theme


def _render(printable) -> str:
    buf = StringIO()
    con = Console(file=buf, width=200, force_terminal=True, color_system="truecolor")
    printable(con)
    return buf.getvalue()


def _plain(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestHeadlineGlyphsAreColoured:
    """
    Warnings were yellow and errors red; every headline success and failure used
    the default style, so `✓ 40 scraped` and `✗ 2 failed` came out the same colour.
    """

    def test_ok_marked_is_green(self):
        out = _render(lambda c: c.print(theme.OK_STYLED))
        assert "\x1b[32m" in out or "\x1b[38;5;" in out, repr(out)
        assert _plain(out).strip() == theme.OK

    def test_err_marked_is_red(self):
        out = _render(lambda c: c.print(theme.ERR_STYLED))
        assert _plain(out).strip() == theme.ERR
        # Red, specifically: green would read as success.
        assert "\x1b[31m" in out or "38;5;" in out, repr(out)

    def test_a_headline_can_colour_its_glyph_without_eating_its_markup(self):
        """
        Why `OK_STYLED` exists at all.

        A summary line mixes the glyph with markup its neighbours produced, so it
        cannot use `ok()` — that escapes its argument, which would eat
        `bright(count)` and strip the colour from the number next to it.
        """
        out = _render(lambda c: c.print(f"{theme.OK_STYLED} 40 {theme.bright('scraped')}"))
        assert _plain(out).strip() == f"{theme.OK} 40 scraped"
        assert out.count("\x1b[") > 2, "the count lost its styling"


class TestSkippedIsNotDrawnAsWork:
    """
    A skipped page is a finished outcome, and it was drawn as a spinner in yellow.

    `protor scrape` had no case for `skipped`, so it fell through to the branch
    that renders in-progress work. The crawl's live view already drew it with the
    skip glyph in grey, so the two commands disagreed about the same state.
    """

    ROWS: ClassVar = [
        ("done", theme.OK, "done"),
        ("error", theme.ERR, "error"),
        ("blocked", theme.ERR, "blocked"),
        ("skipped", theme.SKIP, "skipped"),
    ]

    def _status_cell(self, status: str):
        from protor.scraper import _build_table

        table = _build_table([{"idx": 1, "domain": "ex.com", "status": status}])
        # rich keeps no public accessor for a cell; this reads the renderable
        # the table would draw, which is what the assertion is about.
        return table.columns[2]._cells[0]

    @pytest.mark.parametrize(("status", "glyph", "word"), ROWS)
    def test_the_glyph_matches_the_outcome(self, status, glyph, word):
        cell = self._status_cell(status)
        rendered = _plain(str(cell))
        assert glyph in rendered, f"{status} drew {rendered!r}, expected {glyph!r}"
        assert word in rendered, rendered

    def test_skipped_is_not_the_in_progress_glyph(self):
        """The specific bug: `◌` in yellow for a page that will never change."""
        cell = self._status_cell("skipped")
        assert theme.SPIN not in _plain(str(cell)), "a finished page is drawn as spinning"
        assert cell.style == "grey50", f"skipped should be quiet, got {cell.style!r}"

    def test_fetching_still_looks_like_work(self):
        """The control: the skip glyph must not have leaked onto live rows."""
        cell = self._status_cell("fetching")
        assert theme.SPIN in _plain(str(cell))
        assert cell.style == "yellow"


class TestTheBarAgreesWithItsOwnPercentage:
    """
    `round(32 * 63/64)` is 32, so the bar was full while the number beside it read
    98%. Two figures for the same quantity, on the same line, disagreeing — and
    the full bar is the one a glance trusts.
    """

    def _bar(self, scraped: int, max_pages: int) -> str:
        """The rendered progress line, ANSI stripped."""
        from collections import deque

        from protor.crawler import _render, _State

        state = _State(scraped=scraped, max_pages=max_pages, log=deque(maxlen=1))
        return _plain(_render_text(_render(state, "/tmp/out")))

    @pytest.mark.parametrize(
        ("scraped", "max_pages"), [(1, 100), (7, 10), (31, 32), (63, 64), (32, 32), (0, 10)]
    )
    def test_the_two_figures_never_disagree(self, scraped, max_pages):
        from protor.crawler import _BAR_WIDTH

        bar = self._bar(scraped, max_pages)
        filled = bar.count("█")
        pct = int(bar.split("%")[0].split()[-1])

        assert filled + bar.count("░") == _BAR_WIDTH, f"the bar changed width: {bar!r}"
        expected = int(_BAR_WIDTH * min(scraped / max_pages, 1.0))
        assert filled == expected, f"bar and percentage disagree: {bar!r}"
        if pct < 100:
            assert filled < _BAR_WIDTH, f"a full bar at {pct}%: {bar!r}"

    def test_the_last_page_fills_it(self):
        bar = self._bar(32, 32)
        assert bar.count("█") == 32, bar
        assert "100%" in bar, bar


def _render_text(renderable, width: int = 200) -> str:
    """Render to text through a console of a known size, the way a terminal would."""
    buf = StringIO()
    Console(file=buf, width=width, force_terminal=False, color_system=None).print(renderable)
    return buf.getvalue()


def _crawl_state(rows: int = 40):
    """A crawl mid-flight, with a log long enough to overflow the view."""
    from collections import deque

    from protor.crawler import _CrawlLog, _State

    return _State(
        scraped=12,
        max_pages=100,
        queue_n=40,
        log_total=rows,
        log=deque(
            [_CrawlLog("ok", f"host{i}.example.com", "", url=f"u{i}") for i in range(rows)],
            maxlen=200,
        ),
    )


def _rows(n: int = 30) -> list[dict]:
    """Batch rows with a domain long enough to test the column budget."""
    return [
        {
            "idx": i,
            "domain": f"a-really-quite-long-domain-name-{i}.example.com",
            "status": "done",
            "bytes": 123456,
            "ms": 1200,
            "js": 3,
        }
        for i in range(n)
    ]


class TestTheViewsFitTheWindow:
    """
    A live region taller than the terminal scrolls its own top out of view.

    Both views used fixed row counts with no relation to the window: the crawler's
    rendered 29 lines and the batch table 28, on a terminal that is 24. What
    scrolled away was the bar, the percentage and the counts — the part the log
    table does not duplicate — so the user watched a list of domains with no idea
    how far along the crawl was.
    """

    @pytest.mark.parametrize("height", [24, 30, 40, 60])
    def test_the_crawl_view_is_not_taller_than_the_terminal(self, height):
        from protor.crawler import _render

        out = _render_text(_render(_crawl_state(), "/tmp/out", height=height), width=80)
        assert len(out.splitlines()) <= height, (
            f"{len(out.splitlines())} lines into {height}; the bar has scrolled away"
        )

    @pytest.mark.parametrize("height", [24, 30, 40, 60])
    def test_the_batch_table_is_not_taller_than_the_terminal(self, height):
        from protor.scraper import _build_table

        out = _render_text(_build_table(_rows(), width=80))
        assert len(out.splitlines()) <= height, f"{len(out.splitlines())} lines into {height}"

    def test_a_short_terminal_shows_fewer_rows_not_nothing(self):
        from protor.crawler import _render

        short = _render_text(_render(_crawl_state(), "/tmp/out", height=24), width=80)
        tall = _render_text(_render(_crawl_state(), "/tmp/out", height=50), width=80)
        assert short.count("host") < tall.count("host"), "row count ignores the window"

    def test_a_tall_terminal_is_still_capped(self):
        """
        Otherwise a 200-row terminal renders 170 rows per frame, which is the
        repaint cost the live view was throttled to avoid in the first place.
        """
        from protor.crawler import _LOG_VIEW, _render

        out = _render_text(_render(_crawl_state(rows=200), "/tmp/out", height=200), width=80)
        assert out.count("host") <= _LOG_VIEW

    def test_the_bar_is_in_the_frame_that_survives(self):
        """The point of fitting: the counts must be on screen at 24 lines."""
        from protor.crawler import _render

        out = _render_text(_render(_crawl_state(), "/tmp/out", height=24), width=80)
        assert "progress" in out, "the bar scrolled out of its own view"
        assert "12/100" in out
        assert "12%" in out


class TestTheBatchTableFitsEightyColumns:
    """
    The columns needed 81 and 80 is the canonical terminal width.

    Rich dropped the last one outright rather than narrowing anything, so at 80
    columns the JS count — the answer to "did this page pull scripts?" — silently
    disappeared with no ellipsis and no warning. A `min_width` on Domain is not
    enough on its own: it is a floor, not a ceiling, so a long domain ate the whole
    budget and squeezed the rest out.
    """

    @pytest.mark.parametrize("width", [60, 72, 80, 100, 120])
    def test_nothing_runs_past_the_edge(self, width):
        from protor.scraper import _build_table

        out = _render_text(_build_table(_rows(), width=width), width=width)
        widest = max(len(line) for line in out.splitlines())
        assert widest <= width, f"{widest} columns into {width}"

    def test_every_column_survives_at_eighty(self):
        from protor.scraper import _build_table

        header = _render_text(_build_table(_rows(), width=80), width=80).splitlines()[0]
        for column in ("#", "Domain", "Status", "Size", "Time", "JS"):
            assert column in header, f"{column} dropped at 80 columns: {header!r}"

    @pytest.mark.parametrize("width", [62, 72, 80, 100])
    def test_nothing_is_squeezed_when_the_layout_fits(self, width):
        """
        Rich's failure mode here is worse than overflowing.

        It sizes columns to their content and only then finds the table too wide,
        at which point it shrinks *every* column: a 45-character domain in a
        72-column window rendered a size as `120.…` and a time as `1…`, with no
        dropped column to explain it. Truncated numbers read as corrupt data.
        """
        from protor.scraper import _build_table

        out = _render_text(_build_table(_rows(), width=width), width=width)
        assert "120.6 KB" in out, f"a size was squeezed at {width} columns:\n{out}"
        assert "1.2s" in out, f"a time was squeezed at {width} columns:\n{out}"

    def test_a_narrow_terminal_drops_a_column_rather_than_squeezing(self):
        """The trade `list_runtimes` already makes with its URL column."""
        from protor.scraper import _build_table

        header = _render_text(_build_table(_rows(), width=50), width=50).splitlines()[0]
        assert "JS" not in header, header
        assert "#" in header and "Domain" in header, "the columns that matter stay"

    def test_a_dropped_column_does_not_shift_the_others(self):
        """
        Rich lays the row out from the declared columns, so dropping one must not
        leave a hole — every value after Domain would otherwise move.
        """
        from protor.scraper import _build_table

        lines = _render_text(_build_table(_rows(), width=50), width=50).splitlines()
        # Width 50 keeps #, Domain, Status and Time; the Size column is one of the
        # sacrifices, so key off the status every layout has.
        body = [line for line in lines if "done" in line]
        assert len(body) >= 3, lines
        assert len({len(line) for line in body}) == 1, f"ragged rows: {body}"

    def test_every_declared_column_fits_at_its_declared_width(self):
        """The layout is chosen from arithmetic, so the arithmetic had better hold."""
        from protor.scraper import _BATCH_COLUMNS, _columns_for, _columns_width

        full = _columns_width(_BATCH_COLUMNS)
        assert _columns_width(_columns_for(full)) == full, (
            "the full layout does not fit its own width"
        )

    def test_the_row_number_and_domain_are_never_sacrificed(self):
        from protor.scraper import _SACRIFICE_ORDER, _columns_for

        for width in range(20, 130, 2):
            names = {name for name, *_ in _columns_for(width)}
            assert "#" in names and "Domain" in names, f"at {width}: {names}"
        assert not ({"#", "Domain", "Status"} & set(_SACRIFICE_ORDER))
