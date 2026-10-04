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


def _render_text(renderable) -> str:
    """Render to text through a console, the way a terminal would."""
    buf = StringIO()
    Console(file=buf, width=200, force_terminal=False, color_system=None).print(renderable)
    return buf.getvalue()
