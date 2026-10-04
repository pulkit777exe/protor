"""
protor.progress
~~~~~~~~~~~~~~~
The single owner of live, stateful terminal rendering.

Everything that can take a while — scraping, crawling, waiting on an LLM —
renders through this module, so progress is one concept in one place instead of
ad-hoc ``Live`` blocks scattered across the engine, the crawler and the analyzer.

Three ideas drive the design, all borrowed from how production terminal agents
handle the same problems:

1. **Rate-limit redraws, not work.** A display that repaints once per completed
   page is O(n²) over a batch and starves the event loop it is reporting on. A
   progress bar at 10 Hz is indistinguishable from one at 1000 Hz to a human and
   orders of magnitude cheaper.
2. **Degrade, never break.** A pipe, a CI log and a dumb terminal get plain
   sequential lines instead of cursor-up escape sequences. Animation is only
   ever attempted where it can actually work.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.live import Live

from .theme import ERR, SafeTable, muted
from .theme import console as _console

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from types import TracebackType

    from rich.console import Console

__all__ = [
    "MAX_REASONS_SHOWN",
    "StreamWriter",
    "Throttle",
    "live_display",
    "live_enabled",
    "normalise_reason",
    "print_failure_reasons",
    "probing",
    "visible_rows",
]


#: Values read as "off" for the two environment switches below. Bare truthiness
#: made ``PROTOR_NO_LIVE=0`` *disable* live rendering while ``CI=0`` did not
#: disable anything — the same question answered two ways, three lines apart,
#: and the first answer is never what someone setting the variable to 0 meant.
_FALSY = {"", "0", "false", "no", "off"}


def _env_flag(name: str) -> bool:
    """True when *name* is set to anything that is not an explicit "off"."""
    return os.environ.get(name, "").strip().lower() not in _FALSY


def live_enabled(con: Console | None = None) -> bool:
    """
    Whether in-place rendering can work here.

    False for pipes, files and CI logs: writing cursor-up sequences there
    produces an unreadable transcript of escape codes. Rich already honours
    ``TERM`` (including ``dumb``) and ``NO_COLOR`` for the terminal's own
    capabilities; this adds the "is anyone watching" half of the question, which
    Rich cannot see. An ascii-capable terminal is refused too — the glyphs would
    be replaced on every repaint.
    """
    con = con or _console
    if _env_flag("PROTOR_NO_LIVE"):
        return False
    if _env_flag("CI"):
        return False
    return bool(con.is_terminal) and con.encoding != "ascii"


class Throttle:
    """
    Allow a call at most once every *interval* seconds.

    Used to keep a progress display from costing more than the work it reports.
    A forced call always goes through, so the last update of a run is never
    swallowed by the rate limit.
    """

    __slots__ = ("_interval", "_last")

    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second if per_second > 0 else 0.0
        # None, not 0.0: "never called" must always be ready, even on a clock
        # that has not advanced past the first interval.
        self._last: float | None = None

    def ready(self, *, force: bool = False) -> bool:
        """True when a call is allowed now. Records the call when it is."""
        if force or self._last is None:
            self._last = time.monotonic()
            return True
        if time.monotonic() - self._last >= self._interval:
            self._last = time.monotonic()
            return True
        return False


@dataclass
class LiveDisplay:
    """
    Handle returned by :func:`live_display`.

    ``update`` is safe to call as often as the caller likes; it renders at most
    ``per_second`` times. ``note`` writes a line that scrolls above the live
    region — the correct way to emit a message while a Live block is active.
    ``line`` is the mirror image: it writes one plain line only when there is no
    live region, which is what makes a piped run readable as it happens.
    """

    _render: Callable[[], Any]
    _live: Live | None
    _throttle: Throttle
    _enabled: bool
    _console: Console
    _notes: list[str] = field(default_factory=list)

    # ── output ───────────────────────────────────────────────────────────────

    def update(self, *, force: bool = False) -> None:
        """
        Redraw the live region, rate-limited.

        A no-op when live rendering is unavailable, so callers need no branching:
        piped output simply never animates.
        """
        if self._live is None or not self._throttle.ready(force=force):
            return
        self._live.update(self._render(), refresh=True)

    def note(self, message: str) -> None:
        """Print *message* so it scrolls above the live region and persists."""
        self._console.print(message)

    @property
    def wants_lines(self) -> bool:
        """
        Whether :meth:`line` would print anything.

        Callers build the string before handing it over, and on a live run the
        answer is always no — so a caller that formats a line first and asks second
        pays for a string it throws away, once per finished page.
        """
        return self._live is None

    def line(self, message: str) -> None:
        """
        Print one plain line, but only when there is no live region.

        A pipe and a CI log got one aggregate at the end and nothing at all while
        the work ran, so a ten-minute scrape logged a header, then ten minutes of
        nothing, then a table. The README has promised "one clean line per result"
        for exactly this case.

        A no-op while animating, because then the table already carries the row and
        the line would scroll past it.
        """
        if self._live is None:
            self._console.print(message)


@contextmanager
def live_display(
    render: Callable[[], Any],
    *,
    console: Console | None = None,
    per_second: float = 10.0,
    transient: bool = True,
    enabled: bool | None = None,
) -> Iterator[LiveDisplay]:
    """
    Show *render()* in place while long work runs.

    Parameters
    ----------
    render:
        Builds the renderable for the current state. Called at most
        ``per_second`` times per second, not once per event.
    transient:
        Erase the display when the block exits. Right for in-progress feedback;
        the caller prints the durable summary afterwards.
    enabled:
        Set False to forbid animation entirely. True and None both mean "you may
        animate" and still auto-detect: a pipe, a CI log and a dumb terminal are
        never animated, because you cannot force a cursor-up sequence to render
        somewhere that has no cursor.

    Yields
    ------
    LiveDisplay
    """
    con = console or _console
    # False forbids animation outright; anything else still auto-detects, so a
    # redirected stdout can never be handed escape codes it has no cursor for.
    enabled = enabled is not False and live_enabled(con)

    if not enabled:
        # Nothing to animate: callers keep calling update() and pay nothing.
        display = LiveDisplay(
            _render=render, _live=None, _throttle=Throttle(0), _enabled=False, _console=con
        )
        yield display
        # Print the final state once, so a pipe still gets the per-result detail
        # rather than only the summary line the caller prints afterwards. The
        # README promises "one clean line per result" for a pipe or a CI log, and
        # this used to deliver the header, one aggregate, and the output path:
        # which URLs failed was visible nowhere.
        #
        # Regardless of `transient`: with animation off there is no Live holding
        # a frame, so erasing one is not a concern — and the callers that pass
        # `transient=False` (the engine does, so its summary survives) are exactly
        # the ones whose aggregate line says nothing about individual results.
        # The per-result lines the engine emits as rows finish are the detail here,
        # so the final table would only repeat them. `contextlib.suppress` because
        # display code must never be able to end a run.
        return

    display = LiveDisplay(
        _render=render, _live=None, _throttle=Throttle(per_second), _enabled=True, _console=con
    )
    try:
        # auto_refresh=False, so the Throttle above is the *only* thing that
        # repaints. Rich's background refresh thread repaints the retained
        # renderable on its own schedule, which is not the schedule the throttle
        # governs: measured on a 300-row table, 35 throttled renders produced 70
        # prints. For the batch scraper that unthrottled half dominated the run.
        with Live(
            console=con,
            refresh_per_second=per_second,
            transient=transient,
            auto_refresh=False,
        ) as live:
            display._live = live
            display.update(force=True)
            yield display
            # Guarantee the final state is on screen even if the last event
            # arrived inside the rate-limit window.
            display.update(force=True)
    except KeyboardInterrupt:
        # Ctrl-C must not leave a half-drawn frame behind. Leaving the Live
        # context restores the screen; the caller decides whether to report the
        # cancellation or re-raise.
        raise


class StreamWriter:
    """
    Throttled writer for streaming text (an LLM response) to the terminal.

    Printing every token is the obvious approach and the wrong one: each print
    is a full render pass, and a 4k-token answer costs tens of thousands of
    them — measurable on the render alone, and it turns into visible flicker on
    a slow terminal. Chunks are coalesced and flushed on a time or size budget,
    which keeps output visibly live while the cost stays proportional to the
    response rather than to the token count.
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        per_second: float = 15.0,
        min_chars: int = 24,
        style: str = "grey85",
    ) -> None:
        self._console = console or _console
        self._throttle = Throttle(per_second)
        self._min_chars = min_chars
        self._style = style
        self._buf: list[str] = []
        self._pending = 0

    def write(self, chunk: str) -> None:
        """Buffer *chunk*, flushing when the time or size budget is reached."""
        if not chunk:
            return
        self._buf.append(chunk)
        self._pending += len(chunk)
        if self._pending >= self._min_chars and self._throttle.ready():
            self.flush()

    def flush(self) -> None:
        """Emit everything buffered. Safe to call when nothing is buffered."""
        if not self._buf:
            return
        text = "".join(self._buf)
        self._buf.clear()
        self._pending = 0
        # A model can emit escape sequences; never let them reach the terminal.
        self._console.print(
            text.replace("\x1b", ""), end="", style=self._style, markup=False, highlight=False
        )

    def __enter__(self) -> StreamWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.flush()


# ── failure reasons ───────────────────────────────────────────────────────────

#: Distinct causes shown before the tail is folded into a count.
MAX_REASONS_SHOWN = 6


def normalise_reason(note: str) -> str:
    """
    Collapse a failure note to its cause.

    The engine records one note per URL, so "HTTP 403 for https://a/b" and
    "HTTP 403 for https://c/d" are the same failure and want to be one line. Only
    the URL is collapsed.

    The status code used to be collapsed too, and that lost the answer: 403, 404
    and 500 are three different problems with three different remedies, so a run
    dominated by missing pages came out indistinguishable from one being
    rate-limited. That is the question this summary exists to answer.
    """
    return re.sub(r"\s+", " ", re.sub(r"https?://\S+", "<url>", note)).strip()


def print_failure_reasons(counts: Mapping[str, int]) -> None:
    """
    Print accumulated failure causes, most common first.

    Takes a mapping rather than the notes themselves, because the two callers
    accumulate differently: the scraper holds every row of the run, while the
    crawler keeps only the last 200 in its live log and has to count as it goes
    or it misreports every failure earlier in a long crawl.

    Only `scrape` had this at all. The crawler recorded a note on every failed
    row and read none of them back, so a run could report "6 errors" and leave the
    user to guess between DNS failure, HTTP 403, a timeout and robots.txt.
    """
    if not counts:
        return

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ranked[:MAX_REASONS_SHOWN]
    hidden = len(ranked) - len(shown)

    # A Table rather than an f-string, because the reason is an exception message
    # with no length bound and an unbounded value in an f-string is exactly what
    # makes a line wrap at the left margin and read as a separate fact. A column
    # folds inside itself, keeping the count and the cause visually attached.
    table = SafeTable(box=None, show_header=False, show_edge=False, padding=(0, 2))
    table.add_column(ERR, justify="right", width=3, style="red")
    table.add_column("count", justify="right", width=5, style="grey74")
    table.add_column("reason", ratio=1, overflow="fold", style="grey50")
    for reason, count in shown:
        table.add_row(ERR, f"{count:>5}", muted(reason))
    if hidden > 0:
        table.add_row("", "", muted(f"+ {hidden} more distinct reason(s)"))

    _console.print()
    _console.print(table)


#: Assumed terminal height when nothing knows better. Rich reports 25 for a
#: non-terminal stream, which is a reasonable guess for a log.
_DEFAULT_HEIGHT = 25

#: Never show fewer than this many rows, however short the terminal: an empty
#: table tells the user less than a truncated one.
_MIN_VISIBLE_ROWS = 3


def visible_rows(reserved: int, *, ceiling: int, height: int | None = None) -> int:
    """
    How many table rows fit the terminal once *reserved* lines are spent.

    Both live views used fixed row counts that had nothing to do with the window:
    the crawler's rendered 29 lines and the batch table 28, on a terminal that is
    24. A live region taller than the screen scrolls its own top out of view, so
    the user was left watching a log table with no bar, no percentage and no
    counts — and in the batch table's case with the JS column dropped entirely,
    because its columns needed 81 and 80 is the canonical width.

    *reserved* is everything else the render spends: rules, headers and the stat
    block. *ceiling* keeps a tall terminal from rendering thousands of rows.

    *height* overrides the console's, so a caller that knows the window — and a
    test — does not have to mutate global state to say so.
    """
    rows_available = (height if height is not None else _console.height) or _DEFAULT_HEIGHT
    return max(_MIN_VISIBLE_ROWS, min(ceiling, rows_available - reserved))


@contextmanager
def probing(message: str, con: Console | None = None) -> Iterator[None]:
    """
    Say that a slow probe is under way, and degrade honestly if it is not.

    `protor runtimes` checks every registered runtime in turn — seventeen of them,
    one HTTP request each, a second apiece behind a firewall that DROPs rather than
    refuses. It printed its heading and then sat there, so a blank screen for several
    seconds is indistinguishable from a hang.

    Rich's ``status`` animates on a terminal and prints *nothing at all* to a pipe,
    which would leave a CI log exactly as blank as before. So both renderings come
    from one message: a spinner where there is a cursor, one plain line where there
    is not.
    """
    target = con or _console
    if target.is_terminal:
        with target.status(message):
            yield
        return
    target.print(message)
    yield
