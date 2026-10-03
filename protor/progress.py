"""
protor.progress
~~~~~~~~~~~~~~~
The single owner of live, stateful terminal rendering.

Everything that can take a while — scraping, crawling, waiting on an LLM —
renders through this module, so progress is one concept in one place instead of
ad-hoc ``Live`` blocks scattered across the engine, the crawler and the analyzer.

Three ideas drive the design, all borrowed from how production terminal agents
handle the same problems:

1. **One explicit state, one renderer.** :class:`RunState` is the only thing that
   describes "what is happening"; the display is a pure function of it. No
   component decides on its own what to draw.
2. **Rate-limit redraws, not work.** A display that repaints once per completed
   page is O(n²) over a batch and starves the event loop it is reporting on. A
   progress bar at 10 Hz is indistinguishable from one at 1000 Hz to a human and
   orders of magnitude cheaper.
3. **Degrade, never break.** A pipe, a CI log and a dumb terminal get plain
   sequential lines instead of cursor-up escape sequences. Animation is only
   ever attempted where it can actually work.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rich.live import Live
from rich.text import Text

from .theme import ERR, OK, SPIN
from .theme import console as _console

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import TracebackType

    from rich.console import Console

__all__ = [
    "RunState",
    "StreamWriter",
    "Throttle",
    "live_display",
    "live_enabled",
    "status_line",
]


class RunState(StrEnum):
    """What the tool is doing right now.

    Display code switches on this; it never infers state from timing or output.
    """

    IDLE = "idle"
    WORKING = "working"
    STREAMING = "streaming"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


def live_enabled(con: Console | None = None) -> bool:
    """
    Whether in-place rendering can work here.

    False for pipes, files, CI logs and ``TERM=dumb``: writing cursor-up
    sequences there produces an unreadable transcript of escape codes. Rich
    already honours ``NO_COLOR`` and ``TERM``; this adds the "is anyone
    watching" half of the question.
    """
    con = con or _console
    if os.environ.get("PROTOR_NO_LIVE"):
        return False
    if os.environ.get("CI", "").lower() in ("1", "true", "yes"):
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
    """

    _render: Callable[[], Any]
    _live: Live | None
    _throttle: Throttle
    _enabled: bool
    _console: Console
    _state: RunState = RunState.IDLE
    _notes: list[str] = field(default_factory=list)

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def state(self) -> RunState:
        """Current run state; the only input the renderer needs."""
        return self._state

    @state.setter
    def state(self, value: RunState) -> None:
        self._state = value
        self.update(force=True)

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
        yield LiveDisplay(
            _render=render, _live=None, _throttle=Throttle(0), _enabled=False, _console=con
        )
        return

    display = LiveDisplay(
        _render=render, _live=None, _throttle=Throttle(per_second), _enabled=True, _console=con
    )
    try:
        with Live(console=con, refresh_per_second=per_second, transient=transient) as live:
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


def status_line(state: RunState, detail: str = "") -> Text:
    """One-line state summary, e.g. a footer under a live table."""
    glyph = {
        RunState.DONE: f"[green]{OK}[/green]",
        RunState.FAILED: f"[red]{ERR}[/red]",
        RunState.CANCELLED: "[yellow]-[/yellow]",
    }.get(state, f"[cyan]{SPIN}[/cyan]")
    line = Text.from_markup(f"{glyph} {state.value}")
    if detail:
        line.append(f"  {detail}", style="grey50")
    return line
