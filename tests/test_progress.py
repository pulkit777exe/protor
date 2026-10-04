"""Tests for protor.progress — live rendering, throttling and streaming.

These decide what the user sees while long work runs, and whether piped output
stays readable, so both paths are asserted rather than assumed.
"""

import io

import pytest
from rich.console import Console

from protor.progress import (
    LiveDisplay,
    StreamWriter,
    Throttle,
    live_display,
    live_enabled,
)


def _console() -> Console:
    return Console(file=io.StringIO(), width=80, force_terminal=False, legacy_windows=False)


def _terminal_console() -> Console:
    """A console that looks like a real utf-8 terminal, for the env-flag tests."""
    return Console(file=io.StringIO(), width=80, force_terminal=True, legacy_windows=False)


# ── throttle ──────────────────────────────────────────────────────────────────


class TestThrottle:
    def test_allows_the_first_call(self):
        assert Throttle(10).ready() is True

    def test_blocks_an_immediate_second_call(self):
        t = Throttle(10)
        t.ready()
        assert t.ready() is False

    def test_force_bypasses_the_limit(self):
        """The final update of a run must never be swallowed by the rate limit."""
        t = Throttle(1)
        t.ready()
        assert t.ready(force=True) is True

    def test_zero_rate_disables_throttling(self):
        t = Throttle(0)
        assert t.ready() and t.ready()

    def test_allows_again_after_the_interval(self, monkeypatch):
        clock = {"now": 0.0}
        monkeypatch.setattr("protor.progress.time.monotonic", lambda: clock["now"])
        t = Throttle(10)  # 0.1 s interval
        assert t.ready() is True
        clock["now"] = 0.05
        assert t.ready() is False
        clock["now"] = 0.11
        assert t.ready() is True


# ── live detection ────────────────────────────────────────────────────────────


class TestLiveEnabled:
    def test_disabled_when_stdout_is_not_a_terminal(self):
        assert live_enabled(_console()) is False

    def test_explicit_opt_out_wins(self, monkeypatch):
        monkeypatch.setenv("PROTOR_NO_LIVE", "1")
        assert live_enabled(_console()) is False

    def test_disabled_in_ci(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        assert live_enabled(_console()) is False


# ── live_display ──────────────────────────────────────────────────────────────


class TestLiveDisplay:
    def test_update_is_a_no_op_when_disabled(self):
        """
        Updates cost nothing when live is disabled.

        Piped output must not receive cursor-up escapes, and it must not be
        re-rendered per event either. Detail reaches a pipe through
        :meth:`LiveDisplay.line` instead, once per result, as it happens.
        """
        calls: list[int] = []

        def render():
            calls.append(1)
            return "frame"

        with live_display(render, console=_console()) as display:
            display.update()
            display.update()
            display.update()
            assert calls == [], "no per-event rendering when disabled"
            display.line("one line per result")
            display.line("and another")
        assert calls == [], "the frame is never rendered for a pipe"

    def test_a_disabled_display_writes_one_plain_line_per_call(self):
        """
        The promise the README makes for `protor scrape ... | tee log`.

        A pipe used to get a header, silence for the length of the run, and a
        table at the end — which URLs had failed appeared nowhere until the end,
        and nothing at all appeared while the work ran.
        """
        console = _console()
        with live_display(lambda: "frame", console=console, enabled=False) as display:
            display.line("first result")
            display.line("second result")

        out = console.file.getvalue()
        assert out.count("first result") == 1
        assert out.count("second result") == 1
        assert "\x1b" not in out, "a pipe must not receive escape codes"

    def test_renders_the_final_state_on_exit(self):
        """Whatever the rate limit did, the last frame is on screen."""
        renders = {"n": 0}

        def render():
            renders["n"] += 1
            return f"frame {renders['n']}"

        # A terminal console, or the display disables itself and the live path is
        # never exercised at all.
        with live_display(render, console=_terminal_console()) as display:
            # Entering the block draws once, so the frame is on screen before any
            # work has happened rather than after the first event.
            assert renders["n"] == 1, "the first frame is drawn on entry"
            display.update(force=True)
            assert renders["n"] >= 2, "a forced update redraws"
        assert renders["n"] >= 2, "the last frame is on screen at exit"

    def test_note_writes_through(self):
        buf = io.StringIO()
        con = Console(file=buf, width=80, force_terminal=False)
        with live_display(lambda: "", console=con, enabled=False) as display:
            display.note("hello")
        assert "hello" in buf.getvalue()

    def test_handle_reports_whether_it_is_live(self):
        with live_display(lambda: "", console=_console(), enabled=False) as display:
            assert isinstance(display, LiveDisplay)
            assert display._enabled is False


# ── streaming ─────────────────────────────────────────────────────────────────


class TestStreamWriter:
    def _capture(self):
        buf = io.StringIO()
        return buf, Console(file=buf, width=80, force_terminal=False, legacy_windows=False)

    def test_writes_everything_by_the_time_the_block_exits(self):
        buf, con = self._capture()
        with StreamWriter(console=con, per_second=0.001) as writer:
            for _ in range(500):
                writer.write("chunk")
        assert "chunk" in buf.getvalue()

    def test_buffers_until_the_size_budget(self):
        """Many tiny chunks must not each cost a render."""
        buf, con = self._capture()
        writer = StreamWriter(console=con, per_second=1000, min_chars=10_000)
        for _ in range(100):
            writer.write("x")
        assert buf.getvalue() == "", "should still be buffered"
        writer.flush()
        # Rich wraps at the console width, so compare the content, not the layout.
        assert buf.getvalue().replace("\n", "") == "x" * 100

    def test_escape_sequences_from_the_model_are_stripped(self):
        """A model emitting raw escapes would otherwise repaint the terminal."""
        buf, con = self._capture()
        with StreamWriter(console=con) as writer:
            writer.write("before\x1b[2Jafter")
        out = buf.getvalue()
        assert "\x1b" not in out
        assert "before" in out and "after" in out

    def test_markup_is_not_interpreted(self):
        buf, con = self._capture()
        with StreamWriter(console=con) as writer:
            writer.write("[docs](https://x.com) and [/oops]")
        out = buf.getvalue()
        assert "[docs](https://x.com)" in out
        assert "closing" not in out

    def test_empty_chunks_are_ignored(self):
        buf, con = self._capture()
        writer = StreamWriter(console=con)
        writer.write("")
        writer.flush()
        assert buf.getvalue() == ""

    def test_flush_with_nothing_buffered_is_safe(self):
        buf, con = self._capture()
        StreamWriter(console=con).flush()
        assert buf.getvalue() == ""

    def test_chunk_order_is_preserved(self):
        buf, con = self._capture()
        with StreamWriter(console=con, per_second=0.001, min_chars=8) as writer:
            for word in ("alpha ", "beta ", "gamma ", "delta"):
                writer.write(word)
        out = buf.getvalue()
        assert out.index("alpha") < out.index("beta") < out.index("gamma") < out.index("delta")


class TestEnvironmentFlags:
    """
    Both switches answer "is it set to off?" the same way.

    Bare truthiness made ``PROTOR_NO_LIVE=0`` disable live rendering while
    ``CI=0`` did not — the same question answered two ways, three lines apart,
    and the first is never what someone setting a variable to 0 meant.
    """

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything"])
    def test_an_explicit_on_disables_live(self, value, monkeypatch):
        monkeypatch.setenv("PROTOR_NO_LIVE", value)
        monkeypatch.delenv("CI", raising=False)
        assert live_enabled(_terminal_console()) is False

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_an_explicit_off_leaves_live_enabled(self, value, monkeypatch):
        monkeypatch.setenv("PROTOR_NO_LIVE", value)
        monkeypatch.delenv("CI", raising=False)
        assert live_enabled(_terminal_console()) is True

    @pytest.mark.parametrize("value", ["0", "false", ""])
    def test_ci_off_does_not_disable_live(self, value, monkeypatch):
        monkeypatch.delenv("PROTOR_NO_LIVE", raising=False)
        monkeypatch.setenv("CI", value)
        assert live_enabled(_terminal_console()) is True

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_ci_on_disables_live(self, value, monkeypatch):
        monkeypatch.delenv("PROTOR_NO_LIVE", raising=False)
        monkeypatch.setenv("CI", value)
        assert live_enabled(_terminal_console()) is False


class TestDisabledDisplayStillPrintsTheResult:
    """
    A pipe gets a line per result, not only the caller's summary line.

    The README promises `protor scrape ... | tee log` writes "one clean line per
    result". It did not: with animation off nothing was written while the work ran
    at all, so a ten-minute scrape logged a header, then ten minutes of nothing,
    then a block. The engine now writes each finished row through
    :meth:`LiveDisplay.line`.
    """

    def test_each_line_is_printed_once_as_it_arrives(self, capsys):
        console = Console(file=None, width=80, force_terminal=False, legacy_windows=False)
        console.file = io.StringIO()
        with live_display(lambda: "never rendered", console=console, enabled=False) as display:
            display.line("row for https://a.example")
            display.line("row for https://b.example")

        out = console.file.getvalue()
        assert out.count("row for https://a.example") == 1
        assert out.count("row for https://b.example") == 1
        assert "never rendered" not in out, "the table would only repeat the lines"

    def test_transient_false_still_prints_when_disabled(self):
        """
        The engine passes transient=False, so its summary survives.

        With animation off there is no Live holding a frame to erase, so a result
        line is the only record of an individual URL — which is the whole point.
        """
        console = Console(file=None, width=80, force_terminal=False, legacy_windows=False)
        console.file = io.StringIO()
        with live_display(
            lambda: "row", console=console, enabled=False, transient=False
        ) as display:
            display.line("row")

        assert console.file.getvalue().count("row") == 1

    def test_a_failing_render_does_not_break_the_command(self):
        """The work is already done; a render error must not lose the summary."""

        def boom():
            raise RuntimeError("render exploded")

        console = Console(file=None, width=80, force_terminal=False, legacy_windows=False)
        console.file = io.StringIO()
        with live_display(boom, console=console, enabled=False):
            pass  # must not raise


class TestTheBatchLiveTableIsBounded:
    """
    The batch progress table rebuilt and repainted every row on every tick.

    Rich's own render of the resulting table measured 27 ms at 300 rows, 587 ms at
    1,000 and 1,839 ms at 3,000. On a real 300-URL batch, rendering the progress
    bar was 67% of the wall clock; at 600 URLs, 89%. The crawler bounded its log
    to the last 20 rows; the batch path never got the same treatment, and had no
    cap at all.

    The durable summary printed at the end still carries every row — this is only
    about what is repainted while work is in flight.
    """

    def test_a_huge_batch_renders_a_bounded_table(self):
        from protor.scraper import _TABLE_VIEW, _build_table

        rows = [
            {
                "idx": i,
                "domain": f"site{i}.example",
                "status": "done",
                "bytes": 10,
                "ms": 1,
                "js": 0,
            }
            for i in range(3000)
        ]
        # An explicit height, because the row budget is now derived from the
        # window rather than fixed: this test is about the cap, and a cap that
        # moves with the terminal needs the terminal pinned to say anything.
        from protor.scraper import _TABLE_RESERVED

        height = 60
        out = io.StringIO()
        console = Console(
            file=out, width=120, height=height, force_terminal=False, legacy_windows=False
        )
        console.print(_build_table(rows, width=120, height=height))

        text = out.getvalue()
        budget = min(_TABLE_VIEW, height - _TABLE_RESERVED)
        assert text.count("site") == budget, f"{text.count('site')} rows rendered"
        assert f"{3000 - budget} earlier rows" in text, "the elision is not reported"
        assert text.count("site") <= _TABLE_VIEW, f"{text.count('site')} rows rendered"

    def test_a_small_batch_is_not_truncated(self):
        from protor.scraper import _build_table

        rows = [
            {
                "idx": i,
                "domain": f"site{i}.example",
                "status": "done",
                "bytes": 10,
                "ms": 1,
                "js": 0,
            }
            for i in range(5)
        ]
        out = io.StringIO()
        Console(file=out, width=120, force_terminal=False, legacy_windows=False).print(
            _build_table(rows)
        )
        text = out.getvalue()
        assert text.count("site") == 5
        assert "earlier" not in text

    def test_rendering_a_huge_batch_is_fast(self):
        """Bounded work, so the render cost stops tracking the batch size."""
        import time

        from protor.scraper import _build_table

        rows = [
            {
                "idx": i,
                "domain": f"site{i}.example",
                "status": "done",
                "bytes": 10,
                "ms": 1,
                "js": 0,
            }
            for i in range(3000)
        ]
        out = io.StringIO()
        console = Console(file=out, width=120, force_terminal=False, legacy_windows=False)
        start = time.perf_counter()
        console.print(_build_table(rows))
        elapsed = time.perf_counter() - start

        # Unbounded this measured 8,760 ms for 3,000 rows. A generous bound
        # still catches a regression to rendering everything.
        assert elapsed < 1.0, f"rendering 3,000 rows took {elapsed * 1e3:.0f}ms"

    def test_a_live_display_does_not_also_write_the_line(self, tmp_path):
        """
        The control on `line()`: while animating, the table already carries the row.

        Printing it as well would scroll one extra line past the live region for
        every page of a large crawl, which is the noise the table exists to avoid.
        """
        console = _terminal_console()
        with live_display(lambda: "frame", console=console) as display:
            assert display._live is not None, "this test needs the live path"
            display.line("a result line")

        assert "a result line" not in console.file.getvalue()
