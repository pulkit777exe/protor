"""Tests for protor.progress — live rendering, throttling and streaming.

These decide what the user sees while long work runs, and whether piped output
stays readable, so both paths are asserted rather than assumed.
"""

import io

import pytest
from rich.console import Console

from protor.progress import (
    LiveDisplay,
    RunState,
    StreamWriter,
    Throttle,
    live_display,
    live_enabled,
    status_line,
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
        Updates cost nothing when live is disabled; only the exit render runs.

        Piped output must not receive cursor-up escapes, and it must not be
        re-rendered per event either — the single render on exit is what prints
        the per-result detail a pipe is promised, not each update.
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
        assert calls == [1], "exactly one render, on exit"

    def test_renders_the_final_state_on_exit(self):
        """Whatever the rate limit did, the last frame is on screen."""
        renders = {"n": 0}

        def render():
            renders["n"] += 1
            return f"frame {renders['n']}"

        with live_display(render, console=_console(), enabled=False) as display:
            assert display.state is RunState.IDLE
            assert renders["n"] == 0, "nothing is rendered while the block runs"
        # Disabled mode renders once, on exit, and prints it: the README promises
        # a pipe "one clean line per result", and a header plus an aggregate left
        # the user no record of which URLs failed.
        assert renders["n"] == 1

    def test_state_transitions_are_tracked(self):
        with live_display(lambda: "", console=_console()) as display:
            display.state = RunState.WORKING
            assert display.state is RunState.WORKING
            display.state = RunState.DONE
            assert display.state is RunState.DONE

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


# ── status line ───────────────────────────────────────────────────────────────


class TestStatusLine:
    @pytest.mark.parametrize(
        ("state", "needle"),
        [
            (RunState.WORKING, "working"),
            (RunState.STREAMING, "streaming"),
            (RunState.DONE, "done"),
            (RunState.FAILED, "failed"),
            (RunState.CANCELLED, "cancelled"),
        ],
    )
    def test_state_is_named_in_the_line(self, state, needle):
        assert needle in status_line(state).plain

    def test_detail_is_appended(self):
        assert "12/40 pages" in status_line(RunState.WORKING, "12/40 pages").plain

    def test_run_state_values_are_stable_strings(self):
        """Values reach the CLI and saved output, so they are part of the API."""
        assert RunState.WORKING == "working"
        assert RunState.DONE == "done"


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
    A pipe gets the detail, not only the caller's summary line.

    The README promises `protor scrape ... | tee log` writes "one clean line per
    result". It did not: with animation off, ``render()`` was never called, so a
    piped run printed the header, one aggregate, and the index path — which URLs
    failed appeared nowhere.
    """

    def test_the_final_state_is_printed_once(self, capsys):
        console = Console(file=None, width=80, force_terminal=False, legacy_windows=False)
        console.file = io.StringIO()
        with live_display(lambda: "row for https://a.example", console=console, enabled=False):
            pass
        assert console.file.getvalue().count("row for https://a.example") == 1

    def test_transient_false_still_prints_when_disabled(self):
        """
        The engine passes transient=False, so its summary survives.

        With animation off there is no Live holding a frame to erase, and that
        caller's aggregate line is exactly the one that says nothing about which
        individual URLs failed — so the detail is printed either way.
        """
        console = Console(file=None, width=80, force_terminal=False, legacy_windows=False)
        console.file = io.StringIO()
        with live_display(lambda: "row", console=console, enabled=False, transient=False):
            pass
        assert console.file.getvalue().count("row") == 1

    def test_a_failing_render_does_not_break_the_command(self):
        """The work is already done; a render error must not lose the summary."""

        def boom():
            raise RuntimeError("render exploded")

        console = Console(file=None, width=80, force_terminal=False, legacy_windows=False)
        console.file = io.StringIO()
        with live_display(boom, console=console, enabled=False):
            pass  # must not raise
