"""
The leak detector guarding the test suite.

`_no_leaked_module_patches` in `tests/conftest.py` exists because a test once
patched `protor.engine.fetch` and restored `parse_html` in a `finally` but not
`fetch`. That passed on its own, and then failed a schema-extraction test in
another file with zero records, four hundred tests later. The failure reads as a
product bug in the extractor.

A detector that nobody checks still detects: it can stop finding things the day
the module layout changes, and it will report a clean suite while doing it.
"""

from __future__ import annotations

import io

import pytest

from protor import theme
from tests.conftest import _module_snapshot


def test_the_snapshot_notices_a_swapped_module_attribute():
    """
    The whole mechanism is comparing identities before and after a test, so
    that comparison is what has to be pinned.
    """
    import protor.engine as engine

    real = engine.parse_html
    before = _module_snapshot()
    assert before[("protor.engine", "parse_html")] == id(real)

    engine.parse_html = lambda *a, **k: None  # type: ignore[assignment]
    try:
        after = _module_snapshot()
        changed = [key for key, ident in after.items() if before.get(key, ident) != ident]
        assert ("protor.engine", "parse_html") in changed, changed
    finally:
        engine.parse_html = real

    assert _module_snapshot()[("protor.engine", "parse_html")] == id(real)


def test_the_snapshot_ignores_constants():
    """
    Constants are the one thing a test may legitimately rebind and leave, and
    caches are initialised on first use rather than leaked. Excluding both is what
    keeps the detector quiet enough to be worth running.
    """
    import protor.engine as engine
    from protor.config import RETRY_BACKOFF_BASE

    assert not any(key[1].isupper() for key in _module_snapshot()), "an uppercase attr slipped in"

    engine.RETRY_BACKOFF_BASE = RETRY_BACKOFF_BASE + 1
    try:
        assert not any(key[1].isupper() for key in _module_snapshot())
    finally:
        engine.RETRY_BACKOFF_BASE = RETRY_BACKOFF_BASE


def test_the_snapshot_covers_the_modules_that_get_patched():
    """If it misses a module, a leak in that module goes unreported."""
    names = {key[0] for key in _module_snapshot()}
    for module in ("protor.engine", "protor.scraper", "protor.markdown", "protor.parser"):
        assert module in names, f"{module} is not covered"


class TestStdoutIsTheReportAndStderrIsTheDiagnosis:
    """
    Which stream a line is on is part of the behaviour, not an accident.

    Everything went to stdout, so `protor scrape url > report.txt` interleaved
    "✗ HTTP 403" into the report and `2>/dev/null` could not silence a failure —
    neither of which is what a shell redirection is for. `rg`, `cargo`, `docker`
    and `kubectl` all split it the same way: the result on stdout, the diagnosis on
    stderr.
    """

    def _run(self, capsys, argv):
        import sys

        from protor.cli import cli

        old = sys.argv
        sys.argv = ["protor", *argv]
        try:
            with pytest.raises(SystemExit) as excinfo:
                cli()
            return excinfo.value.code, capsys.readouterr()
        finally:
            sys.argv = old

    def test_a_failure_explains_itself_on_stderr(self, capsys, tmp_path):
        code, captured = self._run(
            capsys, ["extract", "example.com/no-scheme", str(tmp_path / "s.json")]
        )
        assert code == 1
        assert "scheme" in captured.err, captured.err
        assert captured.out == "", f"the report stream carried a diagnostic:\n{captured.out!r}"

    def test_a_file_that_does_not_exist_says_so_on_stderr(self, capsys, tmp_path):
        code, captured = self._run(capsys, ["analyze", str(tmp_path / "nope.json")])
        assert code != 0
        assert captured.out == "", f"the report stream carried a diagnostic:\n{captured.out!r}"
        assert captured.err.strip(), "nothing was said at all"

    def test_a_successful_report_stays_on_stdout(self, capsys, tmp_path):
        """
        The control: routing everything to stderr would be just as wrong.

        `protor version` is a report, so it belongs where a script or a pipe can
        read it without asking for stderr.
        """
        import sys

        from protor.cli import cli

        old = sys.argv
        sys.argv = ["protor", "version"]
        try:
            cli()
            captured = capsys.readouterr()
        finally:
            sys.argv = old
        assert "protor" in captured.out, captured.out
        assert captured.err == "", f"a report went to stderr:\n{captured.err!r}"

    def test_the_two_consores_are_both_encoding_safe(self, monkeypatch):
        """
        A stderr console that is a plain `Console` would raise `UnicodeEncodeError`
        on the same terminals `theme.safe()` exists to protect — and only for the
        lines that happen to contain a glyph, which is the worst place to find out.
        """
        from protor import theme

        for name in ("console", "err_console"):
            assert isinstance(getattr(theme, name), theme.ProtorConsole), name
        assert theme.err_console.stderr is True
        assert theme.console.stderr is False

    def test_failure_reasons_are_a_diagnosis_not_a_result(self, capsys, tmp_path):
        """
        The other half of the split, and the one that was left inconsistent.

        `print_failure_reasons` answers "why did my run fail?" — it is a table of
        HTTP status codes and DNS failures. It was on stdout, so `protor crawl url >
        log` filled the log with them and a script piping stdout got a table of
        errors arriving as data, while every *other* diagnostic in the tool had
        already moved to stderr.
        """
        import protor.progress as progress_mod

        buf = io.StringIO()
        original = progress_mod._err_console
        progress_mod._err_console = theme.ProtorConsole(
            file=buf, width=100, force_terminal=False, highlight=False
        )
        try:
            progress_mod.print_failure_reasons({"blocked by robots.txt": 1})
        finally:
            progress_mod._err_console = original

        assert "robots.txt" in buf.getvalue(), buf.getvalue()

    def test_the_summary_stays_on_stdout(self, capsys, tmp_path):
        """The control: moving the reasons must not move the counts with them."""
        import sys

        from protor.cli import cli

        old = sys.argv
        sys.argv = ["protor", "version"]
        try:
            cli()
            captured = capsys.readouterr()
        finally:
            sys.argv = old
        assert captured.out.strip(), "the report stream is empty"
        assert captured.err == "", captured.err
