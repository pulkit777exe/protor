"""Terminals that cannot encode the glyphs must degrade, not crash.

A cp1252 console cannot encode the status marks and an ASCII pipe cannot encode
anything at all. With fixed glyphs, `protor models` died with a
UnicodeEncodeError traceback on those terminals — the one place a user least
expects a stack trace, since they only asked to see a model list.
"""

import pytest

from protor import theme


class TestEncodingAwareTokens:
    def test_tokens_never_contain_unencodable_characters(self):
        """Every token must be encodable by the terminal it is about to print to."""
        encoding = theme._output_encoding()
        for token in (theme.OK, theme.ERR, theme.SPIN, theme.ARROW, theme.SKIP):
            token.encode(encoding)

    def test_glyphs_are_pretty_when_the_terminal_allows_it(self):
        """A normal UTF-8 terminal keeps the intended look."""
        if theme._can_encode("✓"):
            assert theme.OK == "✓"
            assert theme.ERR == "✗"


class TestSafe:
    def test_ascii_text_passes_through(self):
        assert theme.safe("plain text") == "plain text"

    def test_encodable_glyphs_pass_through(self):
        if theme._can_encode("✓"):
            assert theme.safe("done ✓") == "done ✓"

    def test_fallbacks_are_themselves_always_safe(self, monkeypatch):
        """The replacement text must not contain glyphs needing replacement."""
        for _fancy, plain in theme._FALLBACKS:
            plain.encode("ascii")
        assert ("…", "...") in theme._FALLBACKS

    def test_degrades_when_the_terminal_cannot_encode(self, monkeypatch):
        monkeypatch.setattr(theme, "_output_encoding", lambda: "ascii")
        assert theme.safe("ok ✓") == "ok +"
        assert theme.safe("a — b") == "a - b"

    def test_empty_and_none_like_input(self):
        assert theme.safe("") == ""


class TestPrintHelpersSanitize:
    @pytest.mark.parametrize(
        "helper",
        [
            theme.dim,
            theme.muted,
            theme.label,
            theme.bright,
            theme.ok,
            theme.err,
            theme.warn,
            theme.info,
        ],
    )
    def test_helper_output_is_encodable(self, helper, monkeypatch):
        """
        The *argument* is re-checked here. The leading glyph is fixed at import,
        which is already validated against the real terminal by
        TestEncodingAwareTokens.
        """
        monkeypatch.setattr(theme, "_output_encoding", lambda: "ascii")
        argument = "a ✓ b — c"
        assert (
            helper(argument)
            .replace(theme.OK, "")
            .replace(theme.ERR, "")
            .replace(theme.ARROW, "")
            .encode("ascii")
        )

    @pytest.mark.parametrize("helper", [theme.dim, theme.muted, theme.label, theme.bright])
    def test_argument_glyphs_are_replaced(self, helper, monkeypatch):
        monkeypatch.setattr(theme, "_output_encoding", lambda: "ascii")
        assert "a + b - c" in helper("a ✓ b — c")

    @pytest.mark.parametrize("rule", [theme.header_rule, theme.section_rule])
    def test_rules_are_encodable(self, rule, monkeypatch):
        monkeypatch.setattr(theme, "_output_encoding", lambda: "ascii")
        from io import StringIO

        from rich.console import Console

        con = Console(file=StringIO(), width=80, force_terminal=False)
        con.print(rule("Protor — Analyzer ✓"))


class TestSafeIsTotal:
    """
    `theme.safe()` must return text the terminal can actually encode.

    The substitution table names the glyphs this module uses. Scraped page text
    and model output name none of them, and a character the table does not know
    came straight back — so `é` on an ASCII terminal, or a `♠` from a page, still
    raised at the write. That is the crash `theme.safe()` exists to prevent, so the
    guarantee is total rather than best-effort.
    """

    @pytest.mark.parametrize(
        "text",
        ["♠", "café", "naïve", "日本", "→→→", "\U0001d518\U0001d52d", "emoji \U0001f642 here"],
    )
    @pytest.mark.parametrize("encoding", ["ascii", "cp1252", "latin-1", "cp437"])
    def test_the_result_always_encodes(self, text, encoding, monkeypatch):
        monkeypatch.setattr(theme, "_output_encoding", lambda: encoding)
        try:
            theme.safe(text).encode(encoding)
        except UnicodeEncodeError as exc:  # pragma: no cover - the failure itself
            pytest.fail(f"theme.safe({text!r}) is not encodable in {encoding}: {exc}")

    def test_a_substitutable_glyph_still_reads_as_its_fallback(self, monkeypatch):
        """The table runs first, so the common glyphs keep their readable form."""
        monkeypatch.setattr(theme, "_output_encoding", lambda: "ascii")
        assert theme.safe("crawl — complete ✓") == "crawl - complete +"

    def test_text_that_already_fits_is_returned_untouched(self, monkeypatch):
        monkeypatch.setattr(theme, "_output_encoding", lambda: "ascii")
        original = "plain ascii text"
        assert theme.safe(original) is original


class TestConsoleWritesWhatTheTerminalCanEncode:
    """
    The console is the chokepoint that cannot be forgotten.

    `rich.console.Console` takes no `errors` parameter, so there was nowhere to
    ask for replacement at the write. A `Table` renders its cells without ever
    passing them through `print`, which is why a model name or page title in a
    cell raised `UnicodeEncodeError` from the middle of the render — and took the
    error report printed after it down too.
    """

    @pytest.mark.parametrize("encoding", ["ascii", "cp1252", "cp437"])
    def test_a_table_cell_that_cannot_encode_still_prints(self, encoding, tmp_path):
        import subprocess
        import sys

        script = tmp_path / "render.py"
        script.write_text(
            "from rich.table import Table\n"
            "from protor.theme import console\n"
            "t = Table('model')\n"
            "t.add_row('café — 日本 model')\n"
            "console.print(t)\n"
            "console.print('♠ plain f-string')\n",
            encoding="utf-8",
        )
        done = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": encoding},
            timeout=120,
        )
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        assert done.stdout, "nothing was written"

    def test_the_wrapper_is_transparent_to_the_console(self):
        """isatty and fileno must reach the real stream or Rich stops detecting."""
        import sys

        wrapped = theme.console.file
        assert wrapped.isatty() == sys.stdout.isatty()
        assert wrapped.fileno() == sys.stdout.fileno()

    def test_utf8_output_is_untouched(self, tmp_path):
        """Degrading must not cost anything on a terminal that can encode it."""
        import subprocess
        import sys

        script = tmp_path / "render.py"
        script.write_text(
            "from protor.theme import console, safe\n"
            "console.print('café — 日本 ✓')\n"
            "assert safe('café — 日本 ✓') == 'café — 日本 ✓'\n",
            encoding="utf-8",
        )
        done = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
            timeout=120,
        )
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        assert "café — 日本 ✓" in done.stdout.decode("utf-8")
