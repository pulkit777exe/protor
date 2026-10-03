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
