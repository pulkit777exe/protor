"""Terminals that cannot encode the glyphs must degrade, not crash.

A cp1252 console cannot encode the status marks and an ASCII pipe cannot encode
anything at all. With fixed glyphs, `protor models` died with a
UnicodeEncodeError traceback on those terminals — the one place a user least
expects a stack trace, since they only asked to see a model list.
"""

from io import StringIO

import pytest
from rich.console import Console
from rich.table import Table

from protor import theme


class TestEncodingAwareTokens:
    def test_tokens_never_contain_unencodable_characters(self):
        """Every token must be encodable by the terminal it is about to print to."""
        encoding = theme._output_encoding()
        for token in (theme.OK, theme.ERR, theme.ACTIVE, theme.ARROW, theme.SKIP):
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


class TestSquareBracketsSurviveRichMarkup:
    """
    Rich's markup parser and the terminal's encoder are different parsers.

    `theme.safe()` handled the encoder. Square brackets reach a second parser
    first: rich read `[slug]` in a URL as a style tag and dropped it, so the crawl
    view reported `https://ex.com/docs/` for a page actually at
    `https://ex.com/docs/[slug]` — no crash, just a wrong answer about which page
    was being fetched. `[..]` survived only because a dot is not a legal tag name,
    which is luck rather than a rule.

    The helpers therefore escape markup. They do not, and must not, escape in the
    console's own `print`: most output arrives there as an f-string of helpers that
    have already emitted their own tags, and escaping would strip every colour in
    the tool.
    """

    @pytest.mark.parametrize(
        "value",
        [
            "https://ex.com/docs/[slug]",
            "https://ex.com/a[b]c",
            "https://ex.com/p/[id]",
            "https://ex.com/[org]/[repo]/issues",
            "match [and group] here",
            "[not a tag]",
        ],
    )
    def test_the_text_arrives_intact(self, value):
        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, width=200, force_terminal=False)
        for helper in (
            theme.muted,
            theme.label,
            theme.bright,
            theme.ok,
            theme.err,
            theme.warn,
            theme.info,
        ):
            buf.truncate(0)
            buf.seek(0)
            con.print(helper(value))
            assert value in buf.getvalue(), f"{helper.__name__} ate part of {value!r}"

    def test_markup_is_still_interpreted_where_it_is_meant_to_be(self):
        """
        The guard against over-escaping.

        `content()` is right for the helpers and wrong for the console's `print`,
        which is handed markup. If this ever breaks, the tool loses all colour.
        """
        import re

        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, width=80, force_terminal=True, color_system="truecolor")
        con.print(f"{theme.label('saved')} {theme.muted('/tmp/x')}")
        out = buf.getvalue()
        assert "\x1b[" in out, "no styling reached the terminal"
        # Rich wraps each styled run separately, so the escape codes sit *inside*
        # the text; compare against the plain rendering.
        assert re.sub(r"\x1b\[[0-9;]*m", "", out) == "saved /tmp/x\n"

    def test_the_escaped_form_is_still_readable_text(self):
        """Escaping must not put backslashes on screen."""
        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, width=200, force_terminal=False)
        con.print(theme.muted("https://ex.com/docs/[slug]"))
        out = buf.getvalue()
        assert "\\" not in out, f"the escape leaked into the output: {out!r}"
        assert "[slug]" in out


class TestTableCellsAreSanitisedToo:
    """
    A Table renders its cells without ever passing them through `print`.

    `ProtorConsole` is the chokepoint for everything printed and `_EncodingSafeFile`
    is the last line of defence for everything written. A table cell was neither: the
    helpers never saw it, and by the time the file wrapper could act the string had
    already lost its glyphs to `errors="replace"`. So a glyph in a cell became `?` on
    a cp1252 terminal while the same glyph through `muted()` became `o` — the same
    character meaning two different things on the same screen.

    `SafeTable` closes the gap at the source, the way `ProtorConsole` does for
    `print`. Most cells already pass through a helper and so are unaffected; this is
    about the ones that do not, today or in the next change that adds one.

    The glyph is written literally rather than taken from a module token, because the
    tokens are *already* degraded by the time a cell could use one — using one would
    make the two tables agree for the wrong reason.
    """

    GLYPH = "\u25cc"  # a dotted circle: cp1252 cannot encode it

    def _render(self, table_class, monkeypatch, encoding):
        monkeypatch.setattr(theme, "_output_encoding", lambda: encoding)
        table = table_class(box=None, show_header=False, show_edge=False)
        table.add_column("Domain", no_wrap=True)
        table.add_row(f"x {self.GLYPH} y")
        buf = StringIO()
        Console(file=buf, width=120, force_terminal=False).print(table)
        return buf.getvalue()

    def test_a_bare_cell_degrades_like_every_other_glyph(self, monkeypatch):
        safe_out = self._render(theme.SafeTable, monkeypatch, "cp1252")
        plain_out = self._render(Table, monkeypatch, "cp1252")

        assert self.GLYPH not in safe_out, f"the cell kept an unencodable glyph: {safe_out!r}"
        assert "?" not in safe_out, f"the cell did not degrade at all: {safe_out!r}"
        # The plain table cannot help: the write layer has already lost the glyph.
        assert plain_out != safe_out, "a plain Table already did this"

    def test_it_leaves_encodable_text_alone(self, monkeypatch):
        out = self._render(theme.SafeTable, monkeypatch, "utf-8")
        assert "x \u25cc y" in out, out

    def test_it_keeps_the_styling_cells_deliberately_carry(self):
        """`safe`, not `content`: escaping a cell would eat the tags and the styles."""
        from rich.text import Text

        buf = StringIO()
        table = theme.SafeTable(box=None, show_header=False, show_edge=False)
        table.add_column("Status")
        table.add_row(Text("styled", style="green"))
        Console(file=buf, width=120, force_terminal=True, color_system="truecolor").print(table)
        out = buf.getvalue()
        assert "styled" in out
        assert "\x1b[" in out, "the style was stripped"

    def test_every_table_the_tool_builds_is_one_of_these(self):
        """
        Pin the wiring, not just the class.

        A cell is only sanitised if the table is a `SafeTable`, and nothing stops
        the next change from reaching for `rich.table.Table` directly — the type
        checker is happy either way and every existing test passes. So: the scraper's
        table, the crawler's tables, and a check that `theme` is the only module that
        imports rich's Table at all.
        """
        import pathlib

        import protor.analyzer
        import protor.crawler
        import protor.scraper
        from protor.scraper import _build_table

        table = _build_table([{"idx": 1, "domain": "ex.com", "status": "done"}])
        assert isinstance(table, theme.SafeTable)

        from collections import deque

        from protor.crawler import _CrawlLog, _render, _State

        state = _State(log=deque([_CrawlLog("ok", "ex.com", url="u")], maxlen=10))
        tables = [r for r in _render(state, "/tmp/out").renderables if hasattr(r, "columns")]
        assert tables, "the crawl view built no table"
        for built in tables:
            assert isinstance(built, theme.SafeTable), type(built)

        package = pathlib.Path(theme.__file__).parent
        offenders = [
            f.name
            for f in package.glob("*.py")
            if f.name != "theme.py" and "rich.table" in f.read_text(encoding="utf-8")
        ]
        assert not offenders, f"these reach past SafeTable: {offenders}"
        assert protor.analyzer and protor.crawler and protor.scraper


class TestStaleTokensStillDegrade:
    """
    The glyph tokens are decided at import; that is not the safety mechanism.

    `OK = "✓" if _can_encode("✓") else "+"` reads `sys.stdout` once, at import.
    Swapping stdout afterwards — a daemon, an embedding app, a test harness — leaves
    them stale, and the module's own framing suggested they were load-bearing. They
    are not: `ProtorConsole.print` routes every string through `safe()`, whose
    substitution table is keyed on the literal glyphs rather than on these names.

    Pinned because it is an emergent property of two independent mechanisms. Route the
    tokens around `safe()`, or key the table on the token instead of the glyph, and
    the protection disappears without anything failing.
    """

    def _render(self, monkeypatch, encoding, text):
        monkeypatch.setattr(theme, "_output_encoding", lambda: encoding)
        buf = StringIO()
        # ProtorConsole, not a plain Console: the degradation is its doing, and a
        # plain one would show this test asserting on rich rather than on protor.
        theme.ProtorConsole(file=buf, width=80, force_terminal=False).print(text)
        return buf.getvalue()

    def test_a_token_decided_for_another_terminal_still_degrades(self, monkeypatch):
        stale = "\u2713"  # what OK would be if stdout had been utf-8 at import
        assert not stale.isascii(), "the token should be unencodable for cp1252"

        out = self._render(monkeypatch, "cp1252", f"  {stale} done")

        assert out.isascii(), f"a stale token reached a cp1252 terminal: {out!r}"
        assert "+ done" in out, out

    def test_it_is_the_table_doing_it_not_the_token(self, monkeypatch):
        """A glyph the table does not name degrades too, so nothing is token-specific."""
        out = self._render(monkeypatch, "ascii", "  \u2660 done")
        assert out.isascii(), out
        assert "\u2660" not in out, out

    def test_on_an_encodable_terminal_nothing_is_substituted(self, monkeypatch):
        """The guard must not cost a UTF-8 terminal its glyphs."""
        out = self._render(monkeypatch, "utf-8", "  \u2713 done")
        assert "\u2713" in out, out

    def test_the_table_covers_the_glyphs_the_module_uses(self):
        """
        Why the stale case is safe, as a fact about the data rather than a claim
        about a hypothetical refactor.

        The keys are the literal glyphs, so a token still holding a fancy glyph
        matches on the way out no matter what the token was decided as. An earlier
        version of this test asserted that keying the table on `theme.OK` instead
        would break it — which is indistinguishable from the real thing on a UTF-8
        system, since there `OK` *is* the glyph. It passed against a mutation that
        changed nothing, which is worse than no test at all.
        """
        keys = {fancy for fancy, _ in theme._FALLBACKS}
        for glyph in ("\u2713", "\u2717", "\u25cc", "\u2192"):
            assert glyph in keys, f"{glyph!r} is used by the module but not in the table"

        # Every key must be a non-ASCII glyph, since a key the terminal could encode
        # would never be reached.
        assert all(not k.isascii() for k in keys), keys
