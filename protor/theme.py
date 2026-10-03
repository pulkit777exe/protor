"""
Terminal design tokens — Claude Code aesthetic.
All UI primitives live here so the rest of the codebase stays logic-only.

The glyph tokens are chosen against the terminal's actual encoding. A Windows
cp1252 console cannot encode ``✓``/``✗``/``◌``/``→`` and an ASCII pipe cannot
encode anything, so a fixed set of pretty glyphs meant ``protor models`` died
with a UnicodeEncodeError traceback on those terminals. Degrading the token once,
here, fixes every call site at once.
"""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.rule import Rule
from rich.text import Text


def _output_encoding() -> str:
    return (sys.stdout.encoding or "utf-8").lower()


def _can_encode(text: str) -> bool:
    try:
        text.encode(_output_encoding())
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def safe(text: str) -> str:
    """
    Return *text* with any glyph the terminal cannot encode replaced.

    Applied to every string the CLI prints, so a legacy terminal or a CI log
    gets readable ASCII instead of a UnicodeEncodeError traceback.

    Total by construction: the substitution table holds the glyphs *this* module
    uses, and scraped page text or model output can hold anything at all. A
    character the table does not name — ``♠``, or an ``é`` on an ASCII terminal —
    used to come straight back and raise at the write, which is the crash this
    function exists to prevent. Whatever survives the table is therefore forced
    through the encoding, so the return value is always printable.
    """
    if not text or _can_encode(text):
        return text
    out = text
    for fancy, plain in _FALLBACKS:
        out = out.replace(fancy, plain)
    encoding = _output_encoding()
    try:
        return out.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except LookupError:
        return out


#: (fancy, plain) pairs, applied in order until the text is encodable.
_FALLBACKS: tuple[tuple[str, str], ...] = (
    ("✓", "+"),
    ("✗", "x"),
    ("◌", "o"),
    ("→", "->"),
    ("█", "#"),
    ("░", "."),
    ("⚠", "!"),
    ("·", "-"),
    ("—", "-"),
    ("…", "..."),
)

# ── glyphs (degraded when the terminal cannot encode them) ────────────────────
OK = "✓" if _can_encode("✓") else "+"
ERR = "✗" if _can_encode("✗") else "x"
SKIP = "-"
SPIN = "◌" if _can_encode("◌") else "o"
ARROW = "→" if _can_encode("→") else "->"

# ── console (shared instance; importable) ────────────────────────────────────


class _EncodingSafeFile:
    """
    A text stream that writes what the terminal can encode instead of raising.

    ``rich.console.Console`` takes no ``errors`` parameter and writes straight to
    the stream, so there was nowhere to say "replace what will not fit". The
    helpers above sanitise their own arguments, but a ``Table`` renders its cells
    without ever passing them through ``print`` — a model name or a page title in
    a cell reached the terminal unsanitised and raised from the middle of the
    render, taking the error report that followed down with it.

    Encoding before writing means the replacement happens with the whole string
    in hand: nothing is written half-way, and nothing raises.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        encoding = getattr(self._stream, "encoding", None) or "utf-8"
        try:
            text.encode(encoding)
        except (UnicodeEncodeError, LookupError):
            text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        written: int = self._stream.write(text)
        return written

    def __getattr__(self, name: str) -> Any:
        # isatty, fileno, encoding and the rest must reach the real stream: the
        # console decides whether it may colourise or animate from them.
        return getattr(self._stream, name)


class ProtorConsole(Console):
    """
    Console that degrades glyphs the terminal cannot encode.

    The helpers in this module sanitise their own arguments, but plenty of output
    is an f-string passed straight to ``print`` — ``f"  {OK} crawl complete — "`` —
    and a raw em dash in one of those crashed the command with a
    UnicodeEncodeError. Sanitising at the console is the one chokepoint that
    cannot be forgotten; the helpers stay because a Table renders its cells
    without ever passing them through ``print``, which the write path below
    covers rather than the arguments.
    """

    @property
    def file(self) -> Any:
        """
        The underlying stream, wrapped so a write cannot fail on encoding.

        Resolved per access rather than cached, because ``Console`` picks up
        ``sys.stdout`` afresh every time and a test that swaps it mid-run has to
        be writing to the replacement.
        """
        return _EncodingSafeFile(super().file)

    @file.setter
    def file(self, value: Any) -> None:
        self._file = value

    def print(self, *objects: object, **kwargs: object) -> None:
        super().print(*(_degrade(obj) for obj in objects), **kwargs)  # type: ignore[arg-type]


def _degrade(obj: object) -> object:
    """Return *obj* with unencodable characters replaced, preserving its type."""
    if isinstance(obj, str):
        return safe(obj)
    if isinstance(obj, Text):
        plain = safe(obj.plain)
        if plain == obj.plain:
            return obj
        return Text(plain, style=obj.style, justify=obj.justify, end=obj.end)
    return obj


console = ProtorConsole(highlight=False, soft_wrap=True)


# ── rules ─────────────────────────────────────────────────────────────────────
def header_rule(title: str) -> Rule:
    return Rule(f"[bold white]{safe(title)}[/bold white]", style="grey35")


def section_rule(title: str) -> Rule:
    return Rule(f"[grey50]{safe(title)}[/grey50]", style="grey23")


# ── inline text helpers ───────────────────────────────────────────────────────
def dim(s: str) -> str:
    return f"[grey23]{safe(s)}[/grey23]"


def muted(s: str) -> str:
    return f"[grey50]{safe(s)}[/grey50]"


def label(s: str) -> str:
    return f"[grey74]{safe(s)}[/grey74]"


def bright(s: str) -> str:
    return f"[bold white]{safe(s)}[/bold white]"


def ok(s: str) -> str:
    return f"[green]{OK} {safe(s)}[/green]"


def err(s: str) -> str:
    return f"[red]{ERR} {safe(s)}[/red]"


def warn(s: str) -> str:
    return f"[yellow]! {safe(s)}[/yellow]"


def info(s: str) -> str:
    return f"[grey74]{ARROW} {safe(s)}[/grey74]"
