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
from rich.markup import escape
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


def _output_encoding() -> str:
    return (sys.stdout.encoding or "utf-8").lower()


def _can_encode(text: str) -> bool:
    """
    True when *text* survives the trip to *text*'s terminal and back.

    Testing only that ``encode`` does not raise is not enough, and cp1252 is why.
    It *can* encode an em dash: U+2014 maps to byte 0x97, which is in the range
    that decodes back to the C1 control characters. So the encode succeeds, the
    check passed, and the terminal drew a control character where a dash should be —
    an em dash reaching a Windows console as something that looks like corruption.
    Requiring the round trip to be lossless rejects those, and costs one decode.
    """
    encoding = _output_encoding()
    try:
        raw = text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    try:
        return raw.decode(encoding) == text
    except UnicodeDecodeError:  # pragma: no cover - a codec that will not round-trip
        return False


def content(text: str) -> str:
    """
    Make *text* safe to interpolate into a rich **markup** string.

    :func:`safe` handles the encoding; this handles rich's markup, which is a
    separate parser with a separate opinion about square brackets. Rich read
    ``[slug]`` in a URL as a style tag and silently dropped it, so the crawl
    view showed ``https://ex.com/docs/`` for a page actually at
    ``https://ex.com/docs/[slug]`` — no crash, just a wrong answer about which
    page is being fetched. ``[..]`` happened to survive only because a dot is
    not a legal tag name.

    Every helper below takes *content*, never markup, which is what makes this
    the right thing to do in all of them: there is no call site that means to
    pass a style tag through one. The console's own ``print`` is the opposite
    case and deliberately does *not* escape, because most output reaches it as an
    f-string of helpers that have already emitted their own tags.
    """
    return escape(safe(text))


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

#: The same glyphs, already coloured. A summary line mixes a glyph with markup the
#: surrounding helpers produced — `bright(count)`, `muted(dir)` — so it cannot use
#: `ok()`/`err()`, which escape their argument and would eat those tags. Warnings
#: were yellow and errors red while every headline success and failure sat in the
#: default style, so `✓ 40 scraped` and `✗ 2 failed` were the same colour.
OK_STYLED = f"[green]{OK}[/green]"
ERR_STYLED = f"[red]{ERR}[/red]"

#: The `warn()` glyph, coloured, for a headline that is neither success nor
#: failure. Used when a run is cut short: the work done so far is real and worth
#: reporting, but it is not a completed crawl.
WARN_STYLED = "[yellow]![/yellow]"
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


class SafeTable(Table):
    """
    A :class:`rich.table.Table` whose cells go through :func:`safe`.

    A Table renders its cells without ever passing them through the console's
    ``print``, so the helpers never saw them and only ``_EncodingSafeFile`` stood
    between a scraped string and the terminal. That file cannot degrade anything —
    it has already lost the string by then — so a cell holding an em dash rendered as
    the cp1252 replacement byte while the same glyph written through ``muted()``
    rendered as ``-``. Same glyph, same terminal, two characters.

    ``safe`` and not ``content``: these cells carry markup on purpose (``muted(...)``
    and friends have already emitted their tags), and escaping would eat it.
    """

    def add_row(self, *renderables: Any, **kwargs: Any) -> None:
        super().add_row(*[_degrade(r) for r in renderables], **kwargs)  # type: ignore[arg-type]


console = ProtorConsole(highlight=False, soft_wrap=True)

#: Where diagnostics go.
#:
#: stdout carries the report of what a command did; stderr carries what the user
#: has to act on that is not part of that report. Every line went to stdout before,
#: so `protor scrape url > report.txt` interleaved "✗ HTTP 403" into the report and
#: `2>/dev/null` could not silence a failure — neither of which is what a shell
#: redirection is for. `rg`, `cargo`, `docker` and `kubectl` all split it this way.
#:
#: Two consoles rather than a flag per call, so the destination is a property of
#: the *kind* of line and cannot be forgotten at a call site.
err_console = ProtorConsole(stderr=True, highlight=False, soft_wrap=True)


# ── rules ─────────────────────────────────────────────────────────────────────
def header_rule(title: str) -> Rule:
    return Rule(f"[bold white]{content(title)}[/bold white]", style="grey35")


def section_rule(title: str) -> Rule:
    return Rule(f"[grey50]{content(title)}[/grey50]", style="grey23")


# ── inline text helpers ───────────────────────────────────────────────────────
def dim(s: str) -> str:
    return f"[grey23]{content(s)}[/grey23]"


def muted(s: str) -> str:
    return f"[grey50]{content(s)}[/grey50]"


def label(s: str) -> str:
    return f"[grey74]{content(s)}[/grey74]"


def bright(s: str) -> str:
    return f"[bold white]{content(s)}[/bold white]"


def ok(s: str) -> str:
    return f"[green]{OK} {content(s)}[/green]"


def err(s: str) -> str:
    return f"[red]{ERR} {content(s)}[/red]"


def warn(s: str) -> str:
    return f"[yellow]! {content(s)}[/yellow]"


def info(s: str) -> str:
    return f"[grey74]{ARROW} {content(s)}[/grey74]"
