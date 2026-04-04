"""
Terminal design tokens — Claude Code aesthetic.
All UI primitives live here so the rest of the codebase stays logic-only.
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# ── glyphs ────────────────────────────────────────────────────────────────────
OK = "✓"
ERR = "✗"
SKIP = "-"
SPIN = "◌"
ARROW = "→"

# ── console (shared instance; importable) ────────────────────────────────────
console = Console(highlight=False, soft_wrap=True)


# ── rules ─────────────────────────────────────────────────────────────────────
def header_rule(title: str) -> Rule:
    return Rule(f"[bold white]{title}[/bold white]", style="grey35")


def section_rule(title: str) -> Rule:
    return Rule(f"[grey50]{title}[/grey50]", style="grey23")


# ── inline text helpers ───────────────────────────────────────────────────────
def dim(s: str) -> str:
    return f"[grey23]{s}[/grey23]"


def muted(s: str) -> str:
    return f"[grey50]{s}[/grey50]"


def label(s: str) -> str:
    return f"[grey74]{s}[/grey74]"


def bright(s: str) -> str:
    return f"[bold white]{s}[/bold white]"


def ok(s: str) -> str:
    return f"[green]{OK} {s}[/green]"


def err(s: str) -> str:
    return f"[red]{ERR} {s}[/red]"


def warn(s: str) -> str:
    return f"[yellow]! {s}[/yellow]"


def info(s: str) -> str:
    return f"[grey74]{ARROW} {s}[/grey74]"


# ── panels ────────────────────────────────────────────────────────────────────
def simple_panel(body: str, title: str = "") -> Panel:
    kw: dict = dict(box=box.ROUNDED, border_style="grey35", padding=(0, 2))
    if title:
        kw["title"] = f"[bold white]{title}[/bold white]"
        kw["title_align"] = "left"
    return Panel(body, **kw)
