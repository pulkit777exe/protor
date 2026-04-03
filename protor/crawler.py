"""
protor.crawler
~~~~~~~~~~~~~~
Async recursive site crawler.

Public API
----------
    Crawler(start_url, max_pages, output_dir).crawl()
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from rich import box
from rich.console import Group
from rich.live import Live
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .config import CRAWLER_CONCURRENCY, CRAWLER_DELAY, DEFAULT_MAX_PAGES, HEADERS
from .scraper import _fetch, extract_links, scrape_site_async
from .theme import ERR, OK, SPIN, bright, console, header_rule, label, muted
from .utils import get_default_output_dir

__all__ = ["Crawler"]


@dataclass
class _CrawlLog:
    status: str   # "ok" | "err" | "active"
    domain: str
    note: str = ""


@dataclass
class _State:
    scraped:  int = 0
    errors:   int = 0
    current:  str = ""
    queue_n:  int = 0
    max_pages: int = DEFAULT_MAX_PAGES
    log: list[_CrawlLog] = field(default_factory=list)


def _render(state: _State, output_dir: str) -> Group:
    # progress bar
    filled = "█" * state.scraped
    empty  = "░" * (state.max_pages - state.scraped)
    pct    = int(state.scraped / state.max_pages * 100) if state.max_pages else 0

    stat = Table(box=box.SIMPLE, show_header=False, show_edge=False, padding=(0, 1))
    stat.add_column(width=10, style="grey74")
    stat.add_column(style="white")
    stat.add_row("progress", f"[grey50]{filled}[/grey50][grey23]{empty}[/grey23]  [white]{pct}%[/white]  [grey50]{state.scraped}/{state.max_pages}[/grey50]")
    stat.add_row("current",  muted(state.current[:72]) if state.current else "[grey23]—[/grey23]")
    stat.add_row("queue",    bright(str(state.queue_n)))
    stat.add_row("errors",   str(state.errors) if state.errors else "[grey23]0[/grey23]")
    stat.add_row("output",   muted(output_dir))

    log_t = Table(box=box.SIMPLE, show_header=True, header_style="bold white",
                  show_edge=False, padding=(0, 1))
    log_t.add_column("#",      style="grey50", width=4, justify="right")
    log_t.add_column("Domain", style="white",  min_width=28)
    log_t.add_column("Status", width=10)

    recent = state.log[-20:]
    for i, entry in enumerate(recent, max(1, len(state.log) - 19)):
        if entry.status == "ok":
            s = Text(f"{OK} done",  style="green")
        elif entry.status == "err":
            s = Text(f"{ERR} error", style="red")
        else:
            s = Text(f"{SPIN} ...",  style="yellow")
        log_t.add_row(str(i), entry.domain, s)

    return Group(Rule(style="grey23"), stat, Rule(style="grey23"), log_t)


class Crawler:
    """
    Async BFS crawler for a single domain.

    Parameters
    ----------
    start_url:
        Seed URL; only pages on the same domain are followed.
    max_pages:
        Hard limit on pages scraped.
    output_dir:
        Root directory for scraped artefacts.
    """

    def __init__(
        self,
        start_url: str,
        max_pages: int = DEFAULT_MAX_PAGES,
        output_dir: str | Path | None = None,
    ) -> None:
        self.start_url  = start_url
        self.max_pages  = max_pages
        self.output_dir = Path(output_dir or get_default_output_dir())

        self._visited: set[str]     = set()
        self._queue:   deque[str]   = deque([start_url])
        self._state    = _State(max_pages=max_pages)

    # ── public ────────────────────────────────────────────────────────────────

    def crawl(self) -> None:
        """Run the crawl (blocking)."""
        console.print()
        console.print(header_rule("Protor — Crawler"))
        console.print(
            f"  {label('start')} {bright(self.start_url)}\n"
            f"  {label('limit')} {bright(str(self.max_pages))} pages"
        )
        console.print()
        asyncio.run(self._run())
        console.print()
        console.print(
            f"  {OK} crawl complete — "
            f"{bright(str(self._state.scraped))} pages scraped"
            + (f", {self._state.errors} errors" if self._state.errors else "")
        )
        console.print(f"  {label('output')} {muted(str(self.output_dir))}")
        console.print()

    # ── internal ──────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        connector = aiohttp.TCPConnector(limit=CRAWLER_CONCURRENCY)
        async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
            with Live(console=console, refresh_per_second=6) as live:
                while self._queue and self._state.scraped < self.max_pages:
                    url = self._queue.popleft()
                    if url in self._visited:
                        continue

                    self._visited.add(url)
                    domain = urlparse(url).netloc

                    self._state.current = url
                    self._state.queue_n = len(self._queue)
                    self._state.log.append(_CrawlLog("active", domain))
                    live.update(_render(self._state, str(self.output_dir)))

                    try:
                        html, _ = await _fetch(session, url)
                        for link in extract_links(html, url):
                            if link not in self._visited and link not in self._queue:
                                self._queue.append(link)
                        row: dict = {}
                        await scrape_site_async(session, url, self.output_dir, False, row)
                        self._state.scraped += 1
                        self._state.log[-1].status = "ok"
                    except Exception as exc:
                        self._state.errors += 1
                        self._state.log[-1].status = "err"
                        self._state.log[-1].note = str(exc)

                    self._state.queue_n = len(self._queue)
                    live.update(_render(self._state, str(self.output_dir)))
                    await asyncio.sleep(CRAWLER_DELAY)

                live.update(_render(self._state, str(self.output_dir)))
