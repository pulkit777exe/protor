"""
protor.crawler
~~~~~~~~~~~~~~
Async recursive site crawler with SQLite-backed queue,
checkpoint/resume, and auto-scaling concurrency.

Inspired by:
    - Crawl4AI: crash recovery with resume_state
    - Crawlee: persistent request queue
    - Scrapy: spider pattern with callbacks
    - Scrapling: pause/resume support

Public API
----------
    Crawler(start_url, max_pages, output_dir).crawl()
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
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

from .config import (
    CHECKPOINT_FILENAME,
    CRAWLER_CONCURRENCY,
    CRAWLER_DELAY,
    DEFAULT_MAX_PAGES,
    HEADERS,
)
from .robots import check_robots
from .scaler import AutoScaler
from .scraper import _fetch, extract_links, scrape_site_async
from .theme import ERR, OK, SPIN, bright, console, header_rule, label, muted
from .utils import get_default_output_dir, save_json

__all__ = ["Crawler"]


# ── SQLite-backed crawl queue ────────────────────────────────────────────────
# Inspired by Crawlee's persistent request queue


class _CrawlQueue:
    """
    Persistent SQLite-backed URL queue with deduplication.

    Supports BFS ordering, visited tracking, and checkpoint serialization.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                priority INTEGER DEFAULT 0,
                added_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS visited (
                url TEXT UNIQUE NOT NULL,
                scraped_at REAL,
                success INTEGER DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_queue_priority
            ON queue(priority DESC, added_at ASC)
        """)
        self._conn.commit()

    def enqueue(self, url: str, priority: int = 0) -> bool:
        """Add URL to queue if not already queued or visited. Returns True if added."""
        if self.is_visited(url) or self.is_queued(url):
            return False
        self._conn.execute(
            "INSERT INTO queue (url, priority, added_at) VALUES (?, ?, ?)",
            (url, priority, time.time()),
        )
        self._conn.commit()
        return True

    def dequeue(self) -> str | None:
        """Pop the highest-priority, oldest URL from the queue."""
        row = self._conn.execute(
            "SELECT url FROM queue ORDER BY priority DESC, added_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        url: str = row[0]
        self._conn.execute("DELETE FROM queue WHERE url = ?", (url,))
        self._conn.commit()
        return url

    def is_visited(self, url: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM visited WHERE url = ?", (url,)).fetchone()
        return row is not None

    def is_queued(self, url: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM queue WHERE url = ?", (url,)).fetchone()
        return row is not None

    def mark_visited(self, url: str, success: bool = True) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO visited (url, scraped_at, success) VALUES (?, ?, ?)",
            (url, time.time(), int(success)),
        )
        self._conn.commit()

    @property
    def queue_size(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM queue").fetchone()
        return row[0] if row else 0

    @property
    def visited_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM visited").fetchone()
        return row[0] if row else 0

    @property
    def success_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM visited WHERE success = 1").fetchone()
        return row[0] if row else 0

    def to_checkpoint(self) -> dict:
        """Serialize queue state for checkpoint/resume."""
        queued = [
            row[0]
            for row in self._conn.execute("SELECT url FROM queue ORDER BY added_at ASC").fetchall()
        ]
        visited = [
            row[0]
            for row in self._conn.execute("SELECT url FROM visited WHERE success = 1").fetchall()
        ]
        return {"queued": queued, "visited": visited}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, db_path: Path) -> _CrawlQueue:
        """Restore queue from a checkpoint."""
        q = cls(db_path)
        for url in checkpoint.get("visited", []):
            q.mark_visited(url, success=True)
        for url in checkpoint.get("queued", []):
            q.enqueue(url)
        return q

    def close(self) -> None:
        self._conn.close()


# ── crawl state ──────────────────────────────────────────────────────────────


@dataclass
class _CrawlLog:
    status: str  # "ok" | "err" | "active" | "blocked"
    domain: str
    note: str = ""


@dataclass
class _State:
    scraped: int = 0
    errors: int = 0
    blocked: int = 0
    current: str = ""
    queue_n: int = 0
    max_pages: int = DEFAULT_MAX_PAGES
    log: list[_CrawlLog] = field(default_factory=list)


def _render(state: _State, output_dir: str) -> Group:
    filled = "█" * state.scraped
    empty = "░" * (state.max_pages - state.scraped)
    pct = int(state.scraped / state.max_pages * 100) if state.max_pages else 0

    stat = Table(box=box.SIMPLE, show_header=False, show_edge=False, padding=(0, 1))
    stat.add_column(width=10, style="grey74")
    stat.add_column(style="white")
    stat.add_row(
        "progress",
        f"[grey50]{filled}[/grey50][grey23]{empty}[/grey23]  [white]{pct}%[/white]  [grey50]{state.scraped}/{state.max_pages}[/grey50]",
    )
    stat.add_row("current", muted(state.current[:72]) if state.current else "[grey23]—[/grey23]")
    stat.add_row("queue", bright(str(state.queue_n)))
    stat.add_row("errors", str(state.errors) if state.errors else "[grey23]0[/grey23]")
    stat.add_row("blocked", str(state.blocked) if state.blocked else "[grey23]0[/grey23]")
    stat.add_row("output", muted(output_dir))

    log_t = Table(
        box=box.SIMPLE, show_header=True, header_style="bold white", show_edge=False, padding=(0, 1)
    )
    log_t.add_column("#", style="grey50", width=4, justify="right")
    log_t.add_column("Domain", style="white", min_width=28)
    log_t.add_column("Status", width=10)

    recent = state.log[-20:]
    for i, entry in enumerate(recent, max(1, len(state.log) - 19)):
        if entry.status == "ok":
            s = Text(f"{OK} done", style="green")
        elif entry.status == "err":
            s = Text(f"{ERR} error", style="red")
        elif entry.status == "blocked":
            s = Text(f"{ERR} blocked", style="red")
        else:
            s = Text(f"{SPIN} ...", style="yellow")
        log_t.add_row(str(i), entry.domain, s)

    return Group(Rule(style="grey23"), stat, Rule(style="grey23"), log_t)


class Crawler:
    """
    Async BFS crawler for a single domain with checkpoint/resume.

    Parameters
    ----------
    start_url:
        Seed URL; only pages on the same domain are followed.
    max_pages:
        Hard limit on pages scraped.
    output_dir:
        Root directory for scraped artefacts.
    resume:
        If True, resume from a previous checkpoint if available.
    auto_scale:
        If True, automatically adjust concurrency based on success rates.
    """

    def __init__(
        self,
        start_url: str,
        max_pages: int = DEFAULT_MAX_PAGES,
        output_dir: str | Path | None = None,
        resume: bool = False,
        auto_scale: bool = False,
    ) -> None:
        self.start_url = start_url
        self.max_pages = max_pages
        self.output_dir = Path(output_dir or get_default_output_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.auto_scale = auto_scale

        self._base_domain = urlparse(start_url).netloc
        self._state = _State(max_pages=max_pages)

        # SQLite queue
        db_path = self.output_dir / "crawl_queue.db"
        self._queue = _CrawlQueue(db_path)

        # Load checkpoint if resuming
        checkpoint_path = self.output_dir / CHECKPOINT_FILENAME
        if resume and checkpoint_path.exists():
            try:
                cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                self._queue = _CrawlQueue.from_checkpoint(cp, db_path)
                self._state.scraped = self._queue.success_count
                console.print(
                    f"  {OK} Resumed from checkpoint — {self._state.scraped} pages already scraped"
                )
            except Exception:
                pass

        # Always ensure start_url is queued
        self._queue.enqueue(start_url)

    # ── public ────────────────────────────────────────────────────────────────

    def crawl(self) -> None:
        """Run the crawl (blocking)."""
        console.print()
        console.print(header_rule("Protor — Crawler"))
        console.print(
            f"  {label('start')} {bright(self.start_url)}\n"
            f"  {label('limit')} {bright(str(self.max_pages))} pages"
            + (f"\n  {label('resume')} {bright('enabled')}" if self.resume else "")
            + (f"\n  {label('auto-scale')} {bright('enabled')}" if self.auto_scale else "")
        )
        console.print()
        try:
            asyncio.run(self._run())
        finally:
            self._save_checkpoint()
            self._queue.close()
        console.print()
        console.print(
            f"  {OK} crawl complete — "
            f"{bright(str(self._state.scraped))} pages scraped"
            + (f", {self._state.errors} errors" if self._state.errors else "")
            + (f", {self._state.blocked} blocked" if self._state.blocked else "")
        )
        console.print(f"  {label('output')} {muted(str(self.output_dir))}")
        console.print()

    def _save_checkpoint(self) -> None:
        """Save crawl state to checkpoint file."""
        cp = self._queue.to_checkpoint()
        cp["start_url"] = self.start_url
        cp["max_pages"] = self.max_pages
        cp["scraped"] = self._state.scraped
        cp["timestamp"] = time.time()
        checkpoint_path = self.output_dir / CHECKPOINT_FILENAME
        save_json(cp, checkpoint_path)

    # ── internal ──────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        connector = aiohttp.TCPConnector(limit=CRAWLER_CONCURRENCY)
        async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
            scaler: AutoScaler | None = None
            if self.auto_scale:
                scaler = AutoScaler(initial=CRAWLER_CONCURRENCY)

            with Live(console=console, refresh_per_second=6) as live:
                while self._state.scraped < self.max_pages:
                    url = self._queue.dequeue()
                    if url is None:
                        break

                    domain = urlparse(url).netloc

                    # Only follow same-domain links
                    if urlparse(url).netloc != self._base_domain:
                        self._queue.mark_visited(url, success=False)
                        continue

                    self._state.current = url
                    self._state.queue_n = self._queue.queue_size
                    self._state.log.append(_CrawlLog("active", domain))
                    live.update(_render(self._state, str(self.output_dir)))

                    try:
                        if not await check_robots(url, session):
                            self._state.log[-1].status = "blocked"
                            self._state.log[-1].note = "blocked by robots.txt"
                            self._state.blocked += 1
                            self._queue.mark_visited(url, success=False)
                            if scaler:
                                scaler.record(False)
                            continue

                        html, _ = await _fetch(session, url)

                        # Extract and enqueue new links
                        new_links = 0
                        for link in extract_links(html, url):
                            if self._queue.enqueue(link):
                                new_links += 1

                        row: dict = {}
                        result = await scrape_site_async(
                            session,
                            url,
                            self.output_dir,
                            False,
                            row,
                        )
                        if result:
                            self._state.scraped += 1
                            self._state.log[-1].status = "ok"
                            self._queue.mark_visited(url, success=True)
                        else:
                            self._state.errors += 1
                            self._state.log[-1].status = "err"
                            self._queue.mark_visited(url, success=False)

                        if scaler:
                            scaler.record(result is not None)

                    except Exception as exc:
                        self._state.errors += 1
                        self._state.log[-1].status = "err"
                        self._state.log[-1].note = str(exc)
                        self._queue.mark_visited(url, success=False)
                        if scaler:
                            scaler.record(False)

                    self._state.queue_n = self._queue.queue_size
                    live.update(_render(self._state, str(self.output_dir)))

                    # Auto-scaling
                    if scaler:
                        scaler.maybe_scale()

                    # Checkpoint periodically
                    if self._state.scraped % 5 == 0:
                        self._save_checkpoint()

                    await asyncio.sleep(CRAWLER_DELAY)

                live.update(_render(self._state, str(self.output_dir)))
