"""
protor.crawler
~~~~~~~~~~~~~~
Async recursive site crawler with SQLite-backed queue,
checkpoint/resume, and auto-scaling concurrency. The crawl loop itself lives in
:mod:`protor.engine`; this module supplies the persistent queue, the live render,
and the crawl state observer.

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

from rich import box
from rich.console import Group
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .config import (
    CHECKPOINT_FILENAME,
    CRAWLER_CONCURRENCY,
    CRAWLER_DELAY,
    DEFAULT_MAX_PAGES,
)
from .engine import CrawlEngine, RecursiveSource
from .rate_limiter import DomainRateLimiter
from .scaler import AutoScaler
from .theme import ERR, OK, SKIP, SPIN, bright, console, header_rule, label, muted
from .utils import canonicalize_url, get_default_output_dir, save_json

__all__ = ["Crawler"]


# ── SQLite-backed crawl queue ────────────────────────────────────────────────
# Inspired by Crawlee's persistent request queue


class _CrawlQueue:
    """
    Persistent SQLite-backed URL queue with deduplication.

    Supports BFS ordering, visited tracking, and checkpoint serialization.

    Writes are deferred: mutating calls mark the connection dirty and the
    transaction is committed in batches (and on close). The previous version
    committed per operation, fsyncing three times per page on the event loop —
    ~765 us per page of pure blocking I/O, all of it serialized against fetches.
    In-memory counters answer the ``empty``/``queue_size`` questions that used
    to run ``COUNT(*)`` on every admission check.
    """

    #: Mutations between automatic commits.
    COMMIT_EVERY = 64

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._conn = conn
        self._closed = False
        self._dirty = 0
        self._queued = 0
        self._visited = 0
        self._init_db()

    def _commit(self, force: bool = False) -> None:
        """Commit once enough mutations have accumulated (or when forced)."""
        if self._closed:
            return
        self._dirty += 1
        if force or self._dirty >= self.COMMIT_EVERY:
            self._conn.commit()
            self._dirty = 0

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                priority INTEGER DEFAULT 0,
                added_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS visited (
                url TEXT UNIQUE NOT NULL,
                scraped_at REAL,
                success INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_queue_priority
            ON queue(priority DESC, added_at ASC);
        """)
        self._conn.commit()
        self._queued = self._count("queue")
        self._visited = self._count("visited")

    def _count(self, table: str) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0

    def enqueue(self, url: str, priority: int = 0) -> bool:
        """Add URL to queue if not already queued or visited. Returns True if added."""
        url = canonicalize_url(url)
        if self.is_visited(url) or self.is_queued(url):
            return False
        self._conn.execute(
            "INSERT INTO queue (url, priority, added_at) VALUES (?, ?, ?)",
            (url, priority, time.time()),
        )
        self._queued += 1
        self._commit()
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
        self._queued = max(0, self._queued - 1)
        self._commit()
        return url

    def is_visited(self, url: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM visited WHERE url = ?", (canonicalize_url(url),)
        ).fetchone()
        return row is not None

    def is_queued(self, url: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM queue WHERE url = ?", (canonicalize_url(url),)
        ).fetchone()
        return row is not None

    def mark_visited(self, url: str, success: bool = True) -> None:
        url = canonicalize_url(url)
        self._conn.execute(
            "INSERT OR REPLACE INTO visited (url, scraped_at, success) VALUES (?, ?, ?)",
            (url, time.time(), int(success)),
        )
        self._visited += 1
        self._commit()

    @property
    def empty(self) -> bool:
        return self._queued == 0

    @property
    def queue_size(self) -> int:
        return self._queued

    @property
    def visited_count(self) -> int:
        return self._visited

    @property
    def success_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM visited WHERE success = 1").fetchone()
        return row[0] if row else 0

    def to_checkpoint(self) -> dict:
        """Serialize queue state for checkpoint/resume."""
        self._commit(force=True)
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
        q._commit(force=True)
        return q

    def close(self) -> None:
        """Flush and close. Idempotent, so repeated teardown is safe."""
        if self._closed:
            return
        self._commit(force=True)
        self._closed = True
        self._conn.close()


# ── crawl state ──────────────────────────────────────────────────────────────


@dataclass
class _CrawlLog:
    status: str  # "ok" | "err" | "active" | "blocked" | "skip"
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


_BAR_WIDTH = 32


def _render(state: _State, output_dir: str) -> Group:
    # The bar is scaled to a fixed width. One cell per page made --max-pages
    # 500 render a 500-character bar that wrapped and wrecked the layout.
    if state.max_pages:
        filled = round(_BAR_WIDTH * min(state.scraped / state.max_pages, 1.0))
        pct = int(state.scraped / state.max_pages * 100)
    else:
        filled = 0
        pct = 0
    filled = max(0, min(filled, _BAR_WIDTH))
    bar = "█" * filled + "░" * (_BAR_WIDTH - filled)

    stat = Table(box=box.SIMPLE, show_header=False, show_edge=False, padding=(0, 1))
    stat.add_column(width=10, style="grey74")
    stat.add_column(style="white")
    stat.add_row(
        "progress",
        f"[grey50]{bar}[/grey50]  [white]{pct}%[/white]  [grey50]{state.scraped}/{state.max_pages}[/grey50]",
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
        elif entry.status == "skip":
            s = Text(f"{SKIP} skipped", style="grey50")
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
        self._log_index: dict[str, int] = {}

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
        scaler: AutoScaler | None = None
        if self.auto_scale:
            scaler = AutoScaler(initial=CRAWLER_CONCURRENCY)

        engine = CrawlEngine(
            queue=self._queue,
            link_source=RecursiveSource(),
            output_dir=self.output_dir,
            max_targets=self.max_pages,
            concurrency=CRAWLER_CONCURRENCY,
            auto_scaler=scaler,
            allowed_domain=self._base_domain,
            check_robots=True,
            rate_limiter=DomainRateLimiter(delay=CRAWLER_DELAY),
            checkpoint_interval=5,
            on_checkpoint=self._save_checkpoint,
            on_status=self._on_status,
            live_render=lambda: _render(self._state, str(self.output_dir)),
        )
        await engine.arun()

    def _on_status(self, status: str, url: str, row: dict) -> None:
        """Keep crawl state in sync with engine events for the live render."""
        domain = urlparse(url).netloc
        self._state.queue_n = self._queue.queue_size

        if status == "fetching":
            self._state.current = url
            self._log_index[url] = len(self._state.log)
            self._state.log.append(_CrawlLog("active", domain))
        elif status.startswith("js:"):
            # Status is "js:N"; comparing against the bare "js:" never matched.
            self._update_log(url, "active", domain)
        elif status == "done":
            self._state.scraped += 1
            self._update_log(url, "ok", domain)
        elif status == "error":
            self._state.errors += 1
            self._update_log(url, "err", domain, str(row.get("note", "")))
        elif status == "blocked":
            self._state.blocked += 1
            self._state.log.append(_CrawlLog("blocked", domain, note=str(row.get("note", ""))))
        elif status == "skipped":
            self._update_log(url, "skip", domain, str(row.get("note", "")))

    def _update_log(self, url: str, status: str, domain: str, note: str = "") -> None:
        idx = self._log_index.get(url)
        if idx is not None and idx < len(self._state.log):
            self._state.log[idx].status = status
            self._state.log[idx].note = note
        else:
            self._state.log.append(_CrawlLog(status, domain, note))
