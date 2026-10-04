"""
protor.crawler
~~~~~~~~~~~~~~
Async recursive site crawler with SQLite-backed queue,
checkpoint/resume, and auto-scaling concurrency. The crawl loop itself lives in
:mod:`protor.engine`; this module supplies the persistent queue, the live render,
and the crawl state observer.

The crawl state has exactly one home: ``crawl_queue.db``. It is opened on every
run and committed in batches as pages are discovered and scraped, so it survives
an interrupted run by itself. The JSON checkpoint next to it is a *summary* of
the run (start URL, budget, pages scraped) and is never used to reconstruct
queue state. An earlier version mirrored the whole queue into both files and
then replayed the JSON through a second connection on the same database: every
URL was written twice and the two stores could disagree.

Because that database is always open, its rows decide what a new crawl does, so
the flag has to. Without ``resume`` a crawl clears the queue and the visited
rows first — a second plain run then actually crawls instead of silently
reporting success having requested nothing. Clearing the rows rather than the
file leaves the saved pages, the manifests and the WAL sidecars alone.
``--resume`` keeps them, and prices the page budget from the rows already there.

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
import contextlib
import json
import sqlite3
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
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
from .http_cache import HTTPCache
from .progress import normalise_reason, print_failure_reasons
from .rate_limiter import DomainRateLimiter
from .scaler import AutoScaler
from .theme import (
    ERR,
    OK,
    OK_STYLED,
    SKIP,
    SPIN,
    bright,
    console,
    content,
    header_rule,
    info,
    label,
    muted,
    safe,
    warn,
)
from .utils import canonicalize_url, get_default_output_dir, save_json

__all__ = ["Crawler"]


# ── SQLite-backed crawl queue ────────────────────────────────────────────────
# Inspired by Crawlee's persistent request queue

#: "Have we already seen this URL?", answered in one round trip. ``enqueue`` is
#: on the hot path (once per discovered link) and the two tables it consulted
#: made no difference to the caller: either one means "reject".
#: Refuse a URL that is already queued, already fetched successfully, or already
#: attempted during *this* run.
#:
#: A URL whose earlier attempt failed in a **previous** run is deliberately not
#: "seen": retrying a page that timed out or returned 503 is the whole point of
#: running the crawl again, and treating every visit as final made the second
#: run find an empty queue and report "0 pages queued" without saying why — the
#: retry looked like it had nothing to do rather than being unable to do
#: anything.
#:
#: The current-run cutoff is what keeps that from becoming a loop. Without it, a
#: page that links to itself and keeps failing — a 404 in a nav footer, say —
#: would be re-queued by every page that links to it, on every pass. The cutoff
#: makes the retry happen on the *next* run, once: a failure recorded during this
#: run counts as attempted and is refused, which is what stops the loop, and the
#: same URL is free again once the next run starts.
_SEEN_SQL = (
    "SELECT 1 FROM visited WHERE url = ?"
    " AND (success = 1 OR scraped_at IS NULL OR scraped_at >= ?)"
    " UNION ALL SELECT 1 FROM queue WHERE url = ? LIMIT 1"
)


class _CrawlQueue:
    """
    Persistent SQLite-backed URL queue with deduplication.

    This database is the crawl's state of record: queue membership, which pages
    are done, and the counters the live render reports all live here, so a run
    that dies mid-flight loses nothing beyond the pages that were in flight —
    plus, for a kill rather than an exception, up to one batch of writes, since
    commits are deferred (see ``COMMIT_EVERY``). That window is safe in the
    direction that matters: a page whose row did not make it is re-fetched, not
    skipped.

    Supports BFS ordering and visited tracking.

    Writes are deferred: mutating calls mark the connection dirty and the
    transaction is committed in batches (and on close). The previous version
    committed per operation, fsyncing three times per page on the event loop —
    ~765 us per page of pure blocking I/O, all of it serialized against fetches.
    In-memory counters answer the ``empty``/``queue_size`` questions that used
    to run ``COUNT(*)`` on every admission check, and a counter only ever moves
    when the statement that changed a row actually changed one.
    """

    #: Mutations between automatic commits.
    COMMIT_EVERY = 64

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        #: Marks the boundary between "this run" and earlier attempts, so a
        #: failure is retried on the next run rather than on every rediscovery.
        self._run_started = time.time()
        self._conn = self._connect(db_path)
        self._closed = False
        self._dirty = 0
        self._queued = 0
        self._visited = 0
        self._init_db()

    def _connect(self, db_path: Path) -> sqlite3.Connection:
        """
        Open *db_path*, quarantining it if it is not a usable database.

        A crawl killed mid-write, or a directory synced from somewhere else,
        can leave a file that is not a database at all. That raised
        ``sqlite3.DatabaseError: file is not a database`` out of the crawler
        constructor — unlike the checkpoint JSON beside it, which is already
        handled — so one bad byte turned ``--resume`` into a traceback.

        The file is *renamed*, not deleted: the crawl history it claims to hold
        may still be worth recovering by hand, and silently discarding a user's
        file to make an error go away is worse than reporting it.
        """
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            # Touch the file rather than trusting connect(): a corrupt database
            # opens fine and only fails on first real use.
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite3.DatabaseError as exc:
            conn.close()
            quarantine = db_path.with_name(f"{db_path.name}.corrupt")
            counter = 0
            while quarantine.exists():
                counter += 1
                quarantine = db_path.with_name(f"{db_path.name}.corrupt{counter}")
            with contextlib.suppress(OSError):
                db_path.rename(quarantine)
            console.print(
                f"  {warn('crawl queue')} {muted(str(db_path))} was not a usable database "
                f"({exc}). Moved to {muted(str(quarantine))} and starting a fresh queue."
            )
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        return conn

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
        """
        Add URL to queue if not already queued or visited. Returns True if added.

        This runs once per discovered link on every page, making it the most
        frequently called method in the crawler — and every microsecond here is
        event-loop time. So it canonicalises once (``canonicalize_url`` is
        idempotent, so re-running it inside the existence probes was wasted work)
        and then answers the one question that matters with one statement.
        """
        url = canonicalize_url(url)
        if self._conn.execute(_SEEN_SQL, (url, self._run_started, url)).fetchone() is not None:
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

    def mark_visited(self, url: str, success: bool = True, *, attempted: bool = True) -> None:
        """
        Record *url* as processed, counting it only if the row is genuinely new.

        *attempted* says whether the URL was actually requested. False records a
        filtered URL — off-domain, blocked by robots, blocked by the ad list —
        as distinct from one that was fetched and failed, so :meth:`requeue_failed`
        retries only the latter.

        ``INSERT OR REPLACE`` always reports a change, so incrementing on its
        result counted a repeated mark as another page: ``visited_count`` drifted
        away from ``COUNT(*)`` — the number reopening the database computes — and
        a resumed crawl reported more pages scraped than existed. Insert-if-absent
        first, then refresh the row on the rare repeat: the last outcome still
        wins, but the counter only ever counts rows.
        """
        url = canonicalize_url(url)
        now = time.time()
        # -1 is "not attempted": the URL was filtered out (off-domain, blocked
        # by robots or the ad list) rather than requested and refused. It has to
        # be distinguishable from 0, because a resume retries the pages that
        # failed — and re-queueing a URL that was never requested puts the whole
        # filtered set at the front of the queue, where it is dispatched, skipped
        # and re-skipped on every run instead of fetching anything. Only `success
        # = 1` counts as scraped, so the other two read alike everywhere else.
        outcome = 1 if success else (0 if attempted else -1)
        inserted = self._conn.execute(
            "INSERT OR IGNORE INTO visited (url, scraped_at, success) VALUES (?, ?, ?)",
            (url, now, outcome),
        ).rowcount
        if inserted:
            self._visited += 1
        else:
            self._conn.execute(
                "UPDATE visited SET scraped_at = ?, success = ? WHERE url = ?",
                (now, outcome, url),
            )
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
        """Successful pages. A full scan, so this is a resume-time question only."""
        row = self._conn.execute("SELECT COUNT(*) FROM visited WHERE success = 1").fetchone()
        return row[0] if row else 0

    def requeue_failed(self) -> int:
        """
        Put every previously failed page back on the queue, returning how many.

        Only pages that were *requested* and failed. A URL that was filtered out
        — off-domain, refused by robots.txt, blocked by the ad list — was never
        asked for and would be refused identically, so re-queueing it would put
        the whole filtered set at the head of every resumed crawl, to be
        dispatched and skipped again while the budget it should have been
        fetching with went unspent.

        :meth:`enqueue` refuses a URL whose last attempt was during *this* run,
        which is what keeps a failing page from being rediscovered in a loop — so
        a failure recorded a moment ago is not re-admittable, and saying
        otherwise here would describe a retry that does not happen. What it does
        allow is the retry a resume exists for: a 502 or a timeout from an hour
        ago is usually fine now, and a permanent 404 costs one request to find
        out again. Left to ``enqueue`` alone, a resumed crawl found those pages
        already "seen" and never looked at them again, silently accepting the
        first attempt as the answer.

        The pages stay in ``visited`` — the failure is a fact about the past, and
        ``mark_visited`` overwrites it when the retry lands. Their ``scraped_at``
        stays behind this run's start, so an admission check inside *this* run
        still rejects them if something links to them again: one retry, not a
        loop.
        """
        before = self._conn.total_changes
        self._conn.execute(
            "INSERT OR IGNORE INTO queue (url, priority, added_at) "
            "SELECT url, 0, ? FROM visited WHERE success = 0",
            (time.time(),),
        )
        added = self._conn.total_changes - before
        self._queued += added
        self._commit()
        return added

    def has_state(self) -> bool:
        """Whether this database holds queue or visited rows from a prior run."""
        return bool(self._queued or self._visited)

    def clear_state(self) -> int:
        """
        Forget a previous run's queue and visited rows, returning how many went.

        Only the crawl *state* is discarded: the page files and manifests already
        written stay on disk, and a crawl that revisits a page overwrites them.
        Doing it by SQL rather than by deleting the database avoids disturbing
        WAL sidecar files, and leaving the file itself keeps the path stable for
        anyone watching it. Returns the number of visited rows dropped.
        """
        removed = self._visited
        self._conn.execute("DELETE FROM queue")
        self._conn.execute("DELETE FROM visited")
        self._conn.commit()
        self._queued = 0
        self._visited = 0
        return removed

    def close(self) -> None:
        """Flush and close. Idempotent, so repeated teardown is safe."""
        if self._closed:
            return
        self._commit(force=True)
        self._closed = True
        self._conn.close()


# ── crawl state ──────────────────────────────────────────────────────────────


#: Shortest gap, in scraped pages, between two checkpoint writes. Closer than
#: this and the summary says nothing an interrupted crawl could not recompute.
_CHECKPOINT_MIN_INTERVAL = 5

#: How many checkpoint writes a crawl may spend, whatever its size: the interval
#: is derived from the page budget so writes stay roughly this many per run
#: instead of growing with the crawl.
_CHECKPOINT_WRITES = 20

#: Entries kept in the live log. The view only renders the most recent slice,
#: so retaining every page of a large crawl grew memory for nothing.
_LOG_HISTORY = 200

#: Rows shown in the live log table.
_LOG_VIEW = 20


@dataclass
class _CrawlLog:
    status: str  # "ok" | "err" | "active" | "blocked" | "skip"
    domain: str
    note: str = ""
    url: str = ""


@dataclass
class _State:
    scraped: int = 0
    errors: int = 0
    blocked: int = 0
    #: Pages the server confirmed are unchanged (HTTP 304). A subset of
    #: `scraped`, and the number that distinguishes a re-crawl from a first one.
    unchanged: int = 0
    current: str = ""
    #: Pages the engine declined — not HTML, off-domain, over the link cap. They
    #: were dispatched and then discarded, so a run that queued 30 and scraped 10
    #: used to report only the 10.
    skipped: int = 0
    queue_n: int = 0
    max_pages: int = DEFAULT_MAX_PAGES
    log: deque[_CrawlLog] = field(default_factory=lambda: deque(maxlen=_LOG_HISTORY))
    #: Total log entries ever appended, so numbering stays stable once the
    #: deque starts discarding old rows.
    log_total: int = 0
    #: Failure cause -> count, over the whole run. Not derived from `log`, which
    #: keeps only the last 200 rows: reading reasons back off a bounded view would
    #: silently misreport every failure earlier in a long crawl.
    reasons: Counter[str] = field(default_factory=Counter)


_BAR_WIDTH = 32


def _render(state: _State, output_dir: str) -> Group:
    # The bar is scaled to a fixed width. One cell per page made --max-pages
    # 500 render a 500-character bar that wrapped and wrecked the layout.
    if state.max_pages:
        # Floor, not round, so the bar and the percentage cannot disagree. Rounding
        # filled the whole bar at 98.4% — `round(32 * 63/64)` is 32 — so the last
        # page or two of every crawl rendered as a full bar next to "98%".
        fraction = min(state.scraped / state.max_pages, 1.0)
        filled = int(_BAR_WIDTH * fraction)
        pct = int(fraction * 100)
    else:
        filled = 0
        pct = 0
    filled = max(0, min(filled, _BAR_WIDTH))
    bar = safe("█" * filled + "░" * (_BAR_WIDTH - filled))

    stat = Table(box=box.SIMPLE, show_header=False, show_edge=False, padding=(0, 1))
    stat.add_column(width=10, style="grey74")
    stat.add_column(style="white")
    stat.add_row(
        "progress",
        f"[grey50]{bar}[/grey50]  [white]{pct}%[/white]  [grey50]{state.scraped}/{state.max_pages}[/grey50]",
    )
    stat.add_row(
        "current",
        muted(state.current[:72]) if state.current else f"[grey23]{safe(chr(0x2014))}[/grey23]",
    )
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

    recent = list(state.log)[-_LOG_VIEW:]
    first_index = max(1, state.log_total - len(recent) + 1)
    for i, entry in enumerate(recent, first_index):
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
        log_t.add_row(str(i), content(entry.domain), s)

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
        live: bool = True,
        allow_internal_redirects: bool = False,
        download_js: bool = False,
        use_sitemaps: bool = False,
        use_cache: bool = False,
    ) -> None:
        self.start_url = start_url
        self.max_pages = max_pages
        self.output_dir = Path(output_dir or get_default_output_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.auto_scale = auto_scale
        self._live = live
        self._allow_internal_redirects = allow_internal_redirects
        self._download_js = download_js
        self._use_sitemaps = use_sitemaps
        # Opt-in, like the scraper's --cache: a cache changes what a repeat run
        # sees, and silently serving a day-old page is not a default anyone
        # should get. With it, a re-crawl of an unchanged site costs one
        # conditional request per page and no body.
        self._cache = HTTPCache() if use_cache else None

        self._base_domain = urlparse(start_url).netloc
        self._state = _State(max_pages=max_pages)
        self._log_index: dict[str, _CrawlLog] = {}

        # The queue database *is* the crawl state, and it is opened whether or
        # not --resume was passed: an interrupted run committed its rows in
        # batches, so the next run finds them whether or not it was asked to.
        db_path = self.output_dir / "crawl_queue.db"
        self._queue = _CrawlQueue(db_path)

        checkpoint_path = self.output_dir / CHECKPOINT_FILENAME
        if not resume and self._queue.has_state():
            # The queue database is opened whether or not --resume was passed, so
            # an earlier run's rows used to suppress a fresh crawl silently: the
            # user asked to crawl a site and got zero requests and no
            # explanation. A plain `protor crawl URL` now means "crawl it", and
            # continuing is what --resume is for.
            dropped = self._queue.clear_state()
            console.print(
                f"  {warn('Starting a fresh crawl')}{muted(f' — cleared {dropped} pages of previous crawl state')}"
            )
            console.print(f"  {info('Use --resume to continue an interrupted crawl instead.')}")
        elif resume:
            if checkpoint_path.exists():
                self._report_checkpoint(checkpoint_path)
            # Price the remaining budget from the rows already on disk. This used
            # to be read out of the JSON summary and then *replayed* into a brand
            # new connection on this same database, so every URL was written
            # twice, a visited URL could be pushed back into the queue, and the
            # two counters could disagree.
            scraped = self._queue.success_count
            if scraped:
                self._state.scraped = scraped
                console.print(
                    f"  {OK_STYLED} Resumed from checkpoint — {self._state.scraped} pages already scraped"
                )
            # A page that failed in an earlier run is the one thing a resume has
            # left to offer: the successes are done by definition, so without
            # this the second run could only repeat the first run's failures.
            retrying = self._queue.requeue_failed()
            if retrying:
                console.print(f"  {info(f'Retrying {retrying} previously failed pages')}")

        # Always ensure start_url is queued (a no-op once it has been scraped)
        self._queue.enqueue(start_url)

    def _report_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Parse the run summary, reporting a damaged file rather than swallowing it.

        Silently starting over when the checkpoint was unreadable left the user
        with no clue why their ``--resume`` appeared to do nothing. The crawl
        carries on either way, from the queue database.
        """
        try:
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception as exc:
            console.print(
                f"  {warn(f'Could not resume from checkpoint: {exc}')}\n"
                f"  {muted('Continuing from the queue database instead.')}"
            )

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
            f"  {OK_STYLED} crawl complete — "
            f"{bright(str(self._state.scraped))} pages scraped"
            + (f" ({muted(str(self._state.unchanged))} unchanged)" if self._state.unchanged else "")
            + (f", {self._state.errors} errors" if self._state.errors else "")
            + (f", {self._state.blocked} blocked" if self._state.blocked else "")
            + (f", {self._state.skipped} skipped" if self._state.skipped else "")
        )
        console.print(f"  {label('output')} {muted(str(self.output_dir))}")
        console.print()
        # The log records a reason on every failure and the table has no column
        # for it, so without this a crawl could report "6 errors" and leave the
        # user to guess between DNS failure, HTTP 403, a timeout and robots.txt.
        print_failure_reasons(self._state.reasons)

    def _save_checkpoint(self) -> None:
        """
        Write a summary of the run beside the queue database.

        Constant time by construction: the previous version re-serialised every
        queued and visited URL into this file, so each write cost two full table
        scans plus a JSON encode of the entire crawl history (1.1 MB by 40,000
        pages) — all of which resume then discarded in favour of the rows the
        database had kept anyway.
        """
        cp = {
            "start_url": self.start_url,
            "max_pages": self.max_pages,
            "scraped": self._state.scraped,
            "queued": self._queue.queue_size,
            "visited": self._queue.visited_count,
            "timestamp": time.time(),
        }
        save_json(cp, self.output_dir / CHECKPOINT_FILENAME)

    # ── internal ──────────────────────────────────────────────────────────────

    async def _seed_from_sitemaps(self, budget: int) -> None:
        """
        Enqueue the pages the site's sitemap names, ahead of the link-walk.

        A link-walk only reaches what pages happen to link to, which on a docs
        site is the sidebar and on a shop the top nav. A sitemap is the site
        saying what exists.

        Best-effort throughout: a site with no sitemap, an unreadable one, or a
        hostile one costs a couple of requests and nothing else. The crawl is a
        link-walk with extra routes, never a sitemap reader that cannot start
        without one. Off-domain entries are dropped — a sitemap can list a
        CDN's or a sibling property's URLs — and the remaining budget bounds how
        many are taken, so a 50,000-URL sitemap cannot overrun ``--max-pages``
        before the walk has begun.
        """
        if budget <= 0:
            return
        import aiohttp

        from .robots import RobotsCache
        from .sitemap import discover_sitemap_urls

        domain = self._base_domain.lower()
        try:
            async with aiohttp.ClientSession() as session:
                found = await discover_sitemap_urls(
                    session,
                    self.start_url,
                    robots=RobotsCache(),
                    limit=budget,
                )
        except Exception as exc:  # A sitemap is an optimisation, never a precondition.
            console.print(f"  {warn('Sitemap unavailable')} {muted(f'({exc})')}\n")
            return

        added = 0
        for url, _lastmod in found:
            if urlparse(url).netloc.lower() != domain:
                continue
            if self._queue.enqueue(url):
                added += 1
            if added >= budget:
                break

        if added:
            console.print(
                f"  {OK_STYLED} Sitemap seeded {muted(str(added))} "
                f"additional {muted('page' if added == 1 else 'pages')}\n"
            )
        else:
            console.print(f"  {muted('No sitemap URLs found; crawling links only.')}\n")

    async def _run(self) -> None:
        scaler: AutoScaler | None = None
        if self.auto_scale:
            scaler = AutoScaler(initial=CRAWLER_CONCURRENCY)

        # --max-pages is a ceiling for the whole crawl, not for each run. On resume
        # the queue already holds pages scraped by earlier runs, so handing the
        # engine a fresh budget of max_pages let a resumed crawl finish with up to
        # twice the requested pages (and a progress bar pinned at 100%).
        max_targets = max(0, self.max_pages - self._state.scraped)

        if self._use_sitemaps:
            await self._seed_from_sitemaps(max_targets)

        engine = CrawlEngine(
            queue=self._queue,
            link_source=RecursiveSource(),
            output_dir=self.output_dir,
            max_targets=max_targets,
            concurrency=CRAWLER_CONCURRENCY,
            auto_scaler=scaler,
            allowed_domain=self._base_domain,
            check_robots=True,
            rate_limiter=DomainRateLimiter(delay=CRAWLER_DELAY),
            # A checkpoint is written from the crawl loop, so every one of them is
            # event-loop stall — and a fixed interval made the bill grow twice
            # over: more checkpoints as the crawl got longer, and dearer ones as
            # the history they serialised grew. Budgeting about _CHECKPOINT_WRITES
            # writes for the whole run keeps that flat; a 40,000-page crawl drops
            # from ~8,000 checkpoints to ~20.
            #
            # Replaying the lost interval costs little because the queue database,
            # not the checkpoint file, holds the state: a resumed crawl restarts
            # from the last committed rows rather than from the last summary.
            checkpoint_interval=max(_CHECKPOINT_MIN_INTERVAL, max_targets // _CHECKPOINT_WRITES),
            on_checkpoint=self._save_checkpoint,
            on_status=self._on_status,
            live_render=lambda: _render(self._state, str(self.output_dir)),
            live=self._live,
            allow_internal_redirects=self._allow_internal_redirects,
            download_js=self._download_js,
            cache=self._cache,
            # Nothing here reads engine.manifests; keeping one per page cost
            # ~49 KiB of retained text and markdown per page for a crawl that
            # never asked. The manifests are still written to disk.
            collect_manifests=False,
        )
        try:
            await engine.arun()
        finally:
            # Persist the validators even if the run died: they are the expensive
            # part to obtain, and losing them costs a full re-download next time.
            if self._cache is not None:
                self._cache.flush()

    def _on_status(self, status: str, url: str, row: dict[str, Any]) -> None:
        """Keep crawl state in sync with engine events for the live render."""
        # The engine already parsed this URL to scope the request — that parse is
        # where row["domain"] comes from — so re-parsing on every status event
        # bought nothing. Fall back to parsing for a row that arrives without one.
        domain = row.get("domain") or urlparse(url).netloc
        self._state.queue_n = self._queue.queue_size

        if status == "fetching":
            self._state.current = url
            self._append_log(url, "active", domain)
        elif status.startswith("js:"):
            # Status is "js:N"; comparing against the bare "js:" never matched.
            self._update_log(url, "active", domain)
        elif status == "done":
            self._state.scraped += 1
            if row.get("unchanged"):
                self._state.unchanged += 1
            self._update_log(url, "ok", domain)
        elif status == "error":
            self._state.errors += 1
            self._update_log(url, "err", domain, self._note(url, row))
        elif status == "blocked":
            self._state.blocked += 1
            self._append_log(url, "blocked", domain, self._note(url, row))
        elif status == "skipped":
            self._state.skipped += 1
            self._update_log(url, "skip", domain, self._note(url, row))

    def _note(self, url: str, row: dict[str, Any]) -> str:
        """Record *row*'s failure cause and return it for the log."""
        note = str(row.get("note", "") or "")
        self._state.reasons[normalise_reason(note)] += 1
        return note

    def _append_log(self, url: str, status: str, domain: str, note: str = "") -> None:
        """
        Add a log row, tracking it by URL so later events can update it in place.

        The log is a bounded deque, so rows are tracked by identity rather than
        by index: an index-based map silently pointed at the wrong row once the
        deque began discarding old entries. Evicted rows are dropped from the
        map too, keeping it bounded alongside the log.
        """
        log = self._state.log
        if len(log) == log.maxlen and log.maxlen:
            evicted = log[0]
            self._log_index.pop(evicted.url, None)
        entry = _CrawlLog(status, domain, note, url=url)
        log.append(entry)
        self._state.log_total += 1
        self._log_index[url] = entry

    def _update_log(self, url: str, status: str, domain: str, note: str = "") -> None:
        entry = self._log_index.get(url)
        if entry is None:
            self._append_log(url, status, domain, note)
            return
        entry.status = status
        entry.note = note
