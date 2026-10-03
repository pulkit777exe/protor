"""
protor.engine
~~~~~~~~~~~~~
Deep crawl engine. Every run of protor — batch scraping and recursive crawling
alike — is one loop with a pluggable queue and link source:

* a **queue** supplies the URLs still to visit (:class:`StaticQueue` for batch
  runs, the crawler's SQLite queue for recursive runs);
* a **link source** decides what to do with each scraped page (:class:`StaticSource`
  scrapes no further pages, :class:`RecursiveSource` harvests same-domain links);
* an optional **auto scaler** drives real admission concurrency from the recent
  success rate (batch and crawl both honor it).

The engine owns the aiohttp session and the rich progress display, emits
``on_status``/``on_checkpoint`` events for callers to observe, and guarantees a
hard ``max_targets`` ceiling on pages scraped.

Public API
----------
    CrawlEngine(queue, *, ...).run() → CrawlStats
    StaticQueue, StaticSource, RecursiveSource
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlparse

import aiohttp

from . import config
from .fetcher import download_file, fetch
from .models import SiteManifest
from .parser import parse_html
from .progress import live_display
from .robots import check_robots
from .theme import console
from .utils import manifest_filename, page_filename, safe_filename, save_json, timestamp

if TYPE_CHECKING:
    from collections.abc import Callable

    from .blocklist import Blocklist
    from .extractor import ExtractionSchema
    from .http_cache import HTTPCache
    from .rate_limiter import DomainRateLimiter
    from .scaler import AutoScaler

__all__ = [
    "CrawlEngine",
    "CrawlStats",
    "LinkSource",
    "RecursiveSource",
    "StaticQueue",
    "StaticSource",
    "WorkQueue",
]


# ── plumbing seams ───────────────────────────────────────────────────────────


class WorkQueue(Protocol):
    """Protocol for the URL queue the engine drains."""

    def dequeue(self) -> str | None:
        """Pop the next URL to scrape, or None when empty."""
        raise NotImplementedError

    def enqueue(self, url: str, priority: int = 0) -> bool:
        """Schedule *url* (deduplicated). Returns True if newly added."""
        raise NotImplementedError

    def mark_visited(self, url: str, success: bool = True) -> None:
        """Record *url* as processed."""
        raise NotImplementedError

    @property
    def empty(self) -> bool:
        """True when no URLs remain."""
        return True


class StaticQueue:
    """In-memory queue over a fixed list of URLs (batch runs)."""

    def __init__(self, urls: list[str]) -> None:
        self._urls = list(urls)

    def dequeue(self) -> str | None:
        return self._urls.pop(0) if self._urls else None

    def enqueue(self, url: str, priority: int = 0) -> bool:
        return False

    def mark_visited(self, url: str, success: bool = True) -> None:
        pass

    @property
    def empty(self) -> bool:
        return not self._urls


class LinkSource(ABC):
    """Strategy producing next URLs from a scraped page."""

    @abstractmethod
    def discover(self, url: str, page: Any) -> list[str]:
        """Return the URLs to enqueue next, given *url* and its parsed page."""
        raise NotImplementedError


class StaticSource(LinkSource):
    """Batch scraping: never discovers further URLs."""

    def discover(self, url: str, page: Any) -> list[str]:
        return []


class RecursiveSource(LinkSource):
    """Recursive crawling: harvests the page's same-domain links."""

    def discover(self, url: str, page: Any) -> list[str]:
        return list(page.links)


# ── results ──────────────────────────────────────────────────────────────────


@dataclass
class CrawlStats:
    """Aggregate results of an engine run."""

    scraped: int = 0
    errors: int = 0
    blocked: int = 0
    bytes_total: int = 0
    dispatched: int = 0

    @property
    def total(self) -> int:
        return self.scraped + self.errors + self.blocked


# ── the engine ───────────────────────────────────────────────────────────────


class CrawlEngine:
    """
    Single crawl loop shared by batch scraping and recursive crawling.

    Parameters
    ----------
    queue:
        URL queue to drain, supporting the :class:`WorkQueue` protocol.
    link_source:
        Strategy deciding what to enqueue after each page is scraped.
    output_dir:
        Root directory for per-site artefacts (``<domain>/index.html`` and
        ``manifest.json``).
    max_targets:
        Hard ceiling on pages *attempted*; the engine never spawns work that
        could exceed it. Failures and blocked pages count toward the ceiling,
        so ``--max-pages 10`` issues at most 10 requests rather than walking
        on until 10 happen to succeed.
    concurrency:
        Number of pages fetched concurrently. When *auto_scaler* is provided,
        this is the initial value only; admission follows the scaler.
    rows:
        Optional list of row dicts consumed in dequeue order. Each successful
        page updates its row with ``status``/``bytes``/``ms``/``js``, keeping
        the batch table contract.
    auto_scaler:
        When given, drives admission concurrency from recent success rates.
    on_status:
        Called as ``on_status(url, status, row)`` on state transitions;
        statuses are ``fetching``, ``done``, ``error``, ``blocked`` and
        ``js:N``.
    on_checkpoint:
        Called every *checkpoint_interval* successful scrapes (0 disables).
    live_render:
        When provided, the engine renders it inside a live display.
    session:
        Optional pre-built session to reuse. The engine will not close it.
    """

    def __init__(
        self,
        queue: WorkQueue,
        link_source: LinkSource,
        output_dir: str | Path,
        max_targets: int,
        *,
        concurrency: int = config.CRAWLER_CONCURRENCY,
        rows: list[dict[str, Any]] | None = None,
        timeout: int = config.DEFAULT_TIMEOUT,
        download_js: bool = False,
        cache: HTTPCache | None = None,
        headers: dict[str, str] | None = None,
        hooks: dict[str, list[Callable[..., Any]]] | None = None,
        extraction_schema: ExtractionSchema | None = None,
        blocklist: Blocklist | None = None,
        rate_limiter: DomainRateLimiter | None = None,
        auto_scaler: AutoScaler | None = None,
        allowed_domain: str | None = None,
        check_robots: bool = False,
        checkpoint_interval: int = 0,
        on_checkpoint: Callable[[], None] | None = None,
        on_status: Callable[[str, str, dict[str, Any]], None] | None = None,
        live_render: Callable[[], Any] | None = None,
        live: bool = True,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._queue = queue
        self._link_source = link_source
        self._output_dir = Path(output_dir)
        self._max_targets = max_targets
        self._concurrency = concurrency
        self._rows = list(rows or [])
        self._timeout = timeout
        self._download_js = download_js
        self._cache = cache
        self._headers = headers
        self._hooks = hooks
        self._extraction_schema = extraction_schema
        self._blocklist = blocklist
        self._rate_limiter = rate_limiter
        self._auto_scaler = auto_scaler
        self._allowed_domain = allowed_domain
        self._check_robots = check_robots
        self._checkpoint_interval = checkpoint_interval
        self._on_checkpoint = on_checkpoint
        self._on_status = on_status
        self._live_render = live_render
        self._live = live
        self._session = session

        self._manifests: list[SiteManifest] = []

    @property
    def manifests(self) -> list[SiteManifest]:
        """Site manifests produced by the run (populated after :meth:`run`)."""
        return self._manifests

    # ── public ────────────────────────────────────────────────────────────────

    def run(self) -> CrawlStats:
        """Run the crawl loop to completion (blocking)."""
        return asyncio.run(self.arun())

    async def arun(self) -> CrawlStats:
        """
        Async form of :meth:`run`, for callers already inside an event loop.

        When *session* is supplied it is reused as-is (and left open for its
        owner to close), so callers that already hold a connection pool do not
        pay for a second one.
        """
        if self._session is not None:
            return await self._rendered(self._session)

        connector = aiohttp.TCPConnector(limit=self._connector_limit())
        session_headers = {**config.HEADERS, **(self._headers or {})}

        async with aiohttp.ClientSession(
            headers=session_headers,
            connector=connector,
            cookie_jar=aiohttp.CookieJar(),
        ) as session:
            return await self._rendered(session)

    async def _rendered(self, session: aiohttp.ClientSession) -> CrawlStats:
        """
        Run the loop with progress rendered in place.

        The render callback fires once per completed page and building the table
        costs time proportional to the number of rows, so honouring every tick
        made the display quadratic in the batch size — measured at 8.4 ms of
        blocking render per tick with 3,000 rows, about 25 s of event-loop stall
        over such a run, slowing the very crawl it was reporting on. Throttling
        is invisible to a human: 10 Hz is far past the rate at which progress
        reads as "live".
        """
        render = self._live_render
        if render is None:
            return await self._start(session)
        with live_display(render, console=console, transient=False, enabled=self._live) as display:
            return await self._start(session, on_tick=display.update)

    def _connector_limit(self) -> int:
        """Connection-pool ceiling: static for batch, scaler-aware for crawls."""
        if self._auto_scaler is not None:
            return max(self._concurrency, config.SCALING_MAX_CONCURRENCY)
        return self._concurrency

    async def _start(self, session: aiohttp.ClientSession, on_tick: Any = None) -> CrawlStats:
        stats = CrawlStats()
        pending: set[asyncio.Task[Any]] = set()
        rows = list(self._rows)
        checkpointed = 0
        # Which page each in-flight task is working on, so a failure that escapes
        # the task can be reported against the right row.
        in_flight: dict[asyncio.Task[Any], tuple[str, dict[str, Any]]] = {}

        def spawn() -> None:
            # stats.total counts every dispatched page, so failures and blocked
            # URLs count against max_targets instead of letting the crawl run on.
            while (
                not self._queue.empty
                and len(pending) < self._admission()
                and (stats.total + len(pending) < self._max_targets)
            ):
                url = self._queue.dequeue()
                if url is None:
                    break
                row = rows.pop(0) if rows else {}
                stats.dispatched += 1
                task = asyncio.create_task(self._process_one(session, url, row, stats))
                in_flight[task] = (url, row)
                pending.add(task)

        spawn()
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                url, row = in_flight.pop(task, ("", {}))
                try:
                    discovered = task.result()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # _process_one handles its own failures and routes them
                    # through _fail. Reaching here means a bug (a TypeError in a
                    # hook, a bad manifest field) rather than a bad page, and
                    # swallowing it left the user a row stuck on "fetching" with
                    # no explanation plus a re-fetch on every resumed crawl.
                    self._fail(stats, row, url, f"internal error: {exc!r}")
                    continue
                for link in discovered:
                    self._queue.enqueue(link)

            if self._auto_scaler is not None:
                self._auto_scaler.maybe_scale()
            # Compare against the last checkpoint rather than using a modulo:
            # `scraped % n == 0` is true at zero (firing a checkpoint before any
            # work) and re-fires every round while the count is unchanged.
            if (
                self._checkpoint_interval
                and self._on_checkpoint is not None
                and stats.scraped - checkpointed >= self._checkpoint_interval
            ):
                checkpointed = stats.scraped
                self._on_checkpoint()
            if on_tick is not None:
                on_tick()
            spawn()

        if on_tick is not None:
            on_tick()
        return stats

    def _admission(self) -> int:
        """Concurrency ceiling for new spawns (scaler-aware)."""
        if self._auto_scaler is not None:
            return self._auto_scaler.concurrency
        return self._concurrency

    async def _process_one(
        self,
        session: aiohttp.ClientSession,
        url: str,
        row: dict[str, Any],
        stats: CrawlStats,
    ) -> list[str]:
        parsed = urlparse(url)
        domain = parsed.netloc or url
        row["domain"] = domain

        if self._allowed_domain and parsed.netloc != self._allowed_domain:
            # Counted and reported like any other non-fetch, so off-domain links
            # dropped by the domain filter are visible instead of vanishing.
            self._skip(stats, row, url, f"off-domain ({parsed.netloc})")
            return []

        if self._blocklist is not None and self._blocklist.is_url_blocked(url):
            self._block(stats, row, url, "blocked by the ad/analytics blocklist")
            return []

        if self._check_robots and not await check_robots(url, session):
            self._block(stats, row, url, "blocked by robots.txt")
            return []

        if self._rate_limiter is not None:
            await self._rate_limiter.wait(domain)

        row["status"] = "fetching"
        self._emit("fetching", url, row)
        t0 = time.perf_counter()

        try:
            result = await fetch(
                session,
                url,
                timeout=self._timeout,
                cache=self._cache,
                hooks=self._hooks,
            )
        except Exception as exc:
            self._fail(stats, row, url, str(exc))
            return []

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        site_dir = self._output_dir / safe_filename(parsed.netloc)

        try:
            site_dir.mkdir(parents=True, exist_ok=True)
            html_file = site_dir / page_filename(url)
            html_file.write_text(result.text, encoding="utf-8")

            soup, page = parse_html(result.text, url)
            for hook in (self._hooks or {}).get("before_parse", []):
                self._safe_hook(hook, url, {"soup": soup, "html": result.text})
            for hook in (self._hooks or {}).get("after_parse", []):
                self._safe_hook(hook, url, {"soup": soup, "markdown": page.markdown_content})

            js_downloaded: list[str] = []
            if self._download_js and page.js_links:
                # The blocklist guards the page fetch, but script tags point at
                # third-party CDNs — exactly the ad/tracker hosts --block-ads
                # exists to avoid. Filter them here too, or the flag silently
                # does nothing while still making the requests.
                js_links = [
                    j
                    for j in page.js_links[: config.MAX_JS_FILES]
                    if self._blocklist is None or not self._blocklist.is_url_blocked(j)
                ]
                if js_links:
                    row["status"] = f"js:{len(js_links)}"
                    self._emit(f"js:{len(js_links)}", url, row)
                    js_dir = site_dir / "js"
                    js_dir.mkdir(parents=True, exist_ok=True)
                    taken: set[str] = set()
                    tasks = [
                        asyncio.create_task(
                            download_file(
                                session,
                                jurl,
                                js_dir / self._js_filename(i, jurl, taken),
                            )
                        )
                        for i, jurl in enumerate(js_links)
                    ]
                    # Bound the whole group, not just each download. A page whose
                    # only script pointed at a blackholed CDN stalled for the full
                    # per-file timeout (measured 15.5 s) before the page could
                    # finish, holding its concurrency slot the whole time.
                    # Whatever landed in time is kept; the rest are simply not
                    # downloaded, which is best-effort by design.
                    done, pending_js = await asyncio.wait(tasks, timeout=config.JS_GROUP_TIMEOUT)
                    for task in pending_js:
                        task.cancel()
                    if pending_js:
                        await asyncio.gather(*pending_js, return_exceptions=True)
                    ok_flags: dict[int, bool] = {}
                    for i, task in enumerate(done):
                        if not task.cancelled() and task.exception() is None:
                            ok_flags[i] = bool(task.result())
                    js_downloaded = [u for i, u in enumerate(js_links) if ok_flags.get(i)]

            extracted = None
            if self._extraction_schema is not None:
                from .extractor import extract_from_soup

                # Reuse the tree we already parsed instead of re-parsing the
                # HTML with lxml a second time.
                extracted = extract_from_soup(soup, self._extraction_schema, base_url=url)

            manifest = SiteManifest(
                url=url,
                domain=domain,
                html_file=str(html_file),
                metadata=page.metadata,
                text_content=page.text_content,
                js_files=js_downloaded,
                js_count=len(js_downloaded),
                bytes_received=result.nbytes,
                elapsed_ms=elapsed_ms,
                timestamp=timestamp(),
                success=True,
                markdown_content=page.markdown_content,
                extracted_data=extracted,
            )
            save_json(manifest.to_dict(), site_dir / manifest_filename(url))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(stats, row, url, str(exc))
            return []

        stats.scraped += 1
        stats.bytes_total += result.nbytes
        self._manifests.append(manifest)
        row.update(
            status="done",
            ms=elapsed_ms,
            bytes=result.nbytes,
            js=len(js_downloaded),
            error=False,
        )
        self._emit("done", url, row)
        self._queue.mark_visited(url, success=True)
        self._record_scaler(True)
        return self._link_source.discover(url, page)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _js_filename(index: int, jurl: str, taken: set[str] | None = None) -> str:
        """
        Return a collision-free filename for a downloaded script.

        Basenames are not unique across CDNs (``/static/vendor.js`` and
        ``/lib/vendor.js`` both wanted ``vendor.js``), which silently
        overwrote earlier downloads while the manifest still listed every URL.
        A short hash of the full URL keeps distinct files distinct.
        """
        name = Path(urlparse(jurl).path).name
        stem = safe_filename(Path(name).stem if name else "") or f"script-{index}"
        suffix = Path(name).suffix if name else ".js"
        candidate = f"{stem}{suffix}"
        if taken is None or candidate not in taken:
            if taken is not None:
                taken.add(candidate)
            return candidate
        digest = hashlib.sha256(jurl.encode("utf-8")).hexdigest()[:8]
        candidate = f"{stem}.{digest}{suffix}"
        if taken is not None:
            taken.add(candidate)
        return candidate

    def _skip(
        self, stats: CrawlStats, row: dict[str, Any], url: str, note: str, status: str = "skipped"
    ) -> None:
        """
        Record a page the engine chose not to fetch.

        Not counted as an error: the URL was never requested, so it must not
        consume the request budget beyond the dispatch accounting above.
        """
        row.update(status=status, error=False, note=note)
        self._emit(status, url, row)
        self._queue.mark_visited(url, success=False)

    def _block(self, stats: CrawlStats, row: dict[str, Any], url: str, note: str) -> None:
        stats.blocked += 1
        row.update(status="blocked", error=True, note=note)
        self._emit("blocked", url, row)
        self._queue.mark_visited(url, success=False)
        self._record_scaler(False)

    def _fail(self, stats: CrawlStats, row: dict[str, Any], url: str, note: str) -> None:
        stats.errors += 1
        row.update(status="error", error=True, note=note)
        self._emit("error", url, row)
        self._queue.mark_visited(url, success=False)
        self._record_scaler(False)

    def _record_scaler(self, success: bool) -> None:
        if self._auto_scaler is not None:
            self._auto_scaler.record(success)

    def _emit(self, status: str, url: str, row: dict[str, Any]) -> None:
        if self._on_status is not None:
            with contextlib.suppress(Exception):
                self._on_status(status, url, row)

    def _safe_hook(self, hook: Callable[..., Any], url: str, ctx: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            hook(url, ctx)
