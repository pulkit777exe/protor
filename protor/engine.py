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
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlparse

import aiohttp

from . import config
from .fetcher import download_file, fetch, random_user_agent
from .models import SiteManifest
from .parser import looks_like_html, parse_html
from .progress import LiveDisplay, live_display
from .robots import check_robots
from .theme import console
from .utils import (
    canonicalize_url,
    human_bytes,
    human_duration,
    manifest_filename,
    page_filename,
    safe_filename,
    save_json,
    timestamp,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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

    def mark_visited(self, url: str, success: bool = True, *, attempted: bool = True) -> None:
        """
        Record *url* as processed.

        *attempted* distinguishes a URL that was requested and refused from one
        that was filtered out before any request — see ``_CrawlQueue`` for why the
        distinction has to survive into the database.
        """
        raise NotImplementedError

    @property
    def empty(self) -> bool:
        """True when no URLs remain."""
        return True


class StaticQueue:
    """In-memory queue over a fixed list of URLs (batch runs)."""

    def __init__(self, urls: list[str]) -> None:
        # A deque, not a list: pop(0) shifts every remaining element, so a batch
        # cost grew with the square of its size — measured 2.9 us per dequeue at
        # 50,000 URLs against 0.02 us for popleft, a 117x gap on the only
        # remaining quadratic in the crawl path. Same order, same semantics.
        self._urls: deque[str] = deque(urls)

    def dequeue(self) -> str | None:
        return self._urls.popleft() if self._urls else None

    def enqueue(self, url: str, priority: int = 0) -> bool:
        return False

    def mark_visited(self, url: str, success: bool = True, *, attempted: bool = True) -> None:
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
    #: Pages the server confirmed are unchanged (HTTP 304). A subset of
    #: ``scraped`` — they were served from the cache — and the number that tells
    #: a re-crawl apart from a first one.
    unchanged: int = 0

    @property
    def total(self) -> int:
        return self.scraped + self.errors + self.blocked


# ── the engine ───────────────────────────────────────────────────────────────


#: Status -> (glyph, word) for the one-line-per-result format a pipe gets. Kept here
#: rather than in scraper.py so the engine, which is what knows when a row has an
#: outcome, does not have to import a renderer to describe one.
_STATUS_GLYPHS = {"done": "✓", "error": "✗", "blocked": "✗", "skipped": "-"}
_STATUS_WORDS = {
    "done": "done",
    "error": "error",
    "blocked": "blocked",
    "skipped": "skipped",
}


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
        so ``--max-pages 10`` attempts at most 10 pages rather than walking
        on until 10 happen to succeed.

        Pages, not HTTP requests: the fetcher retries a 429 or 5xx up to
        ``MAX_RETRIES`` before the page is recorded as failed, so one page
        attempt can cost several requests on the wire. That is deliberate — a
        transient 502 is worth another go — and it is why this is a ceiling over
        work rather than over bandwidth.

        The ceiling counts dispatched URLs through ``stats.total``, which a
        *skipped* URL does not advance — so it held only because nothing can skip.
        The parser yields same-host links exclusively, so a recursive crawl can
        never hand the domain filter a URL to reject, and every dispatched task
        ends in scraped, error or blocked. That is an invariant rather than a
        coincidence, and one test now pins it: loosening the parser's host check
        would otherwise turn this ceiling into a suggestion with nothing to say
        so.
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
    requested_hosts:
        Hosts the caller asked for by name, which the blocklist will not refuse.
        *allowed_domain* is included automatically.
    collect_manifests:
        Keep each page's manifest in :attr:`manifests` as well as writing it to
        disk. The batch scraper reads them; the crawler does not, and a crawl of
        40,000 pages was holding about 1.9 GB of text and markdown that nothing
        ever looked at. Manifests are written either way — this only governs
        whether they are also retained in memory.
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
        allow_internal_redirects: bool = False,
        collect_manifests: bool = True,
        requested_hosts: Sequence[str] | None = None,
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
        # Script filenames this run has already reserved, keyed by site
        # directory and then by URL, so a second page of the same site cannot be
        # handed a name the first used. See _reserve_js_filename.
        self._js_names: dict[str, dict[str, str]] = {}
        self._extraction_schema = extraction_schema
        self._blocklist = blocklist
        self._allow_internal_redirects = allow_internal_redirects
        self._rate_limiter = rate_limiter
        self._auto_scaler = auto_scaler
        # Lowercased because the queue canonicalises the host while this arrives from
        # urlparse(), which keeps the case the user typed. Compared raw, a seed of
        # https://EXAMPLE.com/ was rejected as off-domain by its own canonical
        # form and the crawl reported zero pages with no explanation.
        self._allowed_domain = allowed_domain.lower() if allowed_domain else allowed_domain
        self._check_robots = check_robots
        self._checkpoint_interval = checkpoint_interval
        self._on_checkpoint = on_checkpoint
        self._on_status = on_status
        self._live_render = live_render
        self._live = live
        self._session = session

        # Manifests are written to disk either way; whether they are also *kept*
        # is the caller's choice. A crawl has no consumer for them — it reads
        # CrawlStats — and holding one per page cost ~49 KiB of retained strings
        # per page, so a 40,000-page crawl kept about 1.9 GB that nothing read.
        self._manifests: list[SiteManifest] = []
        self._collect_manifests = collect_manifests
        # Set for the duration of arun() when there is a display to write to.
        self._display: LiveDisplay | None = None
        #: Live counts for the current run; see :meth:`_start`.
        self.stats = CrawlStats()
        # Hosts the caller named, which the ad/analytics blocklist must not
        # second-guess. `--block-ads` exists to stop a page pulling a tracker off
        # a CDN; a user who typed `protor scrape https://www.facebook.com
        # --block-ads` asked for facebook, and refusing it as an ad network made
        # the command fetch nothing and say the target was blocked.
        self._requested_hosts = {h.lower() for h in (requested_hosts or ())}
        if self._allowed_domain:
            self._requested_hosts.add(self._allowed_domain.lower())

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
            self._display = display
            try:
                return await self._start(session, on_tick=display.update)
            finally:
                self._display = None

    def _connector_limit(self) -> int:
        """Connection-pool ceiling: static for batch, scaler-aware for crawls."""
        if self._auto_scaler is not None:
            return max(self._concurrency, config.SCALING_MAX_CONCURRENCY)
        return self._concurrency

    async def _start(self, session: aiohttp.ClientSession, on_tick: Any = None) -> CrawlStats:
        stats = CrawlStats()
        # Published as it is built, not on return, so a caller catching
        # KeyboardInterrupt can still read the counts for the pages that finished.
        # Reporting "interrupted" and nothing else is how a ten-minute batch used
        # to end, with every page already on disk.
        self.stats = stats
        pending: set[asyncio.Task[Any]] = set()
        rows = list(self._rows)
        checkpointed = 0
        # Which page each in-flight task is working on, so a failure that escapes
        # the task can be reported against the right row.
        in_flight: dict[asyncio.Task[Any], tuple[str, dict[str, Any]]] = {}
        # URLs currently being fetched. A dispatched page leaves the queue, so
        # without this another page linking to it re-admits it and it is fetched
        # again — and each duplicate re-discovers the same links, which on a
        # cyclic site multiplies the frontier until the queue table outgrows the
        # crawl. Observed: a five-page site producing a 7.8 GB queue database.
        fetching: set[str] = set()

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
                fetching.add(url)
                pending.add(task)

        spawn()
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                url, row = in_flight.pop(task, ("", {}))
                # Free the URL before enqueueing its links: they may include the
                # page that just finished, and that page is now in `visited`, so
                # the queue rejects it anyway.
                fetching.discard(url)
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
                    # The queue rejects what it has already seen, but a page
                    # that is mid-fetch is in neither `queue` nor `visited`
                    # yet — without this it is re-admitted, fetched twice, and
                    # charged twice against --max-pages.
                    #
                    # Compared canonically, because `fetching` holds the URLs
                    # the queue handed out and those are canonical while a
                    # discovered link is however the page spelled it. A site
                    # linking `/docs/index.html` while `/docs/` was in flight
                    # spells the same page two ways, and comparing raw strings
                    # fetched it twice: one extra request, and the crawl
                    # reported one more page than the site has.
                    if canonicalize_url(link) in fetching:
                        continue
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

        if self._allowed_domain and parsed.netloc.lower() != self._allowed_domain:
            # Counted and reported like any other non-fetch, so off-domain links
            # dropped by the domain filter are visible instead of vanishing.
            self._skip(stats, row, url, f"off-domain ({parsed.netloc})")
            return []

        requested = parsed.netloc.lower() in self._requested_hosts
        if self._blocklist is not None and not requested and self._blocklist.is_url_blocked(url):
            self._block(stats, row, url, "blocked by the ad/analytics blocklist")
            return []

        # One identity for both the question and the request. check_robots
        # documents that it must be the string the request will actually send:
        # evaluating the "*" group while transmitting a browser User-Agent asks
        # the site about a policy it never agreed to, so a `User-agent: Mozilla`
        # Disallow is not consulted and a `User-agent: Googlebot` one is not
        # mistaken for ours. Rotation still happens — once per page, decided here.
        user_agent = random_user_agent()
        if self._check_robots and not await check_robots(url, session, user_agent):
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
                allow_internal_redirects=self._allow_internal_redirects,
                user_agent=user_agent,
            )
        except Exception as exc:
            self._fail(stats, row, url, str(exc))
            return []

        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        # A body is not a page by virtue of arriving over HTTP. Without this a
        # link to a manual.pdf is "scraped" into two thousand characters of
        # %PDF-1.4 and reported as a successfully scraped page — the same
        # failure-as-success shape as a stale CSS selector, one layer down.
        # Recorded like a filter rather than a failure, so a resumed crawl does
        # not re-request the same PDF on every run.
        if not looks_like_html(result.content_type, result.text):
            self._skip(
                stats,
                row,
                url,
                f"not a web page ({result.content_type or 'no content-type'})",
            )
            return []

        site_dir = self._output_dir / safe_filename(parsed.netloc)

        try:
            site_dir.mkdir(parents=True, exist_ok=True)
            html_file = site_dir / page_filename(url)
            html_file.write_text(result.text, encoding="utf-8")

            # With a schema, the guessed-noise patterns yield: a schema's
            # selectors are the caller's statement of what the page contains, and
            # a `.ad-card` or `.related-posts` is ordinary content to them. They
            # were being deleted before extraction ran, so the run reported a
            # successful extraction of nothing — and lost the same content from
            # the text and markdown beside it.
            soup, page = parse_html(
                result.text,
                url,
                strip_guessed_noise=self._extraction_schema is None,
            )
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
                    # The index travels with its task. asyncio.wait() returns a
                    # *set*, so numbering the results afterwards paired whatever
                    # order the set happened to iterate with whatever script sat
                    # at that index — the manifest then claimed files that had
                    # 404'd and silently dropped files that were really written.
                    tasks = {
                        asyncio.create_task(
                            download_file(
                                session,
                                jurl,
                                js_dir / self._reserve_js_filename(site_dir.name, i, jurl, taken),
                            )
                        ): i
                        for i, jurl in enumerate(js_links)
                    }
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
                    for task in done:
                        i = tasks[task]
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
        if result.not_modified:
            stats.unchanged += 1
        if self._collect_manifests:
            self._manifests.append(manifest)
        row.update(
            status="done",
            ms=elapsed_ms,
            bytes=result.nbytes,
            js=len(js_downloaded),
            error=False,
            # A 304 means the server says this page is exactly what we already
            # hold. Carried into the row so a re-crawl can report "nothing
            # changed" rather than presenting every page as freshly scraped.
            unchanged=result.not_modified,
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

        Two collisions have to be avoided, and they are not the same one.

        *Within one page*, basenames are not unique across the CDNs it pulls
        from — ``/static/vendor.js`` and ``/lib/vendor.js`` both wanted
        ``vendor.js``, which silently overwrote earlier downloads while the
        manifest still listed every URL.

        *Across pages of one site*, the scripts share a directory: ``js_dir`` is
        per-domain, not per-page. Two pages of a site routinely load the same
        ``app.js`` path — one tagged with a cache-busting query, one not, which
        the server answers differently — and the second download overwrote the
        first. Measured: a two-page scrape left one file on disk where both
        manifests listed a script, and the first page's copy was unrecoverable.

        So a name already used in this site directory is disambiguated by
        :meth:`_reserve_js_filename`, which is the part that can see what is
        already on disk.
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

    def _reserve_js_filename(
        self, site_key: str, index: int, jurl: str, taken: set[str] | None = None
    ) -> str:
        """
        Pick a filename for a script in *site_key*'s ``js/`` directory.

        :meth:`_js_filename` decides a name from the URL alone. This adds the one
        thing it cannot see — what this site directory already holds — and keys
        the result by URL, which matters in both directions:

        * The *same* URL returns the *same* filename. A crawl visits fifty pages
          that all load ``jquery.js``; without this, fifty copies would be
          downloaded and saved under fifty hashed names.
        * A *different* URL with the same basename gets a distinct file. That is
          the clobber: ``app.js`` and ``app.js?v=2`` share a basename, are served
          differently, and the second download used to overwrite the first while
          both manifests listed a script.

        Reservation happens before the download is scheduled, so two pages
        fetched concurrently cannot be handed the same name. Nothing awaits
        between the lookup and the store, which is what makes that safe without a
        lock.
        """
        by_url = self._js_names.setdefault(site_key, {})
        existing = by_url.get(jurl)
        if existing is not None:
            if taken is not None:
                taken.add(existing)
            return existing

        candidate = self._js_filename(index, jurl, taken)
        if candidate in by_url.values():
            digest = hashlib.sha256(jurl.encode("utf-8")).hexdigest()[:8]
            stem, _, suffix = candidate.rpartition(".")
            candidate = f"{stem or candidate}.{digest}.{suffix}"
            if taken is not None:
                taken.add(candidate)
        by_url[jurl] = candidate
        return candidate

    def _skip(
        self, stats: CrawlStats, row: dict[str, Any], url: str, note: str, status: str = "skipped"
    ) -> None:
        """
        Record a page the engine chose not to fetch.

        Not counted as an error: the URL was never requested, so it must not
        consume the request budget beyond the dispatch accounting above.

        Recorded as *not attempted* rather than failed, so a later resume does
        not put the whole filtered set back at the front of the queue, where it
        would be dispatched and skipped again on every run.
        """
        row.update(status=status, error=False, note=note)
        self._emit(status, url, row)
        self._queue.mark_visited(url, success=False, attempted=False)

    def _block(self, stats: CrawlStats, row: dict[str, Any], url: str, note: str) -> None:
        stats.blocked += 1
        row.update(status="blocked", error=True, note=note)
        self._emit("blocked", url, row)
        # Attempted=false: robots and the ad list answer without a fetch, so a
        # retry would be refused identically.
        self._queue.mark_visited(url, success=False, attempted=False)
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

    #: Statuses a row never leaves. A row reaching one of these has an outcome, so
    #: it is worth a line of its own when there is no table to put it in.
    _TERMINAL_STATUSES = frozenset({"done", "error", "blocked", "skipped"})

    def _emit(self, status: str, url: str, row: dict[str, Any]) -> None:
        # `wants_lines` before the string is built, not after: formatting a line
        # costs 1.2us and a live run throws every one of them away.
        if (
            status in self._TERMINAL_STATUSES
            and self._display is not None
            and self._display.wants_lines
        ):
            self._display.line(self._result_line(status, url, row))
        if self._on_status is not None:
            with contextlib.suppress(Exception):
                self._on_status(status, url, row)

    def _result_line(self, status: str, url: str, row: dict[str, Any]) -> str:
        """
        One row of a piped run, as a single plain line.

        The alternative was a header, silence for the length of the run, and a
        table at the end — which is what a pipe and a CI log used to get, and what
        the README describes as "one clean line per result". Deliberately not a
        Table: a box drawn 3,000 times in a log file is noise, and a table's width
        is meaningless when nothing wraps it.
        """
        domain = str(row.get("domain") or url)
        nbytes = human_bytes(row.get("bytes") or 0)
        parts = [
            f"  {_STATUS_GLYPHS.get(status, '-')} {_STATUS_WORDS.get(status, status)}".rstrip()
        ]
        parts.append(f"{domain[:60]}")
        if row.get("bytes"):
            parts.append(nbytes)
        if row.get("ms"):
            parts.append(human_duration(row.get("ms")))
        if row.get("js"):
            parts.append(f"{row['js']} js")
        if status != "done" and row.get("note"):
            parts.append(f"- {str(row['note'])[:80]}")
        return "  ".join(parts)

    def _safe_hook(self, hook: Callable[..., Any], url: str, ctx: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            hook(url, ctx)
