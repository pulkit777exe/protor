"""
protor.scraper
~~~~~~~~~~~~~~
Thin orchestrator on top of the fetch, parser, and crawl engine modules.
Scraping a single site or a batch of URLs both route through
:class:`protor.engine.CrawlEngine`; this module only builds inputs and renders
the terminal output.

Public API
----------
    scrape_site_async(session, url, output_dir, download_js, row_state, cache) → SiteManifest | None
    scrape_multiple(urls, output_dir, *, ...) → path to sites_index.json
    extract_links(html, base_url) → list[str]   (re-exported from parser)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from rich import box
from rich.table import Table
from rich.text import Text

from .blocklist import Blocklist
from .config import DEFAULT_CONCURRENCY, DEFAULT_TIMEOUT, RATE_LIMIT_DELAY
from .engine import CrawlEngine, StaticQueue, StaticSource
from .http_cache import HTTPCache
from .parser import extract_links
from .rate_limiter import DomainRateLimiter
from .scaler import AutoScaler
from .theme import ERR, OK, SPIN, bright, console, header_rule, label, muted
from .utils import human_bytes, save_json

if TYPE_CHECKING:
    from collections.abc import Callable

    from .extractor import ExtractionSchema
    from .models import SiteManifest

__all__ = ["extract_links", "scrape_multiple", "scrape_site_async"]


# ── live-table helpers ────────────────────────────────────────────────────────


def _build_table(rows: list[dict]) -> Table:
    t = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold white",
        show_edge=False,
        padding=(0, 1),
    )
    t.add_column("#", style="grey50", width=3, justify="right")
    t.add_column("Domain", style="white", min_width=32)
    t.add_column("Status", width=14)
    t.add_column("Size", style="grey74", width=9, justify="right")
    t.add_column("Time", style="grey74", width=7, justify="right")
    t.add_column("JS", style="grey50", width=4, justify="right")

    for r in rows:
        status = r.get("status", "waiting")
        if status == "done":
            s = Text(f"  {OK} done", style="green")
        elif status == "error":
            s = Text(f"  {ERR} error", style="red")
        elif status == "waiting":
            s = Text("  · waiting", style="grey35")
        elif status == "fetching":
            s = Text(f"  {SPIN} fetch", style="yellow")
        elif status == "blocked":
            s = Text(f"  {ERR} blocked", style="red")
        elif status.startswith("js:"):
            n = status.split(":")[1]
            s = Text(f"  {SPIN} js ({n})", style="cyan")
        else:
            s = Text(f"  {SPIN} {status}", style="yellow")

        t.add_row(
            str(r.get("idx", "")),
            r.get("domain", ""),
            s,
            human_bytes(r["bytes"]) if r.get("bytes") else "—",
            f"{r['ms']}ms" if r.get("ms") else "—",
            str(r["js"]) if r.get("js") else "—",
        )
    return t


# ── site scraper ──────────────────────────────────────────────────────────────


async def scrape_site_async(
    session: Any,
    url: str,
    output_dir: str | Path,
    download_js: bool = False,
    row_state: dict | None = None,
    cache: HTTPCache | None = None,
    *,
    extraction_schema: ExtractionSchema | None = None,
    hooks: dict[str, list[Callable[..., Any]]] | None = None,
    blocklist: Blocklist | None = None,
    check_robots: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> SiteManifest | None:
    """
    Scrape a single *url* using the shared crawl engine.

    A thin wrapper over :class:`~protor.engine.CrawlEngine` with a one-URL
    queue, so single-site scraping and batch/crawl runs share one
    implementation. Previously this function carried its own copy of the
    pipeline, which had already drifted from the engine: it ignored
    ``block_ads``, skipped the robots check, wrote a hard-coded
    ``manifest.json`` that later pages overwrote, and dropped *timeout*.

    Mutates *row_state* in-place so a Live table can show progress.
    Returns None on failure (errors are recorded in row_state).

    If *session* is given it is reused and left open; otherwise the engine
    opens and closes its own.
    """
    row = {} if row_state is None else row_state
    row["status"] = "fetching"

    engine = CrawlEngine(
        queue=StaticQueue([url]),
        link_source=StaticSource(),
        output_dir=output_dir,
        max_targets=1,
        timeout=timeout,
        download_js=download_js,
        cache=cache,
        hooks=hooks,
        extraction_schema=extraction_schema,
        blocklist=blocklist,
        rate_limiter=DomainRateLimiter(delay=RATE_LIMIT_DELAY),
        check_robots=check_robots,
        session=session,
        # Hand the engine our dict so its in-place row updates land in
        # *row_state*, which is what a Live table renders from.
        rows=[row],
    )
    await engine.arun()

    manifests = engine.manifests
    return manifests[0] if manifests else None


# ── orchestrator ──────────────────────────────────────────────────────────────


def scrape_multiple(
    urls: list[str],
    output_dir: str | Path = "data",
    *,
    download_js: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    concurrency: int = DEFAULT_CONCURRENCY,
    cache: HTTPCache | None = None,
    use_cache: bool = False,
    headers: dict[str, str] | None = None,
    on_progress: Callable[[str, str, dict], None] | None = None,
    extraction_schema: ExtractionSchema | None = None,
    hooks: dict[str, list[Callable[..., Any]]] | None = None,
    block_ads: bool = False,
    auto_scale: bool = False,
) -> str:
    """
    Scrape *urls* concurrently and write a ``sites_index.json`` index file.

    Returns the absolute path to ``{output_dir}/sites_index.json``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Caching was previously always on, so every batch run paid to build and
    # populate a cache the user never asked for. Opt in via --cache.
    if cache is None and not use_cache:
        cache = None
    elif cache is None:
        cache = HTTPCache()

    console.print()
    console.print(header_rule("Protor — Scraper"))
    features = []
    if block_ads:
        features.append("ad-blocking")
    if auto_scale:
        features.append("auto-scaling")
    if cache is not None:
        features.append("http-cache")
    if extraction_schema:
        features.append(f"extract:{extraction_schema.name}")
    if hooks:
        features.append(f"hooks:{'+'.join(hooks.keys())}")

    console.print(
        f"  {label('targets')} {bright(str(len(urls)))}   "
        f"{label('concurrency')} {bright(str(concurrency))}   "
        f"{label('output')} {muted(str(out))}"
    )
    if features:
        console.print(f"  {label('features')} {bright(', '.join(features))}")
    console.print()

    rows = [
        {
            "idx": i + 1,
            "domain": urlparse(u).netloc or u,
            "status": "waiting",
            "bytes": None,
            "ms": None,
            "js": None,
            "error": False,
        }
        for i, u in enumerate(urls)
    ]

    engine = CrawlEngine(
        queue=StaticQueue(urls),
        link_source=StaticSource(),
        output_dir=out,
        max_targets=len(urls),
        concurrency=concurrency,
        rows=rows,
        timeout=timeout,
        download_js=download_js,
        cache=cache,
        headers=headers,
        hooks=hooks,
        extraction_schema=extraction_schema,
        blocklist=Blocklist(block_ads=True) if block_ads else None,
        rate_limiter=DomainRateLimiter(delay=RATE_LIMIT_DELAY),
        auto_scaler=(
            AutoScaler(
                initial=concurrency,
                min_c=2,
                max_c=min(concurrency * 3, 20),
            )
            if auto_scale
            else None
        ),
        check_robots=True,
        on_status=on_progress,
        live_render=lambda: _build_table(rows),
    )
    if cache is not None:
        stats = engine.run()
        cache.flush()
    else:
        stats = engine.run()

    manifests = engine.manifests
    ok_n = sum(1 for m in manifests if m.success)
    error_count = stats.errors + stats.blocked
    total = stats.bytes_total
    avg_ms = round(sum(m.elapsed_ms for m in manifests) / max(ok_n, 1)) if ok_n else 0

    console.print()
    console.print(
        f"  {OK} {bright(str(ok_n))} scraped  "
        + (f"{ERR} {bright(str(error_count))} failed  " if error_count else "")
        + f"{muted(human_bytes(total) + ' total')}  {muted(f'avg {avg_ms}ms')}"
    )

    index = out / "sites_index.json"
    save_json([m.to_dict() for m in manifests], index)
    console.print(f"  {label('index')} {muted(str(index))}")
    console.print()

    return str(index)
