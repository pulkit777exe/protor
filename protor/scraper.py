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

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from rich import box
from rich.text import Text

from .blocklist import Blocklist
from .config import DEFAULT_CONCURRENCY, DEFAULT_TIMEOUT, RATE_LIMIT_DELAY
from .engine import CrawlEngine, StaticQueue, StaticSource
from .http_cache import HTTPCache
from .parser import extract_links
from .progress import normalise_reason, print_failure_reasons, visible_rows
from .rate_limiter import DomainRateLimiter
from .scaler import AutoScaler
from .theme import (
    ERR,
    ERR_STYLED,
    OK,
    OK_STYLED,
    SKIP,
    SPIN,
    WARN_STYLED,
    SafeTable,
    bright,
    console,
    content,
    header_rule,
    label,
    muted,
    safe,
)
from .utils import ensure_output_dir, human_bytes, human_duration

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from .extractor import ExtractionSchema
    from .models import SiteManifest

__all__ = ["extract_links", "scrape_multiple", "scrape_site_async"]


# ── live-table helpers ────────────────────────────────────────────────────────


#: Distinct failure reasons to show, so a run against 500 dead URLs stays readable.
_FAILED_STATES = ("error", "blocked", "skipped")


def _print_failure_reasons(rows: list[dict[str, Any]]) -> None:
    """
    Summarise why pages failed.

    The engine records a reason on every non-success row, but the live table has
    no room for it, so a run could only report "3 failed" — leaving the user to
    guess between DNS failure, HTTP 403, a timeout and robots.txt. The grouping
    lives in :mod:`protor.progress` because the crawler needs the same summary
    and did not have it at all.
    """
    reasons: Counter[str] = Counter()
    for row in rows:
        if row.get("status") in _FAILED_STATES:
            reasons[normalise_reason(str(row.get("note", "") or ""))] += 1
    print_failure_reasons(reasons)


#: How many rows the live batch table shows. The crawler bounds its log the same
#: way; the batch path never got the same treatment, so every row — including the
#: ones still ``waiting`` — was rebuilt and repainted on every tick.
#:
#: Measured on Rich's own render of the resulting table: 300 rows 27ms, 1,000
#: 587ms, 3,000 1,839ms. On a real 300-URL batch, rendering the progress bar was
#: 67% of the wall clock; at 600 URLs, 89%. The durable summary printed at the end
#: still carries every row.
_TABLE_VIEW = 25

#: A column of the batch table: name, minimum width, style, justification.
_Column = tuple[str, int, str, Literal["left", "right"]]


#: The batch table's columns.
#:
#: The widths are floors, not preferences, and they are what keeps the table inside
#: the window. They used to need 81 columns, and at 80 — the canonical width — rich
#: dropped the last column outright, so the JS count vanished with no ellipsis and
#: no warning. Worse, leaving the deficit to rich is worse than overflowing: at 72
#: it squeezed *every* column instead of dropping one, rendering sizes as `120.…`
#: and times as `1.…`.
_BATCH_COLUMNS: tuple[_Column, ...] = (
    ("#", 3, "grey50", "right"),
    ("Domain", 16, "white", "left"),
    ("Status", 12, "white", "left"),
    ("Size", 9, "grey74", "right"),
    ("Time", 6, "grey74", "right"),
    ("JS", 4, "grey50", "right"),
)


#: What a table costs at its narrowest: the columns' widths plus one space of
#: padding on each side of each. `box.SIMPLE` with `show_edge=False` draws no rules.
def _columns_width(columns: Sequence[_Column]) -> int:
    return sum(width for _, width, _, _ in columns) + 2 * len(columns)


#: Columns given up when the window is too narrow, cheapest information first.
#: The same trade `list_runtimes` makes with its URL column: reference
#: information yields to the counts. Never the row number or the domain, which are
#: what makes the table a table.
_SACRIFICE_ORDER = ("JS", "Size", "Time")

#: Domain never shrinks below this, however narrow the window: a truncated
#: hostname is no longer a hostname.
_DOMAIN_MIN_WIDTH = 16


def _columns_for(width: int) -> tuple[_Column, ...]:
    """The columns that fit *width*, dropping the least useful until they do."""
    columns = list(_BATCH_COLUMNS)
    for name in _SACRIFICE_ORDER:
        if _columns_width(columns) <= width:
            break
        columns = [c for c in columns if c[0] != name]
    return tuple(columns)


#: Lines the batch table spends before its rows start, measured rather than
#: counted: the top rule, the header row, and the elision row that names how many
#: earlier rows were dropped.
_TABLE_RESERVED = 4


def _build_table(
    rows: list[dict[str, Any]], width: int | None = None, height: int | None = None
) -> SafeTable:
    t = SafeTable(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold white",
        show_edge=False,
        padding=(0, 1),
    )
    columns = _columns_for(width if width is not None else console.width or 80)
    # Domain is the only elastic column, so it needs a ceiling as well as a floor.
    # Rich sizes columns to their *content* first and only then discovers the table
    # is too wide, at which point it shrinks every column: a 45-character domain in
    # a 72-column window came out as `120.…` for a size and `1…` for a time, with no
    # dropped column to explain it. The ceiling is whatever the window has left
    # after the columns that matter.
    spare = (width if width is not None else console.width or 80) - _columns_width(columns)
    domain_max = _DOMAIN_MIN_WIDTH + max(0, spare)
    for name, column_width, style, justify in columns:
        elastic = name == "Domain"
        # `no_wrap` throughout: a column that wraps inflates the table's height,
        # which is the budget just spent sizing it to the window.
        t.add_column(
            name,
            style=style,
            width=None if elastic else column_width,
            min_width=column_width if elastic else None,
            max_width=domain_max if elastic else column_width,
            justify=justify,
            no_wrap=True,
            overflow="ellipsis",
        )

    budget = visible_rows(_TABLE_RESERVED, ceiling=_TABLE_VIEW, height=height)
    shown = rows[-budget:] if len(rows) > budget else rows
    hidden = len(rows) - len(shown)
    for r in shown:
        status = r.get("status", "waiting")
        if status == "done":
            s = Text(f"  {OK} done", style="green")
        elif status == "error":
            s = Text(f"  {ERR} error", style="red")
        elif status == "waiting":
            s = Text(safe("  · waiting"), style="grey35")
        elif status == "fetching":
            s = Text(f"  {SPIN} fetch", style="yellow")
        elif status == "blocked":
            s = Text(f"  {ERR} blocked", style="red")
        elif status == "skipped":
            # Its own case: the crawler's live view already draws this one with
            # SKIP in grey, and it is a finished page, so borrowing the yellow
            # spinner below made a permanent outcome look like work in progress.
            s = Text(f"  {SKIP} skipped", style="grey50")
        elif status.startswith("js:"):
            n = status.split(":")[1]
            s = Text(f"  {SPIN} js ({n})", style="cyan")
        else:
            s = Text(f"  {SPIN} {status}", style="yellow")

        cells: dict[str, Any] = {
            "#": str(r.get("idx", "")),
            "Domain": content(r.get("domain", "")),
            "Status": s,
            "Size": human_bytes(r["bytes"]) if r.get("bytes") else "—",
            "Time": human_duration(r.get("ms")),
            "JS": str(r["js"]) if r.get("js") else "—",
        }
        t.add_row(*[cells[name] for name, *_ in columns])

    if hidden:
        note = muted(f"… {hidden} earlier {'row' if hidden == 1 else 'rows'}")
        t.add_row(*(note if name == "Domain" else "" for name, *_ in columns))
    return t


# ── site scraper ──────────────────────────────────────────────────────────────


async def scrape_site_async(
    session: Any,
    url: str,
    output_dir: str | Path,
    download_js: bool = False,
    row_state: dict[str, Any] | None = None,
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
        requested_hosts=[urlparse(url).netloc],
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


def _write_manifest_index(manifests: Iterable[SiteManifest], path: str | Path) -> None:
    """
    Write *manifests* to *path* as a JSON array, one manifest at a time.

    ``save_json`` builds the whole document as a single ``str`` before handing it
    to ``write_text``, which then encodes it a second time. For a run of real
    pages that is two copies of the entire index live at once: 2,000 manifests
    of ``text_content`` plus ``markdown_content`` measured 201 MiB of peak
    allocation for a 97 MiB file, and it grows without bound — ~50 MB of
    transient string per 1,000 pages scraped.

    Dumping manifest-by-manifest bounds the transient copy to the largest single
    manifest instead (0.3 MiB for the same 2,000). The result is byte-for-byte
    what ``save_json`` produced — same indent, same separators, same ``[]`` for
    no manifests — so existing indexes are unaffected; only the peak is.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write("[")
        written = False
        for manifest in manifests:
            # Indent each element two spaces to match json.dumps(indent=2) for a
            # list, so the file stays as readable as the one-liner produced.
            fh.write("\n  " if not written else ",\n  ")
            written = True
            dumped = json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False)
            fh.write(dumped.replace("\n", "\n  "))
        # An empty run must still leave a valid, empty array behind.
        fh.write("\n]" if written else "]")


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
    on_progress: Callable[[str, str, dict[str, Any]], None] | None = None,
    extraction_schema: ExtractionSchema | None = None,
    hooks: dict[str, list[Callable[..., Any]]] | None = None,
    block_ads: bool = False,
    auto_scale: bool = False,
    live: bool = True,
    allow_internal_redirects: bool = False,
) -> str:
    """
    Scrape *urls* concurrently and write a ``sites_index.json`` index file.

    Returns the absolute path to ``{output_dir}/sites_index.json``.
    """
    out = ensure_output_dir(output_dir)

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
        requested_hosts=[urlparse(u).netloc for u in urls],
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
        allow_internal_redirects=allow_internal_redirects,
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
        live=live,
    )
    interrupted = False
    try:
        stats = engine.run()
    except KeyboardInterrupt:
        # Everything fetched so far is on disk and in `engine.manifests`; the point
        # of catching this is to write the index and report the counts before the
        # interrupt reaches the CLI, which otherwise printed one line and nothing
        # about what had already been saved.
        interrupted = True
        stats = engine.stats
    finally:
        # A crashed run must still persist what it fetched, or every conditional
        # request from the next run starts cold.
        if cache is not None:
            cache.flush()

    manifests = engine.manifests
    ok_n = sum(1 for m in manifests if m.success)
    error_count = stats.errors + stats.blocked
    total = stats.bytes_total
    avg_ms = round(sum(m.elapsed_ms for m in manifests) / max(ok_n, 1)) if ok_n else 0

    console.print()
    headline = (
        f"{WARN_STYLED} stopped at {bright(str(ok_n))} scraped"
        if interrupted
        else f"{OK_STYLED} {bright(str(ok_n))} scraped"
    )
    console.print(
        f"  {headline}  "
        + (f"{ERR_STYLED} {bright(str(error_count))} failed  " if error_count else "")
        + f"{muted(human_bytes(total) + ' total')}  {muted(f'avg {avg_ms}ms')}"
    )
    _print_failure_reasons(rows)

    index = out / "sites_index.json"
    _write_manifest_index(manifests, index)
    console.print(f"  {label('index')} {muted(str(index))}")
    console.print()

    if interrupted:
        raise KeyboardInterrupt from None

    return str(index)
