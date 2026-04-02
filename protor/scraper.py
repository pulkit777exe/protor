"""
protor.scraper
~~~~~~~~~~~~~~
Async HTML + JS scraper built on aiohttp.

Public API
----------
    scrape_multiple(urls, output_dir, *, download_js, timeout, concurrency)
      → str   path to the generated sites_index.json

    scrape_site_async(session, url, output_dir, download_js, row_state)
      → SiteManifest | None   (None on failure)

Internal helpers are prefixed with _ and are not part of the public API.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from rich import box
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .exceptions import FetchError
from .models import SiteManifest, SiteMetadata
from .theme import OK, ERR, SPIN, console, header_rule, label, bright, muted
from .utils import safe_filename, save_json, timestamp, human_bytes

__all__ = ["scrape_multiple", "scrape_site_async", "extract_links"]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

_DEFAULT_CONCURRENCY = 6
_DEFAULT_TIMEOUT     = 30
_MAX_JS_FILES        = 15
_MAX_TEXT_CHARS      = 10_000
_MAX_DATA_CHARS      = 8_000


# ── HTML parsing helpers ──────────────────────────────────────────────────────

def _extract_metadata(soup: BeautifulSoup) -> SiteMetadata:
    meta = SiteMetadata()
    if soup.title and soup.title.string:
        meta.title = soup.title.string.strip()
    for tag in soup.find_all("meta"):
        name    = tag.get("name", "").lower()
        prop    = tag.get("property", "").lower()
        content = tag.get("content", "")
        if name == "description":
            meta.description = content
        elif name == "keywords":
            meta.keywords = [k.strip() for k in content.split(",") if k.strip()]
        elif name == "author":
            meta.author = content
        elif prop.startswith("og:"):
            meta.og_tags[prop] = content
    return meta


def _extract_js_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    links: list[str] = []
    for tag in soup.find_all("script", src=True):
        full = urljoin(base_url, tag["src"])
        if full not in seen and full.startswith("http"):
            seen.add(full)
            links.append(full)
    return links


def extract_links(html: str, base_url: str) -> list[str]:
    """Return de-duplicated internal links from *html*, same domain as *base_url*."""
    soup = BeautifulSoup(html, "lxml")
    base_domain = urlparse(base_url).netloc
    seen: set[str] = set()
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        full = urljoin(base_url, tag["href"]).split("#")[0]
        p = urlparse(full)
        if p.netloc == base_domain and p.scheme in ("http", "https") and full not in seen:
            seen.add(full)
            links.append(full)
    return links


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    lines = (ln.strip() for ln in soup.get_text("\n").splitlines())
    return "\n".join(ln for ln in lines if ln)[:_MAX_TEXT_CHARS]


# ── async fetch primitives ────────────────────────────────────────────────────

async def _fetch(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int = _DEFAULT_TIMEOUT,
) -> tuple[str, int]:
    """
    Fetch *url* and return (text, bytes_received).
    Raises FetchError on HTTP >= 400 or connection problems.
    """
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status >= 400:
                raise FetchError(url, f"HTTP {r.status}")
            data = await r.read()
            return data.decode("utf-8", errors="replace"), len(data)
    except FetchError:
        raise
    except asyncio.TimeoutError as exc:
        raise FetchError(url, "timeout") from exc
    except aiohttp.ClientError as exc:
        raise FetchError(url, str(exc)) from exc


async def _download_file(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status == 200:
                dest.write_bytes(await r.read())
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


# ── live-table helpers ────────────────────────────────────────────────────────

def _build_table(rows: list[dict]) -> Table:
    t = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold white",
        show_edge=False,
        padding=(0, 1),
    )
    t.add_column("#",      style="grey50", width=3,  justify="right")
    t.add_column("Domain", style="white",  min_width=32)
    t.add_column("Status",                 width=14)
    t.add_column("Size",   style="grey74", width=9,  justify="right")
    t.add_column("Time",   style="grey74", width=7,  justify="right")
    t.add_column("JS",     style="grey50", width=4,  justify="right")

    for r in rows:
        status = r.get("status", "waiting")
        if status == "done":
            s = Text(f"  {OK} done",    style="green")
        elif status == "error":
            s = Text(f"  {ERR} error",  style="red")
        elif status == "waiting":
            s = Text("  · waiting",     style="grey35")
        elif status == "fetching":
            s = Text(f"  {SPIN} fetch", style="yellow")
        elif status.startswith("js:"):
            n = status.split(":")[1]
            s = Text(f"  {SPIN} js ({n})", style="cyan")
        else:
            s = Text(f"  {SPIN} {status}", style="yellow")

        t.add_row(
            str(r["idx"]),
            r["domain"],
            s,
            human_bytes(r["bytes"]) if r.get("bytes") else "—",
            f"{r['ms']}ms"          if r.get("ms")    else "—",
            str(r["js"])            if r.get("js")    else "—",
        )
    return t


# ── site scraper ──────────────────────────────────────────────────────────────

async def scrape_site_async(
    session: aiohttp.ClientSession,
    url: str,
    output_dir: Path,
    download_js: bool,
    row_state: dict,
) -> SiteManifest | None:
    """
    Scrape a single *url*.

    Mutates *row_state* in-place so the Live table can show progress.
    Returns None on failure (errors are captured in row_state).
    """
    parsed   = urlparse(url)
    site_dir = output_dir / safe_filename(parsed.netloc)
    site_dir.mkdir(parents=True, exist_ok=True)

    row_state["status"] = "fetching"
    t0 = time.perf_counter()

    try:
        html, nbytes = await _fetch(session, url)
    except FetchError as exc:
        row_state["status"] = "error"
        row_state["note"]   = str(exc)
        return None

    elapsed = time.perf_counter() - t0

    (site_dir / "index.html").write_text(html, encoding="utf-8")

    soup     = BeautifulSoup(html, "lxml")
    metadata = _extract_metadata(soup)
    text     = _extract_text(html)

    js_downloaded: list[str] = []
    if download_js:
        js_links = _extract_js_links(html, url)[:_MAX_JS_FILES]
        if js_links:
            row_state["status"] = f"js:{len(js_links)}"
            js_dir = site_dir / "js"
            tasks  = [
                _download_file(
                    session,
                    jurl,
                    js_dir / (safe_filename(Path(urlparse(jurl).path).name) or f"s{i}.js"),
                )
                for i, jurl in enumerate(js_links)
            ]
            results       = await asyncio.gather(*tasks)
            js_downloaded = [u for u, ok in zip(js_links, results) if ok]

    manifest = SiteManifest(
        url            = url,
        domain         = parsed.netloc,
        html_file      = str(site_dir / "index.html"),
        metadata       = metadata,
        text_content   = text,
        js_files       = js_downloaded,
        js_count       = len(js_downloaded),
        bytes_received = nbytes,
        elapsed_ms     = round(elapsed * 1000),
        timestamp      = timestamp(),
        success        = True,
    )
    save_json(manifest.to_dict(), site_dir / "manifest.json")

    row_state.update(status="done", ms=manifest.elapsed_ms, bytes=nbytes, js=len(js_downloaded))
    return manifest


# ── orchestrator ──────────────────────────────────────────────────────────────

async def _run_all(
    urls: list[str],
    output_dir: Path,
    download_js: bool,
    timeout: int,
    concurrency: int,
) -> list[SiteManifest]:
    rows = [
        {"idx": i + 1, "domain": urlparse(u).netloc or u,
         "status": "waiting", "bytes": None, "ms": None, "js": None}
        for i, u in enumerate(urls)
    ]

    manifests: list[SiteManifest] = []
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(session: aiohttp.ClientSession, url: str, row: dict) -> SiteManifest | None:
        async with sem:
            return await scrape_site_async(session, url, output_dir, download_js, row)

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(headers=_HEADERS, connector=connector) as session:
        with Live(console=console, refresh_per_second=10) as live:
            task_group = asyncio.gather(
                *[_bounded(session, u, rows[i]) for i, u in enumerate(urls)],
                return_exceptions=True,
            )
            while not task_group.done():
                live.update(_build_table(rows))
                await asyncio.sleep(0.08)
            live.update(_build_table(rows))

        results = await task_group

    manifests = [m for m in results if isinstance(m, SiteManifest)]
    return manifests


def scrape_multiple(
    urls: list[str],
    output_dir: str | Path = "data",
    *,
    download_js: bool = True,
    timeout: int = _DEFAULT_TIMEOUT,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> str:
    """
    Scrape *urls* concurrently and write a ``sites_index.json`` index file.

    Parameters
    ----------
    urls:
        List of URLs to scrape.
    output_dir:
        Root directory for scraped artefacts (created if absent).
    download_js:
        Whether to download linked ``<script src>`` files.
    timeout:
        Per-request timeout in seconds.
    concurrency:
        Maximum simultaneous requests.

    Returns
    -------
    str
        Absolute path to ``{output_dir}/sites_index.json``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.print()
    console.print(header_rule("Protor — Scraper"))
    console.print(
        f"  {label('targets')} {bright(str(len(urls)))}   "
        f"{label('concurrency')} {bright(str(concurrency))}   "
        f"{label('output')} {muted(str(out))}"
    )
    console.print()

    manifests = asyncio.run(_run_all(urls, out, download_js, timeout, concurrency))

    ok_n   = sum(1 for m in manifests if m.success)
    err_n  = len(urls) - ok_n
    total  = sum(m.bytes_received for m in manifests)
    avg_ms = round(sum(m.elapsed_ms for m in manifests) / max(ok_n, 1)) if ok_n else 0

    console.print()
    console.print(
        f"  ✓ {bright(str(ok_n))} scraped  "
        + (f"✗ {bright(str(err_n))} failed  " if err_n else "")
        + f"{muted(human_bytes(total) + ' total')}  {muted(f'avg {avg_ms}ms')}"
    )

    index = out / "sites_index.json"
    save_json([m.to_dict() for m in manifests], index)
    console.print(f"  {label('index')} {muted(str(index))}")
    console.print()

    return str(index)