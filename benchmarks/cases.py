"""
Benchmark cases for protor's hot paths.

Every case returns a callable plus a size knob, so the same workload can be run
at two or more scales. That is the whole point: a single-scale timing tells you
a number, but the *ratio* between scales tells you whether the algorithm is
linear. A change that quietly reintroduces a per-page full-document scan looks
fine at 200 pages and terrible at 2,000, and a flat threshold cannot tell those
apart.

Cases cover the paths that run once per page or per URL during a crawl, which
is where a scraper's wall-clock time actually goes:

- ``parse_html`` — noise filtering and Markdown rendering
- ``clean_soup`` — the canonical page-filtering pass
- ``_extract_text`` — visible-text extraction under a character budget
- ``prepare_context`` — flattening a batch of sites for the LLM prompt
- ``HTTPCache.put`` — cache writes
- ``canonicalize_url`` — queue de-duplication
- ``is_url_blocked`` — per-URL ad/tracker filtering
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

__all__ = ["BENCH_CASES", "BenchCase", "artificial_page", "site_batch"]
if TYPE_CHECKING:
    from collections.abc import Callable


_WORD_POOL = [
    "scraping",
    "throughput",
    "latency",
    "pipeline",
    "ingestion",
    "throughput",
    "render",
    "parse",
    "cache",
    "queue",
    "budget",
    "throttle",
    "concurrency",
    "schema",
    "manifest",
    "payload",
    "validator",
    "revalidate",
    "checksum",
    "artifact",
    "regression",
    "coverage",
    "benchmark",
    "profile",
    "allocation",
    "overhead",
]


def artificial_page(blocks: int, *, nested: bool = False, noise: bool = True) -> str:
    """
    Build a page of roughly *blocks* content sections.

    Shaped like a real scraped page rather than a flat list: nested containers,
    headings, prose with links and emphasis, lists, and (optionally) the noise
    patterns ``clean_soup`` has to strip. Deterministic for a given *blocks*, so
    a before/after comparison measures code rather than input variation.
    """
    rng = random.Random(blocks * 31 + (7 if nested else 0) + (13 if noise else 0))
    parts = [
        "<!DOCTYPE html><html><head><title>Benchmark page</title>"
        '<meta name="description" content="A synthetic page for measuring.">'
        "</head><body>"
    ]

    if noise:
        parts.append(
            "<nav><ul>"
            + "".join(f'<li class="nav-item"><a href="/n/{i}">Nav {i}</a></li>' for i in range(8))
            + "</ul></nav>"
            '<aside><div class="widget promo">Sponsored</div></aside>'
            "<script>window.tracker=1;</script>"
        )

    for i in range(blocks):
        prose = " ".join(rng.choice(_WORD_POOL) for _ in range(rng.randint(18, 34)))
        parts.append(f'<section id="s{i}" class="content-block">')
        parts.append(f"<h2>Section {i}</h2>")
        parts.append(
            f'<p>{prose.capitalize()} with <a href="/a/{i}">a link</a>, '
            f"<strong>emphasis</strong> and <em>stress</em>.</p>"
        )
        parts.append(
            "<ul>" + "".join(f"<li>{rng.choice(_WORD_POOL)}</li>" for _ in range(4)) + "</ul>"
        )
        if i % 4 == 0:
            parts.append(
                "<blockquote><p>A quoted passage that a scraper should keep.</p></blockquote>"
            )
        if i % 7 == 0:
            parts.append(
                "<table><tr><th>Key</th><th>Value</th></tr>"
                + "".join(
                    f"<tr><td>{rng.choice(_WORD_POOL)}</td><td>{i}</td></tr>" for _ in range(3)
                )
                + "</table>"
            )
        if nested:
            parts.append("<div>" * 6 + f'<div class="deep">Nested detail {i}.' + "</div>" * 6)
        parts.append("</section>")

    parts.append(
        '<footer class="site-footer"><p>&copy; Benchmark</p>'
        '<script src="/analytics.js"></script></footer>'
    )
    parts.append("</body></html>")
    return "".join(parts)


def site_batch(count: int) -> list[dict[Any, Any]]:
    """Build a batch of scraped-site records, as ``sites_index.json`` would hold."""
    rng = random.Random(count)
    return [
        {
            "domain": f"site{i}.example.com",
            "url": f"https://site{i}.example.com/page",
            "js_count": rng.randint(0, 40),
            "metadata": {
                "title": f"Page {i} — a reasonably long title for a scraped page",
                "description": "A description field of typical length for a real page.",
                "author": "Benchmark",
                "keywords": ["alpha", "beta"],
                "og_tags": {},
            },
            "text_content": " ".join(rng.choice(_WORD_POOL) for _ in range(rng.randint(200, 400))),
        }
        for i in range(count)
    ]


@dataclass(frozen=True)
class BenchCase:
    """One measurable workload."""

    name: str
    #: Run the workload and return an opaque result (a dict, count, string...).
    run: Callable[[int], Any]
    #: The small-scale size, and the large-scale size, to compare.
    scales: tuple[int, int]
    #: Whether a per-item scaling ratio means anything for this case. Set False
    #: where the returned item count does not grow with the input, so the
    #: runner reports a timing instead of a misleading ratio.
    scaling_meaningful: bool = True

    @property
    def growth(self) -> float:
        """How much the workload size grows between the two scales."""
        small, large = self.scales
        return large / small


#: Stable byte size of one generated page, measured once. Using a constant
#: rather than the live page length matters: the size grows with the block
#: count, so dividing by it would divide away exactly the growth being measured.
#: It is filled in on first use by :func:`_page_size`.
_PAGE_BYTES = 0


def _page_size() -> int:
    """
    Byte size of one 200-block page, measured once and used only for reporting.

    The size of a generated page grows with the block count, so dividing a cost
    by the current page's length would divide away exactly the growth being
    measured. Page cases therefore count *blocks* -- a fixed, atomic unit of
    generated content -- and this is used to state the bytes-per-block figure
    so a reader can convert if they want to.
    """
    global _PAGE_BYTES
    if not _PAGE_BYTES:
        _PAGE_BYTES = len(artificial_page(200).encode("utf-8"))
    return _PAGE_BYTES


def _page_units(size: int) -> int:
    """Work item count for page cases: content blocks processed."""
    return size


def _parse_page(size: int) -> Any:
    """One page parsed end to end; items are bytes of input processed."""
    from protor.parser import parse_html

    _, page = parse_html(artificial_page(size), "https://example.com/")
    assert page.markdown_content, "parse produced no output"
    return _page_units(size)


def _clean_soup(size: int) -> Any:
    from bs4 import BeautifulSoup

    from protor.markdown import clean_soup

    soup = BeautifulSoup(artificial_page(size), "lxml")
    clean_soup(soup)
    assert soup.get_text()
    return _page_units(size)


def _extract_text(size: int) -> Any:
    from bs4 import BeautifulSoup

    from protor.parser import _extract_text

    soup = BeautifulSoup(artificial_page(size), "lxml")
    assert _extract_text(soup, max_chars=10_000)
    return _page_units(size)


def _prepare_context(size: int) -> Any:
    from protor.analyzer import _prepare_context

    data = site_batch(size)
    return len(_prepare_context(data))  # type: ignore[arg-type]


#: Bytes in a 200-block reference page. Reported so the per-block figures can
#: be converted into a bytes-per-second figure; not used as a divisor, because
#: a page's byte length grows with its block count.
PAGE_BYTES = _page_size()

#: Content blocks per reference page.
BLOCKS_PER_PAGE = 200


def _calibration(size: int) -> Any:
    """
    A fixed amount of pure-Python work, used to measure the machine.

    This exists because an absolute timing recorded on one machine says nothing
    about another: the first CI run of this suite reported every case 1.6-1.7x
    slower than the baseline recorded locally, purely because GitHub's runner is
    slower than the developer's. A case's cost is therefore divided by this
    one's cost *from the same run*, which cancels the machine out and leaves the
    algorithmic difference.
    """
    total = 0
    for i in range(size):
        acc = [j * j for j in range(200)]
        total += sum(acc) + len(str(i))
    assert total > 0
    return size


def _cache_puts(size: int) -> Any:
    import tempfile
    from pathlib import Path

    from protor.http_cache import CacheEntry, HTTPCache

    body = artificial_page(2)
    with tempfile.TemporaryDirectory() as tmp:
        cache = HTTPCache(cache_dir=Path(tmp))
        for i in range(size):
            cache.put(f"https://s{i}.example.com/", CacheEntry(body=body, status=200))
        cache.flush()
        written = len(list((Path(tmp) / "bodies").glob("*.body")))
    assert written == size, f"wrote {written} of {size}"
    return size


def _canonicalize(size: int) -> Any:
    from protor.utils import canonicalize_url

    return sum(
        1
        for i in range(size)
        if canonicalize_url(f"https://example.com/a/b/c/page{i}.html?utm_source=x&i={i}#frag")
    )


def _is_url_blocked(size: int) -> Any:
    from protor.blocklist import Blocklist

    bl = Blocklist(block_ads=True)
    for i in range(size):
        bl.is_url_blocked(f"https://site{i}.example.com/page")
    return size


#: The machine-speed reference case. Measured in every run and used as the
#: divisor for every other case's normalised figure.
CALIBRATION_CASE = BenchCase("calibration", _calibration, (2_000, 8_000))

#: Every case the benchmark runner can execute. The calibration case runs too,
#: but is excluded from scaling checks since it is linear by construction.
#:
#: Page scales are chosen to stay *below* the renderer's character budgets, and
#: a test asserts that. Getting it wrong is not subtle-but-harmless: with both
#: scales saturated, each case stops at the cap and measures the same fixed work
#: regardless of page size, so the ratio is flat no matter what the code does.
#: A quadratic re-scan injected into `_process_element` passed the gate unnoticed
#: for exactly that reason.
BENCH_CASES: tuple[BenchCase, ...] = (
    CALIBRATION_CASE,
    BenchCase("parse_html", _parse_page, (20, 80)),
    BenchCase("clean_soup", _clean_soup, (200, 800)),
    BenchCase("extract_text", _extract_text, (10, 30)),
    # The context is capped by a character budget, so its item count does not
    # grow with the batch; a per-item ratio here would be noise.
    BenchCase("prepare_context", _prepare_context, (100, 400), scaling_meaningful=False),
    # A fixed count of puts; per-item cost is already reported by the table.
    BenchCase("http_cache_put", _cache_puts, (100, 400), scaling_meaningful=False),
    BenchCase("canonicalize_url", _canonicalize, (20_000, 80_000)),
    BenchCase("is_url_blocked", _is_url_blocked, (20_000, 80_000)),
)
