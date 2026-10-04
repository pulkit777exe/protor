"""
protor.sitemap
~~~~~~~~~~~~~~
Reading sitemaps, so a crawl is not limited to what the homepage links to.

A link-walk only ever sees what a page happens to link. On a documentation site
that is the sidebar; on a shop it is the top nav. A sitemap is the site telling
you what exists, which is the difference between crawling forty pages of a
three-thousand-page site and crawling all three thousand — and the pages found
this way are the ones a link-walk structurally cannot reach.

Three shapes are handled, because all three are common in the wild:

* **urlset** — ``<url><loc>…</loc></url>``, the plain case.
* **sitemapindex** — ``<sitemap><loc>…</loc></sitemap>``, an index of more
  sitemaps. Large sites split by section or by date, and a 50,000-URL sitemap
  index is unremarkable.
* **gzip** — ``sitemap.xml.gz``, which is what most large sites serve, and which
  a ``Content-Type`` of ``application/gzip`` does not always announce.

Everything is best-effort: a sitemap that 404s, is malformed, or is not XML at
all yields nothing rather than failing a crawl. Discovering pages is an
optimisation, never a precondition.
"""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

import aiohttp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["MAX_SITEMAP_DEPTH", "SitemapEntry", "default_sitemap", "iter_sitemap_urls"]

#: How many levels of ``<sitemapindex>`` to follow. Real sites nest one deep; a
#: cycle between two indexes would otherwise be followed forever, since each level
#: is a fresh URL and nothing in the document says "you have been here".
MAX_SITEMAP_DEPTH = 3

#: Sitemaps are read whole rather than streamed. ``sitemap.xml`` files run to a
#: few MB and gzip to a fraction of that; parsing incrementally would mean
#: hand-rolling a pull parser to save memory nobody is short of, at the cost of
#: handling a malformed tail ourselves instead of letting ElementTree refuse it.
_MAX_BYTES = 64 * 1024 * 1024

#: A page's own ``<lastmod>``, kept because it is the only cheap signal that a
#: page changed — useful for an incremental re-crawl, which otherwise has to
#: re-fetch everything to discover that nothing did.
SitemapEntry = tuple[str, str]


def default_sitemap(base_url: str) -> str:
    """The conventional sitemap location for *base_url*'s origin."""
    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}/sitemap.xml"


def _localname(tag: Any) -> str:
    """An element's tag without its namespace.

    Every real sitemap is namespaced (``http://www.sitemaps.org/schemas/sitemap/0.9``)
    and the prefix varies by site, so matching on the qualified name would miss
    documents that are otherwise perfectly valid.
    """
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _decode(body: bytes) -> str:
    """
    Turn a sitemap response body into text.

    Unconditionally gunzips on the magic bytes rather than trusting the
    ``Content-Type``: ``.xml.gz`` files are routinely served as
    ``application/xml`` or ``text/plain``, and a header check would hand the
    compressed bytes to the XML parser.
    """
    if body[:2] == b"\x1f\x8b":
        try:
            body = gzip.decompress(body)
        except (OSError, EOFError, gzip.BadGzipFile):
            return ""
    return body.decode("utf-8", errors="replace")


async def _get(session: aiohttp.ClientSession, url: str, timeout: float) -> bytes | None:
    """Fetch a sitemap, returning its body or None."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                return None
            return await resp.read()
    except (aiohttp.ClientError, TimeoutError, OSError):
        return None


async def iter_sitemap_urls(
    session: aiohttp.ClientSession,
    urls: list[str],
    *,
    base_url: str = "",
    timeout: float = 20.0,
    max_depth: int = MAX_SITEMAP_DEPTH,
    limit: int = 0,
) -> AsyncIterator[SitemapEntry]:
    """
    Yield ``(url, lastmod)`` for every page named by the sitemaps in *urls*.

    Parameters
    ----------
    urls:
        Sitemap locations to read. Relative entries are resolved against
        *base_url*, which a malformed robots.txt does sometimes produce.
    base_url:
        Used to resolve relative ``<loc>`` values and relative sitemap entries.
    max_depth:
        How many ``<sitemapindex>`` levels to follow.
    limit:
        Stop after yielding this many URLs. ``0`` means no limit. A crawl always
        has its own ``--max-pages`` ceiling, but this stops the generator early
        rather than parsing 50,000 URLs to hand back 10.

    Notes
    -----
    Follows the index depth-capped rather than by tracking visited URLs: two
    indexes that name each other are pathological and rare, while the
    alternative is an unbounded set that has to be threaded through every layer.
    A site that does it gets ``max_depth`` extra requests and then stops.
    """
    seen_sitemaps: set[str] = set()
    queue: list[tuple[str, int]] = [(urljoin(base_url, u) if base_url else u, 0) for u in urls]
    emitted = 0

    while queue:
        sitemap_url, depth = queue.pop(0)
        if depth > max_depth or sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)

        fetched = await _get(session, sitemap_url, timeout)
        if fetched is None:
            continue
        text = _decode(fetched)
        if not text.strip():
            continue

        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError:
            continue

        for element in root.iter():
            name = _localname(element.tag)
            if name not in ("url", "sitemap"):
                continue
            loc = lastmod = ""
            for child in element:
                tag = _localname(child.tag)
                if tag == "loc" and child.text:
                    loc = child.text.strip()
                elif tag == "lastmod" and child.text:
                    lastmod = child.text.strip()
            if not loc:
                continue

            if name == "sitemap":
                if depth < max_depth:
                    queue.append((urljoin(sitemap_url, loc), depth + 1))
                continue

            yield (loc if loc.startswith("http") else urljoin(sitemap_url, loc), lastmod)
            emitted += 1
            if limit and emitted >= limit:
                return


async def discover_sitemap_urls(
    session: aiohttp.ClientSession,
    base_url: str,
    *,
    robots: Any = None,
    timeout: float = 20.0,
    limit: int = 0,
    probe_default: bool = True,
) -> list[SitemapEntry]:
    """
    Find a site's sitemaps and return the pages they name.

    Reads the ``Sitemap:`` lines from robots.txt when a :mod:`protor.robots`
    cache is supplied — the crawl has usually fetched it already — and otherwise
    falls back to the conventional ``/sitemap.xml``. A site with neither simply
    yields nothing.
    """
    urls: list[str] = []
    if robots is not None:
        urls = await robots.sitemaps(base_url, session)
    if not urls and probe_default:
        fallback = default_sitemap(base_url)
        if fallback:
            urls = [fallback]
    if not urls:
        return []

    found: list[SitemapEntry] = []
    async for entry in iter_sitemap_urls(
        session, urls, base_url=base_url, timeout=timeout, limit=limit
    ):
        found.append(entry)
    return found
