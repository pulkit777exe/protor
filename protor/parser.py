"""
protor.parser
~~~~~~~~~~~~~
Deep page parser module. One HTML parse derives every output the pipeline needs:
metadata, plain text, Markdown, internal links, and JavaScript references. The
single noise-filtering pass lives in :mod:`protor.markdown`; this module applies
it once and renders the page's derived artefacts from the same tree.

Public API
----------
    parse_html(html, base_url, *, max_chars) → (soup, ParsedPage)
    parse_soup(soup, base_url, *, max_chars) → ParsedPage
    extract_links(html, base_url) → list[str]
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .config import MAX_TEXT_CHARS
from .markdown import clean_soup, soup_to_markdown
from .models import SiteMetadata

__all__ = ["ParsedPage", "extract_links", "parse_html", "parse_soup"]


@dataclass
class ParsedPage:
    """Everything derived from a single HTML parse of one page."""

    metadata: SiteMetadata
    text_content: str
    markdown_content: str
    links: list[str]
    js_links: list[str]


def parse_html(
    html: str,
    base_url: str,
    *,
    max_chars: int = MAX_TEXT_CHARS,
) -> tuple[BeautifulSoup, ParsedPage]:
    """
    Parse *html* once and return the tree plus all derived artefacts.

    The returned tree is the noise-filtered tree; the parsed page is derived
    from it, so callers that need the filtered tree for hooks get it for free.
    """
    soup = BeautifulSoup(html, "lxml")
    return soup, parse_soup(soup, base_url, max_chars=max_chars)


def parse_soup(
    soup: BeautifulSoup,
    base_url: str,
    *,
    max_chars: int = MAX_TEXT_CHARS,
) -> ParsedPage:
    """Derive a :class:`ParsedPage` from an already-parsed tree."""
    # Links and JS references are harvested from the raw tree first, because
    # the canonical noise-filtering pass removes scripts and navigational markup.
    links = _extract_internal_links(soup, base_url)
    js_links = _extract_js_links(soup, base_url)

    # The one canonical filtering pass. Everything below reads this filtered
    # tree, so text and Markdown stay consistent and the walk happens once.
    clean_soup(soup)

    return ParsedPage(
        metadata=_extract_metadata(soup),
        text_content=_extract_text(soup, max_chars),
        markdown_content=soup_to_markdown(soup, base_url),
        links=links,
        js_links=js_links,
    )


def extract_links(html: str, base_url: str) -> list[str]:
    """Return de-duplicated internal links from *html*, same domain as *base_url*."""
    return _extract_internal_links(BeautifulSoup(html, "lxml"), base_url)


# ── internals ────────────────────────────────────────────────────────────────


def _extract_metadata(soup: BeautifulSoup) -> SiteMetadata:
    meta = SiteMetadata()
    if soup.title and soup.title.string:
        meta.title = soup.title.string.strip()
    for tag in soup.find_all("meta"):
        name = str(tag.get("name", "") or "").lower()
        prop = str(tag.get("property", "") or "").lower()
        content = str(tag.get("content", "") or "")
        if name == "description":
            meta.description = content
        elif name == "keywords":
            meta.keywords = [k.strip() for k in content.split(",") if k.strip()]
        elif name == "author":
            meta.author = content
        elif prop.startswith("og:"):
            meta.og_tags[prop] = content
    return meta


def _extract_js_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extract JS script src URLs from a BeautifulSoup tree."""
    seen: set[str] = set()
    links: list[str] = []
    for tag in soup.find_all("script", src=True):
        full = urljoin(base_url, str(tag["src"]))
        if full not in seen and full.startswith("http"):
            seen.add(full)
            links.append(full)
    return links


def _extract_internal_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extract de-duplicated links on the same domain as *base_url*."""
    base_domain = urlparse(base_url).netloc
    seen: set[str] = set()
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        full = urljoin(base_url, str(tag["href"])).split("#")[0]
        p = urlparse(full)
        if p.netloc == base_domain and p.scheme in ("http", "https") and full not in seen:
            seen.add(full)
            links.append(full)
    return links


def _extract_text(soup: BeautifulSoup, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Extract visible text from an already-filtered tree."""
    lines = (ln.strip() for ln in soup.get_text("\n").splitlines())
    return "\n".join(ln for ln in lines if ln)[:max_chars]
