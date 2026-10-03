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

from bs4 import BeautifulSoup, Tag

from .config import MAX_MARKDOWN_CHARS, MAX_TEXT_CHARS
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
    max_markdown_chars: int = MAX_MARKDOWN_CHARS,
) -> ParsedPage:
    """Derive a :class:`ParsedPage` from an already-parsed tree."""
    # Links, JS references and metadata come off the raw tree in a single walk,
    # because the canonical noise-filtering pass below removes the scripts and
    # navigational markup they are read from.
    harvested = _Harvest().harvest(soup, base_url)

    # The one canonical filtering pass. Everything below reads this filtered
    # tree, so text and Markdown stay consistent and the walk happens once.
    clean_soup(soup)

    return ParsedPage(
        metadata=harvested.metadata,
        text_content=_extract_text(soup, max_chars),
        markdown_content=soup_to_markdown(soup, base_url, max_chars=max_markdown_chars),
        links=harvested.links,
        js_links=harvested.js_links,
    )


def extract_links(html: str, base_url: str) -> list[str]:
    """Return de-duplicated internal links from *html*, same domain as *base_url*."""
    return _extract_internal_links(BeautifulSoup(html, "lxml"), base_url)


# ── internals ────────────────────────────────────────────────────────────────


class _Harvest:
    """
    The link, script and metadata references one pass over a page can find.

    Reading these used to be four separate searches over the document, and each
    ``find_all`` builds and runs its own strainer, which cost more than the tags
    it selected: instrumenting one page showed the selectors ``{'a': 1,
    'script': 1, True: 1, 'meta': 1}``, and ``_extract_internal_links`` alone was
    4.0 ms of the 30 ms a page took. One walk that dispatches on ``tag.name``
    replaces all four and touches each tag once.
    """

    __slots__ = (
        "_base_domain",
        "_base_url",
        "_js_seen",
        "_links_seen",
        "js_links",
        "links",
        "metadata",
    )

    def __init__(self) -> None:
        self._base_url = ""
        self._base_domain = ""
        self.metadata = SiteMetadata()
        self.links: list[str] = []
        self.js_links: list[str] = []
        self._links_seen: set[str] = set()
        self._js_seen: set[str] = set()

    def _add_title(self, soup: BeautifulSoup) -> None:
        title = soup.title
        if title and title.string:
            self.metadata.title = title.string.strip()

    def harvest(self, soup: BeautifulSoup, base_url: str) -> _Harvest:
        """Record every link, script and metadata tag in *soup*, in document order."""
        self._base_url = base_url
        self._base_domain = urlparse(base_url).netloc
        self._add_title(soup)
        self.walk(soup)
        return self

    def walk(self, soup: BeautifulSoup) -> None:
        """
        Visit every tag in *soup* once, dispatching on the tag name.

        Iterating the live tree rather than ``find_all(True)`` avoids
        materialising every tag in the document at once; a 213 KiB page peaked at
        2,258 KiB of transient list and attribute objects that way.
        """
        for tag in soup.descendants:
            if not isinstance(tag, Tag):
                continue
            name = tag.name
            if name == "a":
                self._add_link(tag)
            elif name == "script":
                self._add_script(tag)
            elif name == "meta":
                self._add_meta(tag)

    def _add_link(self, tag: Tag) -> None:
        href = tag.get("href")
        if href is None:
            return
        full = urljoin(self._base_url, str(href)).split("#")[0]
        p = urlparse(full)
        if (
            p.netloc == self._base_domain
            and p.scheme in ("http", "https")
            and full not in self._links_seen
        ):
            self._links_seen.add(full)
            self.links.append(full)

    def _add_script(self, tag: Tag) -> None:
        src = tag.get("src")
        if src is None:
            return
        full = urljoin(self._base_url, str(src))
        if full.startswith("http") and full not in self._js_seen:
            self._js_seen.add(full)
            self.js_links.append(full)

    def _add_meta(self, tag: Tag) -> None:
        meta = self.metadata
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


def _extract_metadata(soup: BeautifulSoup) -> SiteMetadata:
    """Read title and meta tags from *soup*."""
    harvested = _Harvest()
    harvested._add_title(soup)
    for tag in soup.descendants:
        if isinstance(tag, Tag) and tag.name == "meta":
            harvested._add_meta(tag)
    return harvested.metadata


def _extract_js_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extract JS script src URLs from a BeautifulSoup tree."""
    harvested = _Harvest()
    harvested._base_url = base_url
    for tag in soup.descendants:
        if isinstance(tag, Tag) and tag.name == "script":
            harvested._add_script(tag)
    return harvested.js_links


def _extract_internal_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extract de-duplicated links on the same domain as *base_url*."""
    return _Harvest().harvest(soup, base_url).links


def _extract_text(soup: BeautifulSoup, max_chars: int = MAX_TEXT_CHARS) -> str:
    """
    Extract visible text from an already-filtered tree.

    The budget is enforced while walking rather than by building the whole page
    text and slicing afterwards: a 140 kB page produced 113,000 characters to
    keep 10,000, so 91 % of the work (and the peak memory that came with it) was
    immediately discarded.
    """
    kept: list[str] = []
    used = 0
    for raw in soup.stripped_strings:
        line = str(raw).strip()
        if not line:
            continue
        # Include the newline this line will be preceded by, except the first.
        cost = len(line) + (1 if kept else 0)
        if used + cost > max_chars:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept)
