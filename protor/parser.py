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

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from .config import MAX_MARKDOWN_CHARS, MAX_TEXT_CHARS
from .markdown import clean_soup, soup_to_markdown
from .models import SiteMetadata
from .utils import clear_url_cache, resolve_url

__all__ = [
    "ParsedPage",
    "extract_links",
    "looks_like_html",
    "parse_html",
    "parse_soup",
]


@dataclass
class ParsedPage:
    """Everything derived from a single HTML parse of one page."""

    metadata: SiteMetadata
    text_content: str
    markdown_content: str
    links: list[str]
    js_links: list[str]


#: Content types that are text and are worth parsing as a page.
_TEXTUAL_TYPES = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "application/xhtml",
        "text/plain",
        "text/markdown",
        "text/xml",
        "application/xml",
        "application/json",
    }
)

#: Types that are definitely a file rather than a page. Matched by prefix so
#: ``image/svg+xml`` and ``video/mp4`` are covered by their families.
#: ``application/octet-stream`` is deliberately absent — see ``looks_like_html``.
_BINARY_PREFIXES = ("image/", "video/", "audio/", "font/")
_BINARY_TYPES = frozenset(
    {
        "application/pdf",
        "application/zip",
        "application/gzip",
        "application/x-gzip",
        "application/x-tar",
        "application/msword",
        "application/rtf",
        "application/x-msdownload",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
    }
)


def looks_like_html(content_type: str, body: str = "") -> bool:
    """
    Whether a response is a web page, rather than a file served over HTTP.

    Nothing checked the ``Content-Type``, so a ``<a href="/manual.pdf">`` was
    "scraped" into two thousand characters of ``%PDF-1.4`` and reported as a
    successfully scraped page — the same failure-as-success shape as a stale CSS
    selector, one layer down.

    Decided from the header where it is decisive, and from the body where the
    header is unhelpful or absent. ``application/octet-stream`` is deliberately
    *not* decisive: plenty of servers send it for perfectly good HTML, so it
    falls through to the sniff and keeps real content. Sniffing is what makes
    this safe to apply to every response — a wrong "not a page" would drop real
    content, so the body gets the casting vote whenever it looks like markup,
    and only an unambiguous type (``application/pdf``, ``image/*``) is believed
    over it.
    """
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if ctype in _TEXTUAL_TYPES:
        return True
    if ctype in _BINARY_TYPES or ctype.startswith(_BINARY_PREFIXES):
        return False

    # Unknown, absent, or something exotic: look at the bytes.
    head = body[:512].lstrip().lower()
    return head.startswith(("<!doctype html", "<html", "<?xml", "<head", "<body", "<div"))


def parse_html(
    html: str,
    base_url: str,
    *,
    max_chars: int = MAX_TEXT_CHARS,
    strip_guessed_noise: bool = True,
) -> tuple[BeautifulSoup, ParsedPage]:
    """
    Parse *html* once and return the tree plus all derived artefacts.

    The returned tree is the noise-filtered tree; the parsed page is derived
    from it, so callers that need the filtered tree for hooks get it for free.
    """
    soup = BeautifulSoup(html, "lxml")
    return soup, parse_soup(
        soup, base_url, max_chars=max_chars, strip_guessed_noise=strip_guessed_noise
    )


def parse_soup(
    soup: BeautifulSoup,
    base_url: str,
    *,
    max_chars: int = MAX_TEXT_CHARS,
    max_markdown_chars: int = MAX_MARKDOWN_CHARS,
    strip_guessed_noise: bool = True,
) -> ParsedPage:
    """
    Derive a :class:`ParsedPage` from an already-parsed tree.

    *strip_guessed_noise* is passed through to :func:`clean_soup`; see
    :data:`protor.markdown._NOISE_PATTERN` for what it gives up.
    """
    # One page owns the URL cache for its duration: the harvest below and the
    # Markdown renderer further down both resolve this page's hrefs, and they are
    # the two halves of the duplication `resolve_url` exists to collapse.
    clear_url_cache()

    # Links, JS references and metadata come off the raw tree in a single walk,
    # because the canonical noise-filtering pass below removes the scripts and
    # navigational markup they are read from.
    harvested = _Harvest().harvest(soup, base_url)

    # The one canonical filtering pass. Everything below reads this filtered
    # tree, so text and Markdown stay consistent and the walk happens once.
    clean_soup(soup, strip_guessed_noise=strip_guessed_noise)
    # …and the renderer is told the same answer, because it re-asks the question
    # per subtree. Both were asked independently and used to disagree, which is
    # the whole of the bug; see the note at the soup_to_markdown call below.

    return ParsedPage(
        metadata=harvested.metadata,
        text_content=_extract_text(soup, max_chars),
        markdown_content=soup_to_markdown(
            soup,
            base_url,
            max_chars=max_markdown_chars,
            # The renderer re-asks _is_noise for every subtree it walks, because
            # clean_soup has already removed the noise and a Tag can also arrive
            # from html_to_markdown's caller with no filtering pass behind it.
            # That second question was asked with the default, so a `--schema`
            # run — the one case where the guesses must be KEPT — got a tree
            # filter that honoured the flag and a renderer that did not. Its
            # markdown_content came back empty while text_content was correct,
            # because on such a page every block sits under the very class the
            # filter would have guessed at.
            strip_guessed_noise=strip_guessed_noise,
        ),
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
        full = resolve_url(self._base_url, str(href)).split("#")[0]
        # urlparse raises on a netloc with an unbalanced bracket, and this is the
        # one place in the parse path where untrusted bytes reach it with no
        # guard. `http://exa[mple.com/` — a single stray bracket in a hostname,
        # not a crafted payload — raised `ValueError: Invalid IPv6 URL` out of
        # `parse_html`, and the engine's catch-all turned the whole page into a
        # recorded scrape error: no title, no text, no Markdown, no links, no
        # manifest. The HTML was already on disk with nothing pointing at it.
        #
        # A link that cannot be parsed is not a link. Skipping it keeps the other
        # 999 on the page, which is the entire point of parsing links.
        try:
            p = urlparse(full)
        except ValueError:
            return
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
        full = resolve_url(self._base_url, str(src))
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


# Any Unicode letter or digit. `\w` alone would accept `_`, which is precisely
# the sort of placeholder-only string being filtered out.
_HAS_ALNUM = re.compile(r"[^\W_]", re.UNICODE)


def _extract_text(soup: BeautifulSoup, max_chars: int = MAX_TEXT_CHARS) -> str:
    """
    Extract visible text from an already-filtered tree.

    The budget is enforced while walking rather than by building the whole page
    text and slicing afterwards: a 140 kB page produced 113,000 characters to
    keep 10,000, so 91 % of the work (and the peak memory that came with it) was
    immediately discarded.

    Strings carrying no letter or digit are dropped. Tables built for layout put
    their structure in the document as text: Hacker News separates every column
    with a literal ``|`` and wraps each link's domain in bare parentheses, and
    31 % of the strings on its front page are punctuation with nothing else in
    them. That is decoration, not content, and it is not free — the preview this
    produces is the body of the prompt the analyser sends, so every ``|`` is
    charged against the context window the budget exists to protect. On the same
    page a modern layout like DuckDuckGo's scores 1 %.

    A page whose content is *only* punctuation would come back empty, which is
    the one way this loses real text; nothing that reads as language is affected.
    """
    kept: list[str] = []
    used = 0
    for raw in soup.stripped_strings:
        line = str(raw).strip()
        if not line or not _HAS_ALNUM.search(line):
            continue
        # Include the newline this line will be preceded by, except the first.
        cost = len(line) + (1 if kept else 0)
        if used + cost > max_chars:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept)
