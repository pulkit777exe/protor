"""
protor.markdown
~~~~~~~~~~~~~~~
Convert HTML to clean, LLM-friendly Markdown.

Inspired by Crawl4AI's fit_markdown and Firecrawl's clean output.
Uses heuristic pruning to remove noise (nav, ads, footers) and
produces structured Markdown with headings, tables, and code blocks.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

# bs4 omits both of these from its top-level ``__all__`` (only ``Tag`` and the
# markup classes are re-exported), so under ``no_implicit_reexport`` the
# canonical ``bs4.element`` module is where they have to come from. They are the
# same objects ``bs4`` itself imports, so this is a source fix, not a
# behavioural one. ``PageElement`` is the common base of ``NavigableString``
# and ``Tag``, which is exactly what a run of soup children contains.
from bs4.element import NavigableString, PageElement

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["clean_soup", "extract_clean_markdown", "html_to_markdown", "soup_to_markdown"]

# Tags to strip entirely (noise)
_NOISE_TAGS = {"nav", "footer", "header", "aside", "form", "button", "input", "select", "textarea"}
_SCRIPT_TAGS = {"script", "style", "noscript", "iframe"}
_INLINE_TAGS = {"span", "em", "strong", "b", "i", "u", "a", "code", "sup", "sub", "small", "mark"}

# Compiled once at import: _is_noise runs per tag, so re-compiling per call
# showed up as a measurable share of parse time on large pages.
_NOISE_PATTERN = re.compile(
    r"sidebar|widget|popup|modal|overlay|banner|cookie|consent|newsletter|"
    r"subscribe|social|share|comment|disqus|related|recommended|advertisement|"
    r"promo|sponsor|ad-|ads-|tracking|analytics|breadcrumb|pagination|pager",
    re.IGNORECASE,
)

# Tags that start a new Markdown block. Inline rendering stops at these so a
# paragraph containing <strong>/<a> stays on one line instead of being
# shredded into one line per text node.
_BLOCK_TAGS = (
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "figure",
        "figcaption",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "td",
        "th",
        "pre",
        "blockquote",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
    | _NOISE_TAGS
    | _SCRIPT_TAGS
)

# Inline tags that map onto Markdown emphasis/code markers.
_EMPHASIS = {"strong": "**", "b": "**", "em": "*", "i": "*"}

_WHITESPACE = re.compile(r"[ \t\r\n\f\v]+")

#: Maximum element nesting the renderer will recurse through.
#:
#: Both the inline and block renderers walk the tree recursively, and Python's
#: default recursion limit put the practical ceiling at 494 nested elements —
#: reachable by machine-generated markup or a hostile page, where the whole
#: scrape died with ``RecursionError``. Past this depth the remaining subtree is
#: flattened to its text, so deeply nested content degrades to plain text
#: instead of crashing.
MAX_RENDER_DEPTH = 150

#: Allowance between the budget the renderer walks to and the cap the caller
#: asked for. :func:`_clean_markdown` strips the finished document, so what the
#: renderer charges and what it returns can differ by a leading or trailing
#: newline; without the allowance a page that fitted the cap could be cut short
#: by exactly those characters.
_BUDGET_SLACK = 8


class _Lines(list[str]):
    """
    The renderer's output buffer, carrying the caller's character budget.

    The Markdown cap used to be applied to the finished string, so a whole page
    was walked, joined and then thrown away: 1200 blocks produced 460 kB of
    output to keep 40 kB. Charging each line as it is appended lets the walk
    stop at the cap instead, which is 2.5x faster at 350 blocks and 8.5x at
    1200.

    ``used`` counts the characters :func:`_clean_markdown` would *return*, not
    the ones appended: a run of blank lines is charged the single newline pair it
    collapses to, and lines are rstripped on the way in, which is exactly what
    the finished-text pass does. Without that the accounting would drift ahead of
    the cap and cut a page that would have fitted.
    """

    __slots__ = ("blank", "capped", "limit", "used")

    def __init__(self, limit: int = 0) -> None:
        super().__init__()
        self.limit = limit
        self.used = 0
        self.blank = 0  # blank lines already charged at the tail, saturating at two
        self.capped = False

    def append(self, text: str) -> None:
        list.append(self, text)
        limit = self.limit
        if limit <= 0:
            return
        if len(self) == 1:
            # Nothing precedes the first line, so it costs no separator newline.
            self.used += len(text)
        else:
            # The newline already charged to a trailing blank line doubles as
            # this line's separator, so a blank line and the line after it
            # together cost one character.
            self.used += len(text) + (1 if self.blank < 2 else 0)
            if text:
                self.blank = 0
            elif self.blank < 2:
                self.blank += 1
        if self.used > limit:
            self.capped = True


def _is_noise(tag: Tag) -> bool:
    """Check if a tag is likely noise based on common patterns."""
    if tag.name in _NOISE_TAGS:
        return True
    raw_classes = tag.get("class")
    raw_ids = tag.get("id")
    # The pattern can only ever match a class or an id, and most tags carry
    # neither: skipping the join-and-search for those was worth 1.19x on the
    # noise pass, and this runs twice per tag (clean_soup, then every
    # _process_element).
    if not raw_classes and not raw_ids:
        return False
    classes = " ".join(raw_classes) if isinstance(raw_classes, list) else str(raw_classes or "")
    ids = " ".join(raw_ids) if isinstance(raw_ids, list) else str(raw_ids or "")
    return bool(_NOISE_PATTERN.search(f"{classes} {ids}"))


def _has_block_child(tag: Tag) -> bool:
    """True if *tag* contains a block-level child element."""
    return any(isinstance(c, Tag) and c.name in _BLOCK_TAGS for c in tag.children)


def _render_inline_children(children: Iterable[PageElement], base_url: str, _depth: int = 0) -> str:
    """
    Render a run of inline nodes as a single Markdown string.

    Emphasis, links, code, and images are folded into the surrounding text
    instead of each becoming its own line, so a paragraph stays a paragraph.
    Block-level nodes are skipped; the caller walks those separately.
    """
    parts: list[str] = []
    for child in children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue

        name = child.name
        if name in _BLOCK_TAGS or name in _SCRIPT_TAGS:
            continue

        if name == "br":
            parts.append("\n")
            continue

        if name == "img":
            src = child.get("src", "")
            if src:
                alt = child.get("alt", "")
                parts.append(f"\n\n![{alt}]({urljoin(base_url, str(src))})\n\n")
            continue

        if _depth >= MAX_RENDER_DEPTH:
            parts.append(child.get_text(" ", strip=True))
            continue
        inner = _render_inline_children(child.children, base_url, _depth + 1)
        if name == "a":
            href = child.get("href", "")
            text = inner.strip()
            if href and text:
                parts.append(f"[{text}]({urljoin(base_url, str(href))})")
            elif text:
                parts.append(text)
            continue

        text = inner.strip()
        if not text:
            continue
        marker = _EMPHASIS.get(name)
        if marker:
            parts.append(f"{marker}{text}{marker}")
        elif name == "code":
            parts.append(f"`{text}`")
        else:
            parts.append(text)

    return _WHITESPACE.sub(" ", "".join(parts))


def _render_inline(node: Tag, base_url: str) -> str:
    """Render *node*'s inline content as a single Markdown string."""
    return _render_inline_children(node.children, base_url)


def _process_element(tag: Tag, base_url: str, lines: _Lines, depth: int, _rd: int = 0) -> None:
    """
    Recursively process a BeautifulSoup element into Markdown lines.

    Only *block* elements reach this dispatcher: :func:`_emit_block` recurses
    into :data:`_BLOCK_TAGS` and :func:`soup_to_markdown` starts at ``<body>``.
    The inline elements are rendered by :func:`_render_inline_children` out of
    the run buffer, which is why there are no ``img``/``a``/``br`` branches here
    - none of the three is in :data:`_BLOCK_TAGS`, so the code that handled them
    could never run.
    """
    if lines.capped:
        # Past the budget: the rest of the document is discarded anyway, so stop
        # before paying for the walk instead of after producing it.
        return

    if _is_noise(tag):
        return

    if _rd >= MAX_RENDER_DEPTH:
        # Too deep to keep walking; flatten rather than exhaust the stack.
        flat = tag.get_text(" ", strip=True)
        if flat:
            lines.append(flat)
        return

    name = tag.name

    if name in _SCRIPT_TAGS:
        return

    # Headings
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1])
        text = tag.get_text(strip=True)
        if text:
            lines.append("")
            lines.append(f"{'#' * level} {text}")
            lines.append("")
        return

    # Code blocks. The body goes in one line at a time, rstripped: joining those
    # lines back reproduces the body exactly, and doing it here keeps the
    # budget's count identical to what the finished-text pass will produce.
    if name == "pre":
        code_tag = tag.find("code")
        text = (code_tag or tag).get_text()
        lang = ""
        if code_tag and code_tag.get("class"):
            for cls in code_tag["class"]:
                if cls.startswith("language-"):
                    lang = cls[9:]
                    break
        lines.append("")
        lines.append(f"```{lang}")
        lines.extend([ln.rstrip() for ln in text.rstrip().split("\n")] or [""])
        lines.append("```")
        lines.append("")
        return

    # Tables
    if name == "table":
        _process_table(tag, base_url, lines)
        return

    # Blockquotes
    if name == "blockquote":
        # A blockquote may wrap block elements (<blockquote><p>…</p>), which
        # inline rendering skips entirely — that silently dropped the whole
        # quotation. Render the children, then prefix every line.
        nested = _Lines()
        if _has_block_child(tag):
            _emit_block(tag, base_url, nested, depth, _rd + 1)
        else:
            text = _render_inline(tag, base_url).strip()
            if text:
                nested.append(text)
        if nested:
            lines.append("")
            for ln in "\n".join(nested).splitlines():
                lines.append(f"> {ln}" if ln.strip() else ">")
            lines.append("")
        return

    # Lists
    if name in ("ul", "ol"):
        _process_list(tag, base_url, lines, depth)
        return

    # Horizontal rules
    if name == "hr":
        lines.append("")
        lines.append("---")
        lines.append("")
        return

    # Paragraphs, sections, definition lists, and generic block containers.
    _emit_block(tag, base_url, lines, depth, _rd + 1)


def _emit_block(tag: Tag, base_url: str, lines: _Lines, depth: int, _rd: int = 0) -> None:
    """
    Render a block container's children in document order.

    Runs of inline content are buffered into a single line and flushed whenever a
    block child is reached. Emitting all inline text first and recursing after
    would hoist trailing links above earlier headings and paragraphs.
    """
    buffer: list[PageElement] = []

    def flush() -> None:
        text = _render_inline_children(buffer, base_url).strip()
        buffer.clear()
        if text:
            lines.append(text)

    for child in tag.children:
        if lines.capped:
            return
        if isinstance(child, Tag) and child.name in _BLOCK_TAGS:
            flush()
            _process_element(child, base_url, lines, depth, _rd + 1)
        else:
            buffer.append(child)
    flush()


def _render_li_body(item: Tag, base_url: str) -> list[str]:
    """
    Render one ``<li>``'s own content, excluding any nested list.

    An item may hold block content — ``<li><p>…</p></li>`` is ordinary markup,
    and so is a definition or paragraph inside a list entry — and inline
    rendering skips block elements, so the whole item used to vanish from the
    output. The nested list is deliberately excluded: the caller emits it
    itself, indented, on its own lines.
    """
    if not _has_block_child(item):
        text = _render_inline(item, base_url).strip()
        return [text] if text else []

    scratch = _Lines()
    buffer: list[PageElement] = []

    def flush() -> None:
        text = _render_inline_children(buffer, base_url).strip()
        buffer.clear()
        if text:
            scratch.append(text)

    for child in item.children:
        if isinstance(child, Tag) and (child.name in _BLOCK_TAGS or child.name in ("ul", "ol")):
            flush()
            if child.name not in ("ul", "ol"):
                _process_element(child, base_url, scratch, 0)
        else:
            buffer.append(child)
    flush()
    return [ln for ln in scratch if ln.strip()]


def _process_list(tag: Tag, base_url: str, lines: _Lines, depth: int) -> None:
    """Process ul/ol elements into Markdown lists."""
    is_ordered = tag.name == "ol"
    items = tag.find_all("li", recursive=False)
    if not items:
        return
    lines.append("")
    for i, item in enumerate(items, 1):
        if lines.capped:
            return
        prefix = f"{i}." if is_ordered else "-"
        indent = "  " * depth
        nested = item.find(("ul", "ol"), recursive=False)

        # The first line carries the marker; any further lines from block
        # content inside the item are indented under it as continuations.
        body = _render_li_body(item, base_url)
        if body:
            lines.append(f"{indent}{prefix} {body[0]}")
            lines.extend(f"{indent}  {ln}" for ln in body[1:])

        if nested:
            _process_list(nested, base_url, lines, depth + 1)
    lines.append("")


def _process_table(tag: Tag, base_url: str, lines: _Lines) -> None:
    """Convert HTML table to Markdown table."""
    # A table is emitted whole or not at all: its column widths come from every
    # row, so stopping part-way through would reflow the part that is kept.
    if lines.capped:
        return
    rows = tag.find_all("tr")
    if not rows:
        return

    table_data: list[list[str]] = []
    for row in rows:
        cells = row.find_all(["th", "td"])
        table_data.append([c.get_text(strip=True) for c in cells])

    if not table_data:
        return

    # Determine column widths
    num_cols = max(len(r) for r in table_data)
    col_widths = [0] * num_cols
    for row_data in table_data:
        for i, cell in enumerate(row_data):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(cell))

    # Normalize row lengths
    for row_data in table_data:
        while len(row_data) < num_cols:
            row_data.append("")

    lines.append("")
    # Header
    header = table_data[0]
    lines.append("| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(header)) + " |")
    lines.append("| " + " | ".join("-" * col_widths[i] for i in range(num_cols)) + " |")
    # Body
    for row_data in table_data[1:]:
        lines.append(
            "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(row_data)) + " |"
        )
    lines.append("")


def _clean_markdown(text: str) -> str:
    """Post-process Markdown text to clean up artifacts."""
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing whitespace
    text = "\n".join(ln.rstrip() for ln in text.splitlines())
    return text.strip()


def _truncate(text: str, limit: int) -> str:
    """Trim *text* to *limit* characters, marking that it was cut."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[truncated]"


def _is_decomposed(tag: Tag) -> bool:
    """
    True once *tag* has already been removed by an earlier :meth:`decompose`.

    Decomposing an element also clears the ``__dict__`` of everything inside it.
    A nested tag inspected afterwards has ``attrs is None``, so the noise check
    would raise ``AttributeError: 'NoneType' object has no attribute 'get'`` and
    the whole page would be recorded as a scrape error. Any page with a ``<nav>``,
    ``<header>``, ``<footer>`` or ``<aside>`` containing elements hit this.

    The check reads ``__dict__`` directly rather than using ``getattr``. bs4's
    ``Tag.__getattr__`` forwards *any* missing attribute to ``self.find(name)``,
    and the ``decomposed`` property itself does ``getattr(self, "_decomposed",
    ...)`` — which misses the instance dict and re-enters ``__getattr__``,
    triggering a full subtree walk per tag. Measured on a 2,003-tag page nested
    2,000 deep: 626 ms via ``getattr`` versus 0.5 ms here, and the cost was
    quadratic in depth.
    """
    return tag.__dict__.get("_decomposed", False) is True or tag.parent is None


def clean_soup(soup: BeautifulSoup) -> None:
    """
    Remove noise and script/style elements from *soup* in place.

    This is the single, canonical page-filtering pass; every page output
    (plain text, Markdown) is derived from a tree already stripped here.
    Idempotent, so it is safe to call on a tree processed by
    :func:`html_to_markdown`. Callers that derive several artefacts from one
    parse should call this exactly once and then use the pure renderers.

    The descent keeps one level of children at a time and never enters a subtree
    it is about to remove, where ``find_all(True)`` materialised every tag in the
    document first - 447 KiB of transient list and attribute objects for a
    177 KiB page, against 9 KiB here. That also makes a removed subtree
    unreachable rather than merely detectable, which is what the
    :func:`_is_decomposed` guard exists for.
    """
    pending: list[Tag] = [soup]
    while pending:
        element = pending.pop()
        for child in list(element.contents):
            if not isinstance(child, Tag) or _is_decomposed(child):
                continue
            if child.name in _SCRIPT_TAGS or _is_noise(child):
                child.decompose()
            else:
                pending.append(child)


def soup_to_markdown(soup: BeautifulSoup, base_url: str = "", max_chars: int = 0) -> str:
    """
    Convert an already-parsed BeautifulSoup tree to Markdown.

    Pure renderer: it assumes :func:`clean_soup` has already run on *soup*, so
    the caller can parse a page once, clean it once, and derive both plain text
    and Markdown from the same tree. Use :func:`html_to_markdown` if you are
    starting from an HTML string and want the filtering pass included.

    *max_chars* stops the walk once the rendered document has grown past that
    many characters and marks the result as cut, rather than rendering the page
    in full and slicing afterwards. Output below the cap is unaffected; a page
    above it keeps the same prefix and marker, and may lose or gain the final
    line depending on where the walk stops.
    """
    body = soup.find("body") or soup
    lines = _Lines(max_chars + _BUDGET_SLACK if max_chars > 0 else 0)
    _process_element(body, base_url, lines, 0)
    text = _clean_markdown("\n".join(lines))
    return _truncate(text, max_chars) if lines.capped else text


def html_to_markdown(html: str, base_url: str = "") -> str:
    """
    Convert HTML string to clean Markdown.

    Parameters
    ----------
    html:
        Raw HTML content.
    base_url:
        Base URL for resolving relative links and images.

    Returns
    -------
    str
        Clean Markdown representation.
    """
    soup = BeautifulSoup(html, "lxml")
    clean_soup(soup)
    return soup_to_markdown(soup, base_url)


def extract_clean_markdown(html: str, base_url: str = "", max_chars: int = 0) -> str:
    """
    Extract clean Markdown content from HTML, removing all noise.

    This is the high-level API meant for scraping workflows.

    Parameters
    ----------
    html:
        Raw HTML content.
    base_url:
        Base URL for resolving relative links.
    max_chars:
        Maximum characters (0 = no limit).

    Returns
    -------
    str
        Clean Markdown suitable for LLM consumption.
    """
    md = html_to_markdown(html, base_url)
    if max_chars and len(md) > max_chars:
        md = md[:max_chars] + "\n\n[truncated]"
    return md
