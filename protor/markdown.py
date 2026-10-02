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
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag

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


def _is_noise(tag: Tag) -> bool:
    """Check if a tag is likely noise based on common patterns."""
    if tag.name in _NOISE_TAGS:
        return True
    raw_classes = tag.get("class")
    classes = " ".join(raw_classes) if isinstance(raw_classes, list) else str(raw_classes or "")
    raw_ids = tag.get("id")
    ids = " ".join(raw_ids) if isinstance(raw_ids, list) else str(raw_ids or "")
    return bool(_NOISE_PATTERN.search(f"{classes} {ids}"))


def _render_inline_children(children, base_url: str) -> str:
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

        inner = _render_inline_children(child.children, base_url)
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


def _process_element(tag: Tag, base_url: str, lines: list[str], depth: int) -> None:
    """Recursively process a BeautifulSoup element into Markdown lines."""
    if _is_noise(tag):
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

    # Code blocks
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
        lines.append(text.rstrip())
        lines.append("```")
        lines.append("")
        return

    # Tables
    if name == "table":
        _process_table(tag, base_url, lines)
        return

    # Blockquotes
    if name == "blockquote":
        text = _render_inline(tag, base_url).strip()
        if text:
            lines.append("")
            for ln in text.splitlines():
                lines.append(f"> {ln}")
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

    # Images
    if name == "img":
        src = tag.get("src", "")
        alt = tag.get("alt", "")
        if src:
            full_src = urljoin(base_url, str(src))
            lines.append(f"![{alt}]({full_src})")
        return

    # Links
    if name == "a":
        return

    # Paragraphs, sections, and generic block containers
    if name in ("p", "div", "section", "article", "main", "figure", "figcaption", "dl", "dt", "dd"):
        _emit_block(tag, base_url, lines, depth)
        return

    # Line breaks
    if name == "br":
        lines.append("")
        return

    # Default: treat as a block container and recurse
    _emit_block(tag, base_url, lines, depth)


def _emit_block(tag: Tag, base_url: str, lines: list[str], depth: int) -> None:
    """
    Render a block container's children in document order.

    Runs of inline content are buffered into a single line and flushed whenever a
    block child is reached. Emitting all inline text first and recursing after
    would hoist trailing links above earlier headings and paragraphs.
    """
    buffer: list = []

    def flush() -> None:
        text = _render_inline_children(buffer, base_url).strip()
        buffer.clear()
        if text:
            lines.append(text)

    for child in tag.children:
        if isinstance(child, Tag) and child.name in _BLOCK_TAGS:
            flush()
            _process_element(child, base_url, lines, depth)
        else:
            buffer.append(child)
    flush()


def _process_list(tag: Tag, base_url: str, lines: list[str], depth: int) -> None:
    """Process ul/ol elements into Markdown lists."""
    is_ordered = tag.name == "ol"
    items = tag.find_all("li", recursive=False)
    if not items:
        return
    lines.append("")
    for i, item in enumerate(items, 1):
        prefix = f"{i}." if is_ordered else "-"
        indent = "  " * depth
        nested = item.find(("ul", "ol"), recursive=False)

        if nested:
            # Emit the item's own inline text, then the nested list indented.
            text = _render_inline(item, base_url).strip()
            if text:
                lines.append(f"{indent}{prefix} {text}")
            _process_list(nested, base_url, lines, depth + 1)
        else:
            text = _render_inline(item, base_url).strip()
            if text:
                lines.append(f"{indent}{prefix} {text}")
    lines.append("")


def _process_table(tag: Tag, base_url: str, lines: list[str]) -> None:
    """Convert HTML table to Markdown table."""
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


def clean_soup(soup: BeautifulSoup) -> None:
    """
    Remove noise and script/style elements from *soup* in place.

    This is the single, canonical page-filtering pass; every page output
    (plain text, Markdown) is derived from a tree already stripped here.
    Idempotent, so it is safe to call on a tree processed by
    :func:`html_to_markdown`. Callers that derive several artefacts from one
    parse should call this exactly once and then use the pure renderers.
    """
    for tag in soup.find_all(True):
        if tag.name in _SCRIPT_TAGS or _is_noise(tag):
            tag.decompose()


def soup_to_markdown(soup: BeautifulSoup, base_url: str = "") -> str:
    """
    Convert an already-parsed BeautifulSoup tree to Markdown.

    Pure renderer: it assumes :func:`clean_soup` has already run on *soup*, so
    the caller can parse a page once, clean it once, and derive both plain text
    and Markdown from the same tree. Use :func:`html_to_markdown` if you are
    starting from an HTML string and want the filtering pass included.
    """
    body = soup.find("body") or soup
    lines: list[str] = []
    _process_element(body, base_url, lines, 0)
    return _clean_markdown("\n".join(lines))


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
