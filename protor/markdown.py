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

__all__ = ["extract_clean_markdown", "html_to_markdown"]

# Tags to strip entirely (noise)
_NOISE_TAGS = {"nav", "footer", "header", "aside", "form", "button", "input", "select", "textarea"}
_SCRIPT_TAGS = {"script", "style", "noscript", "iframe"}
_INLINE_TAGS = {"span", "em", "strong", "b", "i", "u", "a", "code", "sup", "sub", "small", "mark"}


def _is_noise(tag: Tag) -> bool:
    """Check if a tag is likely noise based on common patterns."""
    if tag.name in _NOISE_TAGS:
        return True
    classes = " ".join(tag.get("class", []))
    ids = " ".join(tag.get("id", []))
    noise_patterns = re.compile(
        r"sidebar|widget|popup|modal|overlay|banner|cookie|consent|newsletter|"
        r"subscribe|social|share|comment|disqus|related|recommended|advertisement|"
        r"promo|sponsor|ad-|ads-|tracking|analytics|breadcrumb|pagination|pager",
        re.IGNORECASE,
    )
    combined = f"{classes} {ids}"
    return bool(noise_patterns.search(combined))


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
        text = tag.get_text(strip=True)
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
            full_src = urljoin(base_url, src)
            lines.append(f"![{alt}]({full_src})")
        return

    # Links
    if name == "a":
        href = tag.get("href", "")
        text = tag.get_text(strip=True)
        if href and text:
            full_href = urljoin(base_url, href)
            lines.append(f"[{text}]({full_href})")
        elif text:
            lines.append(text)
        return

    # Paragraphs and divs
    if name in ("p", "div", "section", "article", "main", "figure", "figcaption"):
        for child in tag.children:
            if isinstance(child, NavigableString):
                text = child.strip()
                if text:
                    lines.append(text)
            elif isinstance(child, Tag):
                _process_element(child, base_url, lines, depth)
        if name == "p":
            lines.append("")
        return

    # Line breaks
    if name == "br":
        lines.append("")
        return

    # Default: process children
    for child in tag.children:
        if isinstance(child, NavigableString):
            text = child.strip()
            if text:
                lines.append(text)
        elif isinstance(child, Tag):
            _process_element(child, base_url, lines, depth)


def _process_list(tag: Tag, base_url: str, lines: list[str], depth: int) -> None:
    """Process ul/ol elements into Markdown lists."""
    is_ordered = tag.name == "ol"
    for i, item in enumerate(tag.find_all("li", recursive=False)):
        prefix = f"{i + 1}." if is_ordered else "-"
        item_text = item.get_text(strip=True)
        # Check for nested lists
        nested = item.find(("ul", "ol"))
        if nested:
            # Get text before nested list
            for child in item.children:
                if isinstance(child, NavigableString):
                    text = child.strip()
                    if text:
                        lines.append(f"{'  ' * depth}{prefix} {text}")
                elif isinstance(child, Tag) and child.name not in ("ul", "ol"):
                    text = child.get_text(strip=True)
                    if text:
                        lines.append(f"{'  ' * depth}{prefix} {text}")
            _process_list(nested, base_url, lines, depth + 1)
        else:
            if item_text:
                lines.append(f"{'  ' * depth}{prefix} {item_text}")


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
    for row in table_data:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(cell))

    # Normalize row lengths
    for row in table_data:
        while len(row) < num_cols:
            row.append("")

    lines.append("")
    # Header
    header = table_data[0]
    lines.append("| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(header)) + " |")
    lines.append("| " + " | ".join("-" * col_widths[i] for i in range(num_cols)) + " |")
    # Body
    for row in table_data[1:]:
        lines.append("| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(row)) + " |")
    lines.append("")


def _clean_markdown(text: str) -> str:
    """Post-process Markdown text to clean up artifacts."""
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing whitespace
    text = "\n".join(ln.rstrip() for ln in text.splitlines())
    return text.strip()


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

    # Remove noise elements
    for tag in soup.find_all(True):
        if _is_noise(tag):
            tag.decompose()

    # Remove script/style
    for tag in soup.find_all(_SCRIPT_TAGS):
        tag.decompose()

    # Process from body or root
    body = soup.find("body") or soup
    lines: list[str] = []
    _process_element(body, base_url, lines, 0)

    return _clean_markdown("\n".join(lines))


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
