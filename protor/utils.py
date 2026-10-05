"""Shared utilities: I/O helpers, filename sanitisation, paths."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse, urlunparse

from .exceptions import OutputPathError, URLValidationError

#: Most filesystems cap a single path component at 255 bytes. Staying well under
#: that leaves room for a suffix we may need to add.
MAX_FILENAME_LEN = 200

_UNSAFE = re.compile(r"[^a-zA-Z0-9_.-]")


def safe_filename(name: str) -> str:
    """
    Return a filesystem-safe version of *name*.

    Sanitises illegal characters and keeps the result within
    :data:`MAX_FILENAME_LEN`. A long name is truncated and given a short hash of
    the original, so two different URLs that share a long prefix still produce
    different files rather than overwriting each other.
    """
    cleaned = _UNSAFE.sub("_", name).strip("_") or "unnamed"
    if len(cleaned) <= MAX_FILENAME_LEN:
        return cleaned
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]
    suffix = Path(cleaned).suffix[:16]
    stem = cleaned[: MAX_FILENAME_LEN - len(digest) - len(suffix) - 1].rstrip("_")
    return f"{stem}.{digest}{suffix}"


def canonicalize_url(url: str) -> str:
    """Return a stable deduplication key for *url*.

    Memoising this with ``lru_cache`` was tried and reverted: it makes a repeat
    call ~100x cheaper (0.04 us against 4.1 us), and the engine really does
    canonicalise the same string twice per discovered link. But a crawl's URLs
    are overwhelmingly *distinct*, so in exchange the cache grows without bound
    and starts evicting — which made the project's own scaling gate report
    ``canonicalize_url`` superlinear (17x per-item cost at 80,000 distinct URLs),
    and holding 200k entries resident would cost tens of megabytes against a
    codebase whose site index was brought down to 0.58 MiB. Not worth 4 us a link.

    Lowercases the scheme and host, drops the fragment, and normalises
    ``index.html`` page paths to their directory, so ``/index.html`` and ``/``
    collapse into a single crawl target. Querystrings are preserved.
    """
    parsed = urlparse(url)
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = "/" + path
    lowered = path.lower()
    if lowered == "/index.html":
        path = "/"
    elif lowered.endswith("/index.html"):
        path = path[: -len("index.html")]
    netloc = parsed.netloc.lower()
    query = parsed.query
    return urlunparse((parsed.scheme.lower(), netloc, path, "", query, ""))


def page_filename(url: str, fallback: str = "index.html") -> str:
    """Return a stable, filesystem-safe page filename for *url*.

    The root path (and any path ending in ``/``) maps to *fallback* so the
    site's homepage is always ``index.html``.

    Deeper paths keep their full path, flattened with ``-``, rather than just
    the final segment: ``/docs/a.html`` and ``/blog/a.html`` share the leaf
    ``a.html``, and using the leaf alone made one crawl page silently
    overwrite the other's saved HTML.
    """
    path = unquote(urlparse(url).path).rstrip("/")
    if not path:
        return fallback
    parts = [p for p in path.split("/") if p not in ("", ".", "..")]
    if not parts or parts == ["index.html"] or parts == ["index.htm"]:
        return fallback
    # Strip leading dots so ".." can never survive into a filename.
    flattened = "-".join(p.lstrip(".") for p in parts)
    return safe_filename(flattened) or fallback


def manifest_filename(url: str) -> str:
    """Return the on-disk manifest name for the page at *url*.

    The site root keeps the conventional ``manifest.json`` (the batch-scrape
    contract); other pages get ``<page-name>.manifest.json`` so a multi-page
    crawl never overwrites earlier manifests.
    """
    page = page_filename(url)
    if page == "index.html":
        return "manifest.json"
    return f"{Path(page).stem}.manifest.json"


def ensure_output_dir(path: str | Path) -> Path:
    """
    Create *path* as a directory, or explain why it cannot be one.

    ``mkdir(parents=True, exist_ok=True)`` is happy to do nothing when the path
    already exists as a *directory*, and raises ``FileExistsError`` when it
    exists as a file — so ``protor scrape https://example.com -o notes.txt``
    died with a raw ``[Errno 17]`` traceback raised from deep inside a run,
    after the user had already waited on the network. Checking first also turns
    an unwritable parent into a sentence instead of a traceback.
    """
    p = Path(path)
    if p.exists() and not p.is_dir():
        kind = "a file" if p.is_file() else "not a directory"
        raise OutputPathError(str(p), f"it already exists and is {kind}")
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputPathError(str(p), exc.strerror or str(exc)) from exc
    return p


def save_json(data: Any, path: str | Path) -> None:
    """Serialise *data* to JSON, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Load JSON from *path*, raising FileNotFoundError with a clear message."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_default_output_dir() -> Path:
    """Return a sensible default output directory, cross-platform."""
    return Path.home() / "Downloads" / "protor"


#: Resolved URLs, keyed by (base, href), for the duration of one parse.
#:
#: A page's links are resolved twice: once by the parser, which is collecting them,
#: and again by the Markdown renderer, which is writing `[text](url)`. Measured on a
#: 1,320-link page that is 2,106 `urljoin` calls for 1,200 distinct pairs — 43%
#: redundant, at 4us each. It is the single largest cost in `parse_html` that is ours
#: rather than BeautifulSoup's.
#:
#: `urljoin` is pure, so memoising it is safe. Two independent mechanisms keep it so,
#: and they are not interchangeable — testing either one alone leaves the other
#: untestable, because each masks the other's absence:
#:
#: * The key carries the base, so an entry can never answer a different page's href.
#:   This is the correctness half, and it holds even if the cache is never cleared.
#: * The cache is cleared per parse by :func:`clear_url_cache`. This is the memory
#:   half only — correctness does not depend on it. Capped as well, since
#:   `resolve_url` is importable and a caller outside a parse could grow it.
_RESOLVED: dict[tuple[str, str], str] = {}

#: Entries above which the cache is dropped wholesale. A page has thousands of links
#: at most; anything past this is not one page's worth.
_URL_CACHE_MAX = 8192


def resolve_url(base: str, href: str) -> str:
    """
    ``urljoin(base, href)``, memoised for the duration of a parse.

    Split out rather than called at each site so the cache has one owner; see
    :data:`_RESOLVED` for why it exists.
    """
    key = (base, href)
    cached = _RESOLVED.get(key)
    if cached is not None:
        return cached
    if len(_RESOLVED) >= _URL_CACHE_MAX:
        _RESOLVED.clear()
    resolved = urljoin(base, href)
    _RESOLVED[key] = resolved
    return resolved


def clear_url_cache() -> None:
    """Start a new page's cache. Called by the parser; see :func:`resolve_url`."""
    _RESOLVED.clear()


def human_bytes(n: int) -> str:
    """Human-readable byte count."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def human_duration(ms: int | float | None) -> str:
    """
    Short, fixed-width rendering of a duration in milliseconds.

    The progress table's Time column is seven cells wide because a page that
    answered inside the default 30s timeout renders as ``30000ms``. That is exactly
    the width, and ``--timeout 300`` with three retries reaches 900000ms — which
    rich then ellipsised to ``900000…``, dropping the unit so the cell read as
    corrupt data rather than as a slow page.

    Units chosen so the result never exceeds six characters, whatever the timeout.
    """
    if not ms:
        return "\u2014"
    # Round once, then branch on the rounded value. Deciding first and formatting
    # after produced "60.0s" for 59999ms in one arrangement and "0m59s" in the
    # other, since 59.999 is under 60 but `divmod(int(59.999), 60)` is (0, 59).
    seconds = round(float(ms) / 1000.0, 1)
    if seconds < 1:
        return f"{int(ms)}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def validate_url(url: str) -> str:
    """Validate and normalise *url*.

    Raises URLValidationError if the URL is malformed or missing a scheme — a
    typed error so the CLI can offer a URL-shaped hint and nothing else.
    Returns the normalised URL string.
    """
    if not url or not isinstance(url, str):
        raise URLValidationError(str(url), f"URL must be a non-empty string, got: {url!r}")

    parsed = urlparse(url)
    if not parsed.scheme:
        raise URLValidationError(url, f"URL must include a scheme (http/https): {url!r}")
    if parsed.scheme not in ("http", "https"):
        raise URLValidationError(url, f"URL scheme must be http or https, got: {parsed.scheme!r}")
    if not parsed.netloc:
        raise URLValidationError(url, f"URL must include a hostname: {url!r}")

    return url
