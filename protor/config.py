"""Centralised configuration for protor."""

from __future__ import annotations

import os
from pathlib import Path

# ── scraper ───────────────────────────────────────────────────────────────────

DEFAULT_CONCURRENCY: int = 6
DEFAULT_TIMEOUT: int = 30
MAX_JS_FILES: int = 15
MAX_TEXT_CHARS: int = 10_000
MAX_DATA_CHARS: int = 8_000
JS_DOWNLOAD_TIMEOUT: int = 15
RATE_LIMIT_DELAY: float = 0.5

# ── crawler ───────────────────────────────────────────────────────────────────

CRAWLER_CONCURRENCY: int = 4
CRAWLER_DELAY: float = 0.25
DEFAULT_MAX_PAGES: int = 10

# ── analyzer ──────────────────────────────────────────────────────────────────

OLLAMA_BASE: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ANALYSIS_MAX_DATA_CHARS: int = 8_000
ANALYSIS_TIMEOUT: int = 300
OLLAMA_CHECK_TIMEOUT: int = 5

# ── HTTP headers ──────────────────────────────────────────────────────────────

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

# ── User-Agent rotation (inspired by curl-impersonate / Scrapling) ────────────

USER_AGENTS: list[str] = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:127.0) Gecko/20100101 Firefox/127.0",
    # Firefox on Linux
    "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
    # Safari on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
]

# ── retry ─────────────────────────────────────────────────────────────────────

MAX_RETRIES: int = 3
RETRYABLE_STATUS: set[int] = {429, 500, 502, 503, 504}
RETRY_BACKOFF_BASE: float = 0.5

# ── auto-scaling concurrency (inspired by Crawlee) ───────────────────────────

SCALING_ENABLED: bool = True
SCALING_WINDOW: int = 10  # number of recent requests to evaluate
SCALING_UP_THRESHOLD: float = 0.8  # success rate to scale up
SCALING_DOWN_THRESHOLD: float = 0.5  # success rate to scale down
SCALING_MIN_CONCURRENCY: int = 2
SCALING_MAX_CONCURRENCY: int = 20
SCALING_COOLDOWN: float = 5.0  # seconds between scaling adjustments

# ── crawl checkpointing (inspired by Crawl4AI) ───────────────────────────────

CHECKPOINT_FILENAME: str = "crawl_checkpoint.json"

# ── content filtering (inspired by Crawl4AI's PruningContentFilter) ───────────

CONTENT_FILTER_MIN_WORDS: int = 50
NOISE_TAGS: set[str] = {
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "img",
}

# ── paths ─────────────────────────────────────────────────────────────────────


def get_default_output_dir() -> Path:
    """Return a sensible default output directory, cross-platform."""
    return Path.home() / "Downloads" / "protor"
