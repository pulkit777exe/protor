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

# ── HTTP ──────────────────────────────────────────────────────────────────────

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

# ── retry ─────────────────────────────────────────────────────────────────────

MAX_RETRIES: int = 3
RETRYABLE_STATUS: set[int] = {429, 500, 502, 503, 504}
RETRY_BACKOFF_BASE: float = 0.5

# ── paths ─────────────────────────────────────────────────────────────────────


def get_default_output_dir() -> Path:
    """Return a sensible default output directory, cross-platform."""
    return Path.home() / "Downloads" / "protor"
