"""Centralised configuration for protor."""

from __future__ import annotations

import os

# ── scraper ───────────────────────────────────────────────────────────────────

DEFAULT_CONCURRENCY: int = 6
DEFAULT_TIMEOUT: int = 30
MAX_JS_FILES: int = 15
MAX_TEXT_CHARS: int = 10_000
#: Cap on the Markdown artefact stored per page. Markdown is denser than plain
#: text (links, headings, code fences) so it gets a larger budget, but it is
#: still derived data — the saved HTML is the source of truth — and an uncapped
#: value let one long page add ~119k characters to every manifest.
MAX_MARKDOWN_CHARS: int = 40_000
JS_DOWNLOAD_TIMEOUT: int = 15
#: Ceiling on the JS batch for a *single page*, regardless of per-file timeout.
#: Script assets are best-effort extras, but a page referencing one unreachable
#: CDN held its concurrency slot for the full JS_DOWNLOAD_TIMEOUT and delayed the
#: whole run — measured 15.5 s for one page whose only script was dead. Anything
#: still running when this expires is dropped.
JS_GROUP_TIMEOUT: int = 8
RATE_LIMIT_DELAY: float = 0.5

# ── crawler ───────────────────────────────────────────────────────────────────

CRAWLER_CONCURRENCY: int = 4
CRAWLER_DELAY: float = 0.25
DEFAULT_MAX_PAGES: int = 10

# ── analyzer ──────────────────────────────────────────────────────────────────

OLLAMA_BASE: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
#: Cap on the scraped context handed to the model, in characters. The budget is
#: split evenly across sites so a large batch never truncates the tail.
ANALYSIS_MAX_DATA_CHARS: int = 8_000
#: Per-request timeout when probing Ollama for available models.
OLLAMA_CHECK_TIMEOUT: int = 5
#: Timeout for a streamed generation request, in seconds.
ANALYSIS_TIMEOUT: int = 300

# ── HTTP headers ──────────────────────────────────────────────────────────────

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Accept-Encoding is deliberately NOT set here. Advertising brotli without the
# brotli package installed makes servers return brotli bodies aiohttp cannot
# decode. aiohttp negotiates gzip/deflate (and br when available) itself, so
# overriding the header only removes capability.

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

SCALING_WINDOW: int = 10  # number of recent requests to evaluate
SCALING_UP_THRESHOLD: float = 0.8  # success rate to scale up
SCALING_DOWN_THRESHOLD: float = 0.5  # success rate to scale down
SCALING_MIN_CONCURRENCY: int = 2
SCALING_MAX_CONCURRENCY: int = 20
SCALING_COOLDOWN: float = 5.0  # seconds between scaling adjustments

# ── crawl checkpointing (inspired by Crawl4AI) ───────────────────────────────

CHECKPOINT_FILENAME: str = "crawl_checkpoint.json"

# ── paths ─────────────────────────────────────────────────────────────────────

#: Where `HTTPCache` keeps its index and bodies when no directory is named.
#:
#: Overridable because the default is the user's *real* `~/.cache`, and a test
#: suite that reaches it has two problems at once. It cannot be isolated: every
#: `HTTPCache()` with no argument opens the same `index.json`, and any instance
#: still holding an older copy in memory writes the whole file back on flush. So
#: one test's on-disk edit — ageing entries to force a revalidation — is undone
#: by the next unrelated test to flush, and the failure looks like a timing bug
#: in the crawl rather than shared mutable state. It also leaves the developer's
#: real cache full of entries pointing at a test server's dead port.
#:
#: Read once at import, like `OLLAMA_BASE`. Setting it per-process is what the
#: suite needs; a test that wants a private cache can still pass `cache_dir=`.
HTTP_CACHE_DIR: str | None = os.environ.get("PROTOR_CACHE_DIR") or None
