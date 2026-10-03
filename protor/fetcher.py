"""
protor.fetcher
~~~~~~~~~~~~~~
Deep HTTP fetch module. Owns retry, conditional caching, hooks, and User-Agent
rotation, so the rest of the codebase never touches aiohttp session details.

Public API
----------
    fetch(session, url, *, timeout, max_retries, cache, hooks) → FetchResult
    download_file(session, url, dest) → bool
    random_user_agent() → str
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import aiohttp

from .config import (
    DEFAULT_TIMEOUT,
    JS_DOWNLOAD_TIMEOUT,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    RETRYABLE_STATUS,
    USER_AGENTS,
)
from .exceptions import FetchError
from .http_cache import CacheEntry, HTTPCache
from .netguard import describe_block

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["FetchResult", "download_file", "fetch", "random_user_agent"]

#: Statuses that mean "the real page is at the Location header".
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

#: Redirect hops allowed before a fetch is called a loop.
MAX_REDIRECTS = 5


@dataclass
class FetchResult:
    """A successfully fetched page: its text, byte size, and HTTP status."""

    text: str
    nbytes: int
    status: int = 200


def random_user_agent() -> str:
    """Return a random User-Agent from the rotation pool."""
    return random.choice(USER_AGENTS)


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter for retry delays."""
    delay: float = RETRY_BACKOFF_BASE * (2**attempt) + random.uniform(0.0, 0.5)
    return delay


def _conditional_headers(entry: CacheEntry | None) -> dict[str, str]:
    """
    Build revalidation headers from an entry's validators.

    Only ETag/Last-Modified survive serialisation, so those are the only
    validators available to revalidate with. No entry means no headers, which
    leaves the request an ordinary one.
    """
    if entry is None:
        return {}
    headers: dict[str, str] = {}
    if entry.etag:
        headers["If-None-Match"] = entry.etag
    if entry.last_modified:
        headers["If-Modified-Since"] = entry.last_modified
    return headers


def _from_cache(entry: CacheEntry) -> FetchResult:
    """
    Build a result from a cache hit.

    ``nbytes`` reflects the body actually served, not the bytes transferred.
    Reporting 0 made every cached page render as ``—`` in the results table and
    ``0 B total`` in the summary, which reads as "nothing was scraped".
    """
    return FetchResult(
        text=entry.body,
        nbytes=len(entry.body.encode("utf-8")),
        status=entry.status,
    )


async def fetch(
    session: aiohttp.ClientSession,
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    cache: HTTPCache | None = None,
    hooks: dict[str, list[Callable[..., Any]]] | None = None,
    allow_internal_redirects: bool = False,
) -> FetchResult:
    """
    Fetch *url* with retry logic and conditional caching.

    Each request carries a fresh User-Agent from the rotation pool, so callers
    never mutate a shared session to rotate UAs.

    Raises FetchError on HTTP >= 400 or connection problems.
    """
    # Read the entry without discarding it: an expired one still carries the
    # validators that make a conditional request — and a 304 — possible at all.
    entry = cache.entry_for(url) if cache is not None else None
    if entry is not None and not entry.is_expired:
        body = entry.body
        if entry.nbytes and len(body.encode("utf-8")) != entry.nbytes:
            # The index says this entry holds a body, but reading it produced a
            # different amount — the file was deleted or truncated behind our
            # back. Serving that as a hit returned an empty page marked
            # successful, which is worse than re-fetching. Fall through and get
            # the real content.
            entry = None
        else:
            return _from_cache(entry)

    hook_ctx: dict[str, Any] = {"url": url, "headers": {}}
    for hook in (hooks or {}).get("before_fetch", []):
        with contextlib.suppress(Exception):
            hook(url, hook_ctx)

    headers = {**_conditional_headers(entry), "User-Agent": random_user_agent()}
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            r = await _get_following(
                session,
                url,
                headers=headers,
                timeout=timeout,
                allow_internal_redirects=allow_internal_redirects,
            )
            try:
                if r.status == 304:
                    # 304 means "what you already have is current". Serving it
                    # needs a cache; without one there is nothing to serve, and
                    # passing the empty body off as a page would report a blank
                    # site as successfully scraped.
                    if cache is None:
                        raise FetchError(url, "304 Not Modified with no cache entry")
                    # The entry a 304 refers to is, by definition, the one that
                    # was stale — which get() will not return.
                    served = cache.entry_for(url)
                    if served is None:
                        raise FetchError(url, "304 Not Modified with no cache entry")
                    # Refresh it so the next visit is a disk hit, not another
                    # round trip to re-validate the same unchanged page.
                    cache.touch(url)
                    return _from_cache(served)
                if r.status >= 400:
                    if r.status in RETRYABLE_STATUS and attempt < max_retries - 1:
                        await asyncio.sleep(_backoff(attempt))
                        continue
                    raise FetchError(url, f"HTTP {r.status}")
                data = await r.read()
                text = data.decode("utf-8", errors="replace")
                if cache:
                    cache.put(
                        url,
                        CacheEntry(
                            etag=r.headers.get("ETag"),
                            last_modified=r.headers.get("Last-Modified"),
                            body=text,
                            status=r.status,
                        ),
                    )
                for hook in (hooks or {}).get("after_fetch", []):
                    with contextlib.suppress(Exception):
                        hook(url, {"status": r.status, "body": text})
                return FetchResult(text=text, nbytes=len(data), status=r.status)
        except TimeoutError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt))
                continue
            raise FetchError(url, "timeout") from exc
        except aiohttp.ClientError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt))
                continue
            raise FetchError(url, str(exc)) from exc

    raise FetchError(url, f"failed after {max_retries} retries") from last_exc


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    dest: str | Path,
) -> bool:
    """Download *url* to *dest* (best-effort). Returns True on success."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=JS_DOWNLOAD_TIMEOUT)) as r:
            if r.status == 200:
                dest.write_bytes(await r.read())
                return True
    except Exception:
        pass
    return False
