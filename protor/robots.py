"""robots.txt parser and checker for polite scraping."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from .config import DEFAULT_TIMEOUT, HEADERS

if TYPE_CHECKING:
    import aiohttp

_cache: dict[str, RobotFileParser] = {}
# In-flight fetches, so concurrent page checks for one domain share a single
# request. Without this the cache check happened before the await and every
# concurrent worker missed it: 6 concurrent pages meant 6 robots.txt fetches.
_inflight: dict[str, asyncio.Task[RobotFileParser]] = {}


def _allow_all(rp: RobotFileParser) -> None:
    """Mark *rp* as an allow-everything policy.

    urllib's RobotFileParser defaults to denying when it has not been
    populated, but a missing/unreadable robots.txt means "allow all".
    """
    rp.allow_all = True  # type: ignore[attr-defined]


async def _load_robots(
    base_url: str, session: aiohttp.ClientSession | None = None
) -> RobotFileParser:
    """Fetch and parse robots.txt for *base_url* (no caching)."""
    rp = RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")

    if session:
        try:
            import aiohttp as _aiohttp

            timeout_obj = _aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
            async with session.get(robots_url, timeout=timeout_obj) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    rp.parse(text.splitlines())
                else:
                    # No (or non-OK) robots.txt means "allow everything".
                    _allow_all(rp)
        except Exception:
            _allow_all(rp)
    else:
        try:
            rp.set_url(robots_url)
            rp.read()
        except Exception:
            _allow_all(rp)

    return rp


async def _fetch_robots(
    base_url: str, session: aiohttp.ClientSession | None = None
) -> RobotFileParser:
    """
    Fetch and parse robots.txt for *base_url*, caching the result.

    Concurrent callers for the same domain share one in-flight request, so a
    crawl that starts N pages at once fetches robots.txt once.
    """
    cached = _cache.get(base_url)
    if cached is not None:
        return cached

    task = _inflight.get(base_url)
    if task is None:
        task = asyncio.ensure_future(_load_robots(base_url, session))
        _inflight[base_url] = task

    try:
        rp = await asyncio.shield(task)
    except Exception:
        rp = await _allow_all_parser()
    finally:
        if _inflight.get(base_url) is task:
            del _inflight[base_url]

    _cache[base_url] = rp
    return rp


async def _allow_all_parser() -> RobotFileParser:
    """Return an allow-everything parser (used if the shared task fails)."""
    rp = RobotFileParser()
    _allow_all(rp)
    return rp


def is_allowed(url: str, user_agent: str = "*") -> bool:
    """Check if *url* is allowed by robots.txt.

    Returns True if allowed or if robots.txt couldn't be fetched.
    """
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = _cache.get(base)
    if rp is None:
        return True
    return rp.can_fetch(user_agent, url)


async def check_robots(url: str, session: aiohttp.ClientSession) -> bool:
    """Fetch robots.txt for *url*'s domain and check if *url* is allowed.

    Returns True if allowed or if robots.txt couldn't be fetched.
    """
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = await _fetch_robots(base, session)
    user_agent = HEADERS.get("User-Agent", "*")
    return rp.can_fetch(user_agent, url)


def clear_cache() -> None:
    """Clear the robots.txt cache and drop any in-flight fetches."""
    _cache.clear()
    for task in _inflight.values():
        task.cancel()
    _inflight.clear()
