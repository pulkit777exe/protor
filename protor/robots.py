"""robots.txt parser and checker for polite scraping."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from .config import DEFAULT_TIMEOUT, HEADERS

if TYPE_CHECKING:
    import aiohttp

_cache: dict[str, RobotFileParser] = {}


async def _fetch_robots(
    base_url: str, session: aiohttp.ClientSession | None = None
) -> RobotFileParser:
    """Fetch and parse robots.txt for *base_url*, caching the result."""
    if base_url in _cache:
        return _cache[base_url]

    rp = RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")

    if session:
        try:
            async with session.get(robots_url, timeout=DEFAULT_TIMEOUT) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    rp.parse(text.splitlines())
        except Exception:
            pass
    else:
        try:
            rp.set_url(robots_url)
            rp.read()
        except Exception:
            pass

    _cache[base_url] = rp
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
    """Clear the robots.txt cache."""
    _cache.clear()
