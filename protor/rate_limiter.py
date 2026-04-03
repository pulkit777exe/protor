"""Per-domain rate limiter for polite scraping."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict


class DomainRateLimiter:
    """Enforce minimum delay between requests to the same domain."""

    def __init__(self, delay: float = 0.5) -> None:
        self._delay = delay
        self._last_request: dict[str, float] = defaultdict(float)

    async def wait(self, domain: str) -> None:
        """Sleep if the last request to *domain* was too recent."""
        elapsed = time.monotonic() - self._last_request[domain]
        if elapsed < self._delay:
            await asyncio.sleep(self._delay - elapsed)
        self._last_request[domain] = time.monotonic()
