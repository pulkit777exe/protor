"""Per-domain rate limiting for polite scraping."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict


class DomainRateLimiter:
    """
    Enforce a minimum delay between requests to the same domain.

    Concurrency-safe: each waiter reserves its own slot under a per-domain
    lock. The naive read-then-sleep version let every concurrent task read the
    same stale timestamp, sleep the same amount, and then fire simultaneously,
    so ``concurrency=8`` against one domain produced 8 simultaneous requests
    instead of 8 spaced ones. Requests to *different* domains never block
    each other.
    """

    def __init__(self, delay: float = 0.5) -> None:
        self._delay = delay
        self._next_allowed: dict[str, float] = defaultdict(float)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def wait(self, domain: str) -> None:
        """Block until *domain* has been idle for the configured delay."""
        if self._delay <= 0:
            return

        async with self._locks[domain]:
            now = time.monotonic()
            start = max(now, self._next_allowed[domain])
            # Reserve this slot before releasing the lock, so the next waiter
            # queues behind us instead of racing to the same timestamp.
            self._next_allowed[domain] = start + self._delay
            delay = start - now
            if delay > 0:
                await asyncio.sleep(delay)
