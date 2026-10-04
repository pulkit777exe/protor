"""Per-domain rate limiting for polite scraping."""

from __future__ import annotations

import asyncio
import time


class DomainRateLimiter:
    """
    Enforce a minimum delay between requests to the same domain.

    Concurrency-safe: each waiter reserves its own slot under a per-domain
    lock. The naive read-then-sleep version let every concurrent task read the
    same stale timestamp, sleep the same amount, and then fire simultaneously,
    so ``concurrency=8`` against one domain produced 8 simultaneous requests
    instead of 8 spaced ones. Requests to *different* domains never block
    each other.

    Bounded, in two different ways because the two dicts have different lifetimes.
    The lock goes as soon as the last waiter for a domain leaves — it is only ever
    needed while somebody is inside :meth:`wait`. The timestamp cannot: it is the
    thing that enforces the delay, and it stays meaningful until its slot passes,
    so it is swept lazily once the map outgrows a small bound.

    Both used to grow for the life of the instance, one lock and one timestamp per
    host ever seen, and nothing ever removed them. On a crawl of 40,000 pages
    across 40,000 hosts that is 80,000 permanent entries of bookkeeping.

    The bound this gives is the number of hosts seen *within one delay window*,
    not the number ever seen — and that is the right bound, because a slot still in
    the future is the only thing standing between two requests to the same host.
    For a crawl, which follows one domain, it is 1; for a batch, it is the domains
    in flight.

    Sweeping rather than dropping on release is also what keeps it behaviour-
    preserving: a swept entry reads as ``next_allowed = 0``, and ``max(now, 0)`` is
    ``now``, so an expired entry is indistinguishable from an absent one.
    """

    #: Timestamps are swept once there are more than this, so the map holds the
    #: hosts in a delay window rather than every host ever seen. Small on purpose:
    #: sweeping is a full scan of a dict that is almost always tiny.
    _SWEEP_AT = 32

    def __init__(self, delay: float = 0.5) -> None:
        self._delay = delay
        self._next_allowed: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        #: Tasks currently inside :meth:`wait` for each domain. Plain dict
        #: arithmetic is atomic here because there is no ``await`` between the
        #: read and the write, so this needs no lock of its own.
        self._waiters: dict[str, int] = {}

    async def wait(self, domain: str) -> None:
        """Block until *domain* has been idle for the configured delay."""
        if self._delay <= 0:
            return

        self._waiters[domain] = self._waiters.get(domain, 0) + 1
        try:
            # setdefault, not defaultdict: a default factory here would insert on
            # read, so a lookup could never fail to leave an entry behind.
            lock = self._locks.setdefault(domain, asyncio.Lock())
            async with lock:
                now = time.monotonic()
                start = max(now, self._next_allowed.get(domain, 0.0))
                # Reserve this slot before releasing the lock, so the next waiter
                # queues behind us instead of racing to the same timestamp.
                self._next_allowed[domain] = start + self._delay
                delay = start - now
                if delay > 0:
                    await asyncio.sleep(delay)
        finally:
            self._release(domain)
        self._sweep()

    def _release(self, domain: str) -> None:
        """Drop *domain*'s lock once nothing is waiting on it."""
        remaining = self._waiters.get(domain, 0) - 1
        if remaining > 0:
            self._waiters[domain] = remaining
            return
        self._waiters.pop(domain, None)
        self._locks.pop(domain, None)

    def _sweep(self) -> None:
        """Forget timestamps whose slot has passed. Amortised against the bound."""
        if len(self._next_allowed) <= self._SWEEP_AT:
            return
        now = time.monotonic()
        self._next_allowed = {domain: at for domain, at in self._next_allowed.items() if at > now}

    def tracked_domains(self) -> int:
        """Domains currently holding state. For tests, and for a sanity check."""
        return len(self._locks) | len(self._next_allowed)
