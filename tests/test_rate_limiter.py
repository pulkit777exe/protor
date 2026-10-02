"""Tests for protor.rate_limiter module."""

import asyncio
import time

import pytest

from protor.rate_limiter import DomainRateLimiter


class TestDomainRateLimiter:
    def test_first_request_no_delay(self):
        limiter = DomainRateLimiter(delay=0.1)
        start = time.monotonic()
        asyncio.run(limiter.wait("example.com"))
        elapsed = time.monotonic() - start
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_second_request_waits(self):
        limiter = DomainRateLimiter(delay=0.2)
        await limiter.wait("example.com")
        start = time.monotonic()
        await limiter.wait("example.com")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15

    @pytest.mark.asyncio
    async def test_concurrent_requests_to_one_domain_are_spaced(self):
        """A read-then-sleep race let every waiter fire at the same instant."""
        limiter = DomainRateLimiter(delay=0.1)
        fired: list[float] = []

        async def hit() -> None:
            await limiter.wait("example.com")
            fired.append(time.monotonic())

        await asyncio.gather(*[hit() for _ in range(5)])

        gaps = [b - a for a, b in zip(sorted(fired), sorted(fired)[1:], strict=False)]
        assert all(gap >= 0.09 for gap in gaps), f"requests bunched up: {gaps}"

    @pytest.mark.asyncio
    async def test_zero_delay_is_a_no_op(self):
        limiter = DomainRateLimiter(delay=0)
        await asyncio.gather(*[limiter.wait("example.com") for _ in range(5)])

    @pytest.mark.asyncio
    async def test_different_domains_do_not_block_each_other(self):
        limiter = DomainRateLimiter(delay=5.0)
        start = time.monotonic()
        await asyncio.gather(*[limiter.wait(f"site{i}.com") for i in range(5)])
        assert time.monotonic() - start < 1.0

    @pytest.mark.asyncio
    async def test_different_domains_no_delay(self):
        limiter = DomainRateLimiter(delay=0.5)
        await limiter.wait("domain-a.com")
        start = time.monotonic()
        await limiter.wait("domain-b.com")
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_delay_exceeds_waits_correctly(self):
        limiter = DomainRateLimiter(delay=0.1)
        await limiter.wait("example.com")
        await asyncio.sleep(0.15)
        start = time.monotonic()
        await limiter.wait("example.com")
        elapsed = time.monotonic() - start
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_custom_delay(self):
        limiter = DomainRateLimiter(delay=0.05)
        await limiter.wait("example.com")
        start = time.monotonic()
        await limiter.wait("example.com")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04
