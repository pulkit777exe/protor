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
