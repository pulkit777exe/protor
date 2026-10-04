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


class TestTheLimiterIsBounded:
    """
    Both dicts grew for the life of the instance and nothing ever removed them.

    One lock and one timestamp per host ever seen, kept forever, to remember — in
    the end — that a domain nobody has visited recently should not be delayed at
    all. A crawl of 40,000 pages across 40,000 hosts is 80,000 permanent entries.
    """

    async def test_a_lock_is_released_as_soon_as_the_last_waiter_leaves(self):
        limiter = DomainRateLimiter(delay=0.01)
        await limiter.wait("a.example")
        assert limiter._locks == {}, "the lock outlived its only waiter"
        # The timestamp stays: its slot is 10ms out and it is what enforces the delay.
        assert limiter.tracked_domains() == 1

    async def test_hosts_that_leave_the_window_stop_being_remembered(self):
        """
        500 distinct hosts, each seen once — then the window they were seen in passes.

        All 500 are legitimately live while that window is open, and an earlier
        version of this test asserted they were not, which was wrong: a slot still in
        the future is what stops two requests to the same host from firing together,
        and 500 hosts hit in one millisecond really do have 500 live slots. What must
        not happen is remembering them *afterwards*, which is the difference between
        "hosts in a delay window" and "hosts ever seen".
        """
        limiter = DomainRateLimiter(delay=0.05)
        for i in range(500):
            await limiter.wait(f"host{i}.example")

        await asyncio.sleep(0.1)  # the window every one of them was seen in has passed
        await limiter.wait("unrelated.example")  # one more request sweeps

        assert limiter.tracked_domains() <= 2, (
            f"{limiter.tracked_domains()} entries kept for hosts nobody will revisit"
        )

    async def test_a_live_window_is_retained(self):
        """The complement: nothing is swept while its slot is still in the future."""
        limiter = DomainRateLimiter(delay=30.0)
        for i in range(100):
            await limiter.wait(f"host{i}.example")
        assert limiter.tracked_domains() == 100, "live slots were swept"

    async def test_a_recently_used_domain_is_not_swept(self):
        """
        The sweep must not cost a delay.

        A timestamp still in the future is the only thing stopping two requests to
        the same host from firing together, so sweeping one early would quietly turn
        the limiter off for exactly the case it exists for.
        """
        limiter = DomainRateLimiter(delay=5.0)
        for i in range(200):  # past the sweep bound
            await limiter.wait(f"host{i}.example")
        assert "host199.example" in limiter._next_allowed
        assert limiter.tracked_domains() > 1, "everything was swept"

    async def test_a_slot_in_the_future_is_kept(self):
        """
        The timestamp is the thing that enforces the delay, so it cannot go early.

        Dropping it the moment the last waiter leaves would let the next request
        straight through — turning the limiter into a no-op for sequential crawls,
        which is the case it exists for.
        """
        limiter = DomainRateLimiter(delay=5.0)
        await limiter.wait("slow.example")
        assert "slow.example" in limiter._next_allowed

    async def test_concurrent_waiters_keep_the_entry_alive(self):
        """
        The lock is dropped only when the last waiter leaves.

        Dropping it earlier would hand a second concurrent request a fresh lock, and
        the two would no longer queue behind each other — the exact race the lock
        was introduced to remove.
        """
        limiter = DomainRateLimiter(delay=0.05)
        order: list[str] = []

        async def hit(name: str) -> None:
            await limiter.wait("busy.example")
            order.append(name)

        await asyncio.gather(hit("a"), hit("b"), hit("c"))

        assert len(order) == 3
        assert limiter.tracked_domains() == 1, "released while slots were still reserved"

    async def test_eviction_does_not_change_the_spacing(self):
        """
        The reason eviction is safe: a dropped entry reads as `next_allowed = 0`,
        and `max(now, 0)` is `now`. An absent entry and a reached one must be
        indistinguishable — measured, not argued.
        """
        limiter = DomainRateLimiter(delay=0.1)
        for _ in range(3):
            await limiter.wait("x.example")
            await asyncio.sleep(0.12)  # long enough for the slot to pass

        start = time.monotonic()
        await limiter.wait("x.example")
        elapsed = time.monotonic() - start
        assert elapsed < 0.05, f"a stale entry delayed a request by {elapsed * 1000:.0f}ms"

    async def test_a_delay_of_zero_tracks_nothing(self):
        limiter = DomainRateLimiter(delay=0)
        await limiter.wait("a.example")
        assert limiter.tracked_domains() == 0
