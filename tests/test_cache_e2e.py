"""End-to-end test for conditional caching against a real HTTP server.

Everything else about the cache is tested against stubs. That is how a 304 path
can look covered while the round trip is broken in the one place it matters: the
fetch loop has to send the validator the cache stored, notice the server's 304,
serve the stored body, and refresh the entry so the *next* run costs nothing.

The server here implements RFC 9110 revalidation honestly (ETag plus
If-None-Match -> 304, and a body that changes when the ETag does), so a bug in
any of those steps fails here rather than in production.
"""

from __future__ import annotations

import pytest
from aiohttp import web

from protor.fetcher import fetch
from protor.http_cache import HTTPCache


class _Server:
    """A page whose ETag changes when its body does, counting every request."""

    def __init__(self) -> None:
        self.body = "<html><body><h1>Version 1</h1><p>Original content.</p></body></html>"
        self.etag = '"v1"'
        self.requests: list[str] = []
        self.conditional_hits = 0

    @property
    def handler(self):
        async def handle(request: web.Request) -> web.Response:
            self.requests.append(request.headers.get("If-None-Match", ""))
            if request.headers.get("If-None-Match") == self.etag:
                self.conditional_hits += 1
                return web.Response(status=304, headers={"ETag": self.etag})
            return web.Response(
                text=self.body,
                headers={"ETag": self.etag},
                content_type="text/html",
            )

        return handle

    def update(self, body: str) -> None:
        self.body = body
        self.etag = '"v2"'


@pytest.fixture
async def site():
    """Run a real HTTP server for the duration of a test."""
    state = _Server()
    app = web.Application()
    app.router.add_get("/", state.handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site_ = web.TCPSite(runner, "127.0.0.1", 0)
    await site_.start()
    port = runner.addresses[0][1]
    try:
        yield f"http://127.0.0.1:{port}/", state
    finally:
        await runner.cleanup()


async def _get(url: str, cache: HTTPCache, ttl: int = 3600) -> str:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        result = await fetch(session, url, cache=cache, timeout=10)
        return result.text


def _expire(cache: HTTPCache, url: str) -> None:
    cache._index[url].timestamp -= 7200


class TestConditionalRoundTrip:
    @pytest.mark.asyncio
    async def test_second_run_revalidates_and_serves_the_stored_body(self, site, tmp_path):
        url, state = site
        cache = HTTPCache(cache_dir=tmp_path / "c", ttl=1)

        first = await _get(url, cache)
        assert "Version 1" in first
        assert state.conditional_hits == 0, "nothing to revalidate against yet"

        # The TTL expires: the next fetch must ask, and must be told "not
        # modified" rather than re-downloading the page.
        _expire(cache, url)
        second = await _get(url, cache)
        assert second == first, "a 304 must serve the stored body verbatim"
        assert state.conditional_hits == 1
        assert state.requests[-1] == '"v1"', "the stored ETag must be replayed"

    @pytest.mark.asyncio
    async def test_a_304_leaves_the_entry_fresh_so_the_next_run_costs_nothing(self, site, tmp_path):
        url, state = site
        cache = HTTPCache(cache_dir=tmp_path / "c", ttl=1)

        await _get(url, cache)
        _expire(cache, url)
        await _get(url, cache)

        before = len(state.requests)
        third = await _get(url, cache)
        assert "Version 1" in third
        assert len(state.requests) == before, "a refreshed entry must not hit the network"

    @pytest.mark.asyncio
    async def test_a_changed_page_replaces_the_stored_body(self, site, tmp_path):
        url, state = site
        cache = HTTPCache(cache_dir=tmp_path / "c", ttl=1)

        await _get(url, cache)
        state.update("<html><body><h1>Version 2</h1><p>Fresh content.</p></body></html>")
        _expire(cache, url)

        updated = await _get(url, cache)
        assert "Version 2" in updated, "a changed page must not be served from cache"
        assert "Version 1" not in updated

    @pytest.mark.asyncio
    async def test_a_fresh_entry_is_served_without_any_request(self, site, tmp_path):
        url, state = site
        cache = HTTPCache(cache_dir=tmp_path / "c", ttl=3600)

        await _get(url, cache)
        before = len(state.requests)
        assert "Version 1" in await _get(url, cache)
        assert len(state.requests) == before, "inside the TTL there is no traffic"

    @pytest.mark.asyncio
    async def test_the_cache_survives_reopening(self, site, tmp_path):
        """The validator has to be reloaded from disk, not just remembered."""
        url, state = site
        cache_dir = tmp_path / "c"

        cache = HTTPCache(cache_dir=cache_dir, ttl=1)
        await _get(url, cache)
        _expire(cache, url)
        cache.flush()
        cache.close()

        reopened = HTTPCache(cache_dir=cache_dir, ttl=1)
        text = await _get(url, reopened)
        assert "Version 1" in text
        assert state.conditional_hits == 1, "the reloaded ETag must still revalidate"

    @pytest.mark.asyncio
    async def test_repeated_runs_do_not_grow_the_body_on_disk(self, site, tmp_path):
        """Re-fetching must overwrite one body, not accumulate copies."""
        url, _state = site
        cache_dir = tmp_path / "c"
        cache = HTTPCache(cache_dir=cache_dir, ttl=1)

        for _ in range(4):
            await _get(url, cache)
            _expire(cache, url)
        cache.flush()

        assert len(list((cache_dir / "bodies").glob("*.body"))) == 1

    @pytest.mark.asyncio
    async def test_an_entry_whose_body_vanished_is_dropped_not_served_empty(self, site, tmp_path):
        """
        A body deleted out from under the index must not come back as a blank
        page — that would be a successful scrape reporting zero content.

        Dropping the guard makes every test below it pass, which is why this one
        exists: no other round trip loses a body mid-flight.
        """
        url, _state = site
        cache_dir = tmp_path / "c"

        cache = HTTPCache(cache_dir=cache_dir, ttl=3600)
        await _get(url, cache)
        cache.flush()
        body = cache._body_path(url)
        cache.close()

        body.unlink()
        reopened = HTTPCache(cache_dir=cache_dir, ttl=3600)
        assert url not in reopened._index
        assert reopened.get(url) is None

        # And a fetch with that cache must go to the network, not return "".
        assert "Version 1" in await _get(url, reopened)


class TestNoCacheConfigured:
    @pytest.mark.asyncio
    async def test_a_304_without_a_cache_is_an_error(self):
        """
        A 304 we cannot satisfy must not become an empty page reported as a
        successful scrape. Only a real server can produce this: with no cache,
        protor never sends a validator, so a stubbed session would have to lie
        about the response.
        """
        import aiohttp

        from protor.exceptions import FetchError

        async def always_304(_request: web.Request) -> web.Response:
            return web.Response(status=304)

        app = web.Application()
        app.router.add_get("/", always_304)
        runner = web.AppRunner(app)
        await runner.setup()
        site_ = web.TCPSite(runner, "127.0.0.1", 0)
        await site_.start()
        port = runner.addresses[0][1]
        try:
            async with aiohttp.ClientSession() as session:
                with pytest.raises(FetchError, match="304"):
                    await fetch(session, f"http://127.0.0.1:{port}/", cache=None, timeout=10)
        finally:
            await runner.cleanup()
