"""Tests for protor.fetcher retry, backoff, and hook behaviour.

Retry logic is where failures hide: it only runs when something has already
gone wrong, so it is exactly the code least likely to be exercised by hand.
"""

import asyncio

import pytest

from protor.exceptions import FetchError
from protor.fetcher import _backoff, download_file, fetch
from protor.http_cache import CacheEntry, HTTPCache


class RecordingSession:
    """Session returning a scripted sequence of responses/exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        item = self.script.pop(0) if self.script else {"status": 200, "body": ""}
        if isinstance(item, Exception):
            raise item
        return _Resp(item)


class _Resp:
    def __init__(self, spec):
        self.status = spec.get("status", 200)
        self.headers = spec.get("headers", {})
        self._body = spec.get("body", "")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def read(self):
        return self._body.encode() if isinstance(self._body, str) else self._body

    async def text(self):
        return self._body if isinstance(self._body, str) else self._body.decode()


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Backoff sleeps must not make the suite slow."""
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return slept


def _expire(cache, url):
    """
    Age a stored entry past its TTL.

    `put` stamps the current time and the cache's own TTL, so the only way to
    reach the revalidation path is to move the timestamp back. Same idiom as the
    cache-hygiene tests in test_regressions.py.
    """
    cache._index[url].timestamp -= 7200


class TestSuccessfulFetch:
    @pytest.mark.asyncio
    async def test_returns_text_and_byte_count(self):
        session = RecordingSession([{"status": 200, "body": "<html>hi</html>"}])
        result = await fetch(session, "https://x.com/")
        assert result.text == "<html>hi</html>"
        assert result.nbytes == len("<html>hi</html>")
        assert result.status == 200

    @pytest.mark.asyncio
    async def test_sends_a_rotated_user_agent(self):
        session = RecordingSession([{"status": 200, "body": "x"}])
        await fetch(session, "https://x.com/")
        assert "User-Agent" in session.calls[0]["headers"]

    @pytest.mark.asyncio
    async def test_timeout_is_configured_per_request(self):
        session = RecordingSession([{"status": 200, "body": "x"}])
        await fetch(session, "https://x.com/", timeout=7)
        assert session.calls[0]["timeout"].total == 7

    @pytest.mark.asyncio
    async def test_invalid_utf8_is_replaced_not_fatal(self):
        session = RecordingSession([{"status": 200, "body": b"\xff\xfe bad bytes"}])
        result = await fetch(session, "https://x.com/")
        assert "bad bytes" in result.text


class TestRetryBehaviour:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    async def test_retryable_statuses_are_retried(self, status, no_real_sleep):
        session = RecordingSession(
            [{"status": status}, {"status": status}, {"status": 200, "body": "recovered"}]
        )
        result = await fetch(session, "https://x.com/")
        assert result.text == "recovered"
        assert len(session.calls) == 3
        assert len(no_real_sleep) == 2, "should have backed off between attempts"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
    async def test_client_errors_are_not_retried(self, status, no_real_sleep):
        session = RecordingSession([{"status": status}])
        with pytest.raises(FetchError):
            await fetch(session, "https://x.com/")
        assert len(session.calls) == 1, "4xx must fail fast"
        assert not no_real_sleep

    @pytest.mark.asyncio
    async def test_retries_are_capped(self, no_real_sleep):
        session = RecordingSession([{"status": 503}] * 6)
        with pytest.raises(FetchError):
            await fetch(session, "https://x.com/", max_retries=3)
        assert len(session.calls) == 3

    @pytest.mark.asyncio
    async def test_connection_errors_are_retried(self, no_real_sleep):
        import aiohttp

        session = RecordingSession(
            [aiohttp.ClientConnectionError("refused"), {"status": 200, "body": "ok"}]
        )
        assert (await fetch(session, "https://x.com/")).text == "ok"
        assert len(session.calls) == 2

    @pytest.mark.asyncio
    async def test_timeout_is_retried_then_reported(self, no_real_sleep):
        session = RecordingSession([TimeoutError()] * 3)
        with pytest.raises(FetchError, match="timeout"):
            await fetch(session, "https://x.com/", max_retries=2)
        assert len(session.calls) == 2

    @pytest.mark.asyncio
    async def test_exhausted_retries_report_the_last_error(self, no_real_sleep):
        import aiohttp

        session = RecordingSession([aiohttp.ClientConnectionError("nope")] * 3)
        with pytest.raises(FetchError, match="nope"):
            await fetch(session, "https://x.com/", max_retries=2)


class TestBackoff:
    def test_grows_exponentially(self):
        d0, d1, d2 = _backoff(0), _backoff(1), _backoff(2)
        assert d1 > d0
        assert d2 > d1

    def test_includes_jitter_within_bounds(self):
        base = 0.5
        for attempt in range(6):
            delay = _backoff(attempt)
            assert base * (2**attempt) <= delay <= base * (2**attempt) + 0.5


class TestHooks:
    @pytest.mark.asyncio
    async def test_before_fetch_hook_runs(self):
        seen: list[str] = []
        session = RecordingSession([{"status": 200, "body": "x"}])
        await fetch(
            session, "https://x.com/", hooks={"before_fetch": [lambda u, ctx: seen.append(u)]}
        )
        assert seen == ["https://x.com/"]

    @pytest.mark.asyncio
    async def test_after_fetch_hook_receives_the_body(self):
        got: list[str] = []
        session = RecordingSession([{"status": 200, "body": "payload"}])
        await fetch(
            session,
            "https://x.com/",
            hooks={"after_fetch": [lambda u, ctx: got.append(ctx["body"])]},
        )
        assert got == ["payload"]

    @pytest.mark.asyncio
    async def test_a_failing_hook_does_not_abort_the_fetch(self):
        """Hooks are user code; one raising must not lose the page."""

        def boom(url, ctx):
            raise RuntimeError("bad hook")

        session = RecordingSession([{"status": 200, "body": "survived"}])
        result = await fetch(
            session, "https://x.com/", hooks={"before_fetch": [boom], "after_fetch": [boom]}
        )
        assert result.text == "survived"


class TestConditionalRequests:
    @pytest.mark.asyncio
    async def test_etag_is_replayed_and_304_serves_the_cache(self, tmp_path):
        cache = HTTPCache(cache_dir=tmp_path / "c")
        cache.put("https://x.com/", CacheEntry(body="original", etag='W/"abc"'))
        _expire(cache, "https://x.com/")
        session = RecordingSession([{"status": 304}])
        result = await fetch(session, "https://x.com/", cache=cache)
        assert result.text == "original"
        assert session.calls[0]["headers"]["If-None-Match"] == 'W/"abc"'

    @pytest.mark.asyncio
    async def test_last_modified_is_replayed(self, tmp_path):
        cache = HTTPCache(cache_dir=tmp_path / "c")
        cache.put("https://x.com/", CacheEntry(body="v1", last_modified="Mon, 01 Jan 2024"))
        _expire(cache, "https://x.com/")
        session = RecordingSession([{"status": 304}])
        await fetch(session, "https://x.com/", cache=cache)
        assert session.calls[0]["headers"]["If-Modified-Since"] == "Mon, 01 Jan 2024"

    @pytest.mark.asyncio
    async def test_304_refreshes_the_entry(self, tmp_path):
        """After a 304 the entry is fresh again, so the next visit costs nothing."""
        cache = HTTPCache(cache_dir=tmp_path / "c")
        cache.put("https://x.com/", CacheEntry(body="original", etag='W/"abc"'))
        _expire(cache, "https://x.com/")
        await fetch(RecordingSession([{"status": 304}]), "https://x.com/", cache=cache)
        session = RecordingSession([])
        result = await fetch(session, "https://x.com/", cache=cache)
        assert session.calls == [], "a refreshed entry must not be revalidated again"
        assert result.text == "original"

    @pytest.mark.asyncio
    async def test_fresh_entry_is_served_without_a_request(self, tmp_path):
        """The point of a TTL: inside it, nothing goes over the network."""
        cache = HTTPCache(cache_dir=tmp_path / "c")
        cache.put("https://x.com/", CacheEntry(body="cached", etag='W/"abc"'))
        session = RecordingSession([])
        result = await fetch(session, "https://x.com/", cache=cache)
        assert session.calls == []
        assert result.text == "cached"

    @pytest.mark.asyncio
    async def test_304_without_a_cache_entry_is_an_error(self):
        """A 304 we cannot satisfy must not be passed off as an empty page."""
        session = RecordingSession([{"status": 304}])
        with pytest.raises(FetchError, match="304"):
            await fetch(session, "https://x.com/")

    @pytest.mark.asyncio
    async def test_fresh_response_populates_the_cache(self, tmp_path):
        cache = HTTPCache(cache_dir=tmp_path / "c")
        session = RecordingSession([{"status": 200, "body": "fresh", "headers": {"ETag": 'W/"1"'}}])
        await fetch(session, "https://x.com/", cache=cache)
        entry = cache.get("https://x.com/")
        assert entry.body == "fresh"
        assert entry.etag == 'W/"1"'


class TestDownloadFile:
    @pytest.mark.asyncio
    async def test_writes_the_body_on_200(self, tmp_path):
        session = RecordingSession([{"status": 200, "body": "console.log(1)"}])
        dest = tmp_path / "a.js"
        assert await download_file(session, "https://x.com/a.js", dest) is True
        assert dest.read_text() == "console.log(1)"

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, tmp_path):
        session = RecordingSession([{"status": 200, "body": "x"}])
        dest = tmp_path / "deep" / "nested" / "a.js"
        assert await download_file(session, "https://x.com/a.js", dest) is True
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_non_200_leaves_no_file(self, tmp_path):
        session = RecordingSession([{"status": 404}])
        dest = tmp_path / "a.js"
        assert await download_file(session, "https://x.com/a.js", dest) is False
        assert not dest.exists()

    @pytest.mark.asyncio
    async def test_transport_error_is_swallowed(self, tmp_path):
        session = RecordingSession([ConnectionError("refused")])
        assert await download_file(session, "https://x.com/a.js", tmp_path / "a.js") is False
