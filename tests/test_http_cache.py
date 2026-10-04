"""Tests for protor.http_cache module."""

import json
import pathlib
import tempfile
import time

import pytest

from protor.http_cache import CacheEntry, HTTPCache


class TestCacheEntry:
    def test_defaults(self):
        entry = CacheEntry()
        assert entry.etag is None
        assert entry.last_modified is None
        assert entry.body == ""
        assert entry.status == 200
        assert entry.ttl == 3600

    def test_is_expired_false(self):
        entry = CacheEntry(timestamp=time.time(), ttl=3600)
        assert entry.is_expired is False

    def test_is_expired_true(self):
        entry = CacheEntry(timestamp=time.time() - 7200, ttl=3600)
        assert entry.is_expired is True

    def test_to_dict(self):
        """to_dict carries metadata only; bodies live in their own files."""
        entry = CacheEntry(etag="abc123", body="hello", status=200, timestamp=1000.0, ttl=600)
        d = entry.to_dict()
        assert d["etag"] == "abc123"
        assert d["status"] == 200
        assert d["timestamp"] == 1000.0
        assert d["ttl"] == 600
        assert "body" not in d

    def test_from_dict(self):
        data = {
            "etag": "xyz",
            "last_modified": "Mon, 01 Jan 2024",
            "body": "content",
            "status": 200,
            "timestamp": 500.0,
            "ttl": 1800,
        }
        entry = CacheEntry.from_dict(data)
        assert entry.etag == "xyz"
        assert entry.last_modified == "Mon, 01 Jan 2024"
        assert entry.body == "content"
        assert entry.ttl == 1800


class TestHTTPCache:
    @pytest.fixture
    def cache(self, tmp_path):
        return HTTPCache(cache_dir=tmp_path / "http_cache")

    def test_get_empty(self, cache):
        assert cache.get("https://example.com") is None

    def test_put_and_get(self, cache):
        entry = CacheEntry(etag="abc", body="data")
        cache.put("https://example.com", entry)

        result = cache.get("https://example.com")
        assert result is not None
        assert result.etag == "abc"
        assert result.body == "data"

    def test_get_expired_entry(self, cache):
        entry = CacheEntry(etag="old", body="stale")
        cache.put("https://example.com", entry)

        entry.timestamp = time.time() - 7200
        cache._index["https://example.com"] = entry

        assert cache.get("https://example.com") is None

    def test_conditional_headers_with_etag(self, cache):
        entry = CacheEntry(etag="abc123")
        cache.put("https://example.com", entry)

        headers = cache.conditional_headers("https://example.com")
        assert headers == {"If-None-Match": "abc123"}

    def test_conditional_headers_with_last_modified(self, cache):
        entry = CacheEntry(last_modified="Mon, 01 Jan 2024")
        cache.put("https://example.com", entry)

        headers = cache.conditional_headers("https://example.com")
        assert headers == {"If-Modified-Since": "Mon, 01 Jan 2024"}

    def test_conditional_headers_both(self, cache):
        entry = CacheEntry(etag="abc", last_modified="Mon, 01 Jan 2024")
        cache.put("https://example.com", entry)

        headers = cache.conditional_headers("https://example.com")
        assert headers == {"If-None-Match": "abc", "If-Modified-Since": "Mon, 01 Jan 2024"}

    def test_conditional_headers_no_entry(self, cache):
        assert cache.conditional_headers("https://example.com") == {}

    def test_clear(self, cache):
        cache.put("https://a.com", CacheEntry(body="a"))
        cache.put("https://b.com", CacheEntry(body="b"))
        assert len(cache._index) == 2

        cache.clear()
        assert len(cache._index) == 0
        assert not cache._index_path().exists()

    def test_persists_to_disk(self, tmp_path):
        cache = HTTPCache(cache_dir=tmp_path / "http_cache")
        cache.put("https://example.com", CacheEntry(etag="disk", body="persisted"))
        cache.flush()

        index_path = cache._index_path()
        assert index_path.exists()

        data = json.loads(index_path.read_text())
        assert "https://example.com" in data
        assert data["https://example.com"]["etag"] == "disk"
        # The body is a separate file, so the index stays small.
        assert "body" not in data["https://example.com"]

    def test_index_not_rewritten_per_put(self, tmp_path):
        """The index is flushed once per run, not once per put."""
        cache = HTTPCache(cache_dir=tmp_path / "http_cache")
        for i in range(20):
            cache.put(f"https://example.com/{i}", CacheEntry(body="x" * 1000))
        assert not cache._index_path().exists()

        cache.flush()
        assert cache._index_path().exists()

    def test_flush_is_idempotent(self, tmp_path):
        cache = HTTPCache(cache_dir=tmp_path / "http_cache")
        cache.put("https://example.com", CacheEntry(body="x"))
        cache.flush()
        cache.flush()
        assert cache.get("https://example.com") is not None

    def test_body_survives_reload(self, tmp_path):
        cache = HTTPCache(cache_dir=tmp_path / "http_cache")
        cache.put("https://example.com", CacheEntry(etag="e", body="<html>payload</html>"))
        cache.flush()

        reloaded = HTTPCache(cache_dir=tmp_path / "http_cache")
        entry = reloaded.get("https://example.com")
        assert entry is not None
        assert entry.body == "<html>payload</html>"
        assert entry.etag == "e"

    def test_loads_existing_index(self, tmp_path):
        """
        An index whose bodies are on disk loads.

        This used to be documented as "an index written by an older protor (body
        inline) still loads", which was not true and not what it tested: it also
        wrote the body file, so it exercised the current format. A genuine
        legacy index — body inline in the index, no body file — is dropped
        whole, because bodies moved out of the index to their own files and
        nothing reads the inline copy. Losing a cache is a refetch, not a
        correctness problem, but the compatibility was never there to rely on.
        """
        cache = HTTPCache(cache_dir=tmp_path / "http_cache")
        body_path = cache._body_path("https://example.com")
        body_path.parent.mkdir(parents=True, exist_ok=True)
        body_path.write_text("restored", encoding="utf-8")
        cache._index_path().write_text(
            json.dumps(
                {
                    "https://example.com": {
                        "etag": "loaded",
                        "last_modified": None,
                        "status": 200,
                        "timestamp": time.time(),
                        "ttl": 3600,
                    }
                }
            )
        )

        reloaded = HTTPCache(cache_dir=tmp_path / "http_cache")
        entry = reloaded.get("https://example.com")
        assert entry is not None
        assert entry.etag == "loaded"
        assert entry.body == "restored"

    def test_corrupt_index_handled_gracefully(self, tmp_path):
        cache_dir = tmp_path / "http_cache"
        cache_dir.mkdir()
        (cache_dir / "index.json").write_text("not valid json")

        cache = HTTPCache(cache_dir=cache_dir)
        assert cache._index == {}

    def test_a_legacy_index_with_inline_bodies_is_dropped_not_misread(self, tmp_path):
        """
        The truth about the old on-disk format, pinned so it cannot be claimed.

        Bodies used to live inline in the index file. They now live in their own
        files, and an index entry with no body file beside it is dropped — the
        alternative would be serving a page the cache does not have. Asserted
        here so the behaviour is known rather than assumed, and so nobody
        documents compatibility that does not exist.
        """
        cache = HTTPCache(cache_dir=tmp_path / "http_cache")
        cache._index_path().write_text(
            json.dumps(
                {
                    "https://example.com": {
                        "body": "<html>legacy</html>",
                        "etag": "e1",
                        "last_modified": None,
                        "status": 200,
                        "timestamp": time.time(),
                        "ttl": 3600,
                    }
                }
            ),
            encoding="utf-8",
        )

        reloaded = HTTPCache(cache_dir=tmp_path / "http_cache")
        assert reloaded.get("https://example.com") is None
        assert reloaded._load_index() == {}


class TestTheIndexStaysMetadataOnly:
    """
    A disk-backed cache must not become a memory-resident one.

    The docstring claimed bodies were "read lazily by `get` and dropped again, so
    only what is actually requested is in memory". The read was lazy; the drop
    never happened — `get` and `entry_for` attached the body to the *indexed*
    entry, which then held it for the life of the instance, and `put` stored the
    entry with its body attached. Measured 11.4 MiB retained across 500 entries
    of 24 kB, after reading every one of them.
    """

    def _fill(self, count: int = 200, size: int = 24000) -> tuple[HTTPCache, pathlib.Path]:
        cache_dir = pathlib.Path(tempfile.mkdtemp()) / "c"
        cache = HTTPCache(cache_dir=cache_dir)
        body = "x" * size
        for i in range(count):
            cache.put(f"https://x.example/{i}", CacheEntry(etag=f'"{i}"', body=body))
        cache.flush()  # a fresh cache over this directory needs the index on disk
        return cache, cache_dir

    def test_put_does_not_retain_the_body(self):
        cache, _ = self._fill()
        assert sum(len(e.body) for e in cache._index.values()) == 0

    def test_reading_every_body_retains_nothing(self):
        cache, _ = self._fill()
        for i in range(200):
            entry = cache.entry_for(f"https://x.example/{i}")
            assert entry is not None and entry.body == "x" * 24000, "body not served"
        assert sum(len(e.body) for e in cache._index.values()) == 0

    def test_the_body_is_still_served(self):
        """Released, not lost: a fresh cache over the same directory serves it."""
        cache, cache_dir = self._fill()
        assert cache.entry_for("https://x.example/7").body == "x" * 24000
        fresh = HTTPCache(cache_dir=cache_dir)
        assert fresh.get("https://x.example/7").body == "x" * 24000

    def test_validators_still_survive(self):
        """Releasing the body must not cost the ETag a revalidation needs."""
        cache, cache_dir = self._fill()
        cache.flush()
        fresh = HTTPCache(cache_dir=cache_dir)
        assert fresh.entry_for("https://x.example/3").etag == '"3"'
        assert fresh.conditional_headers("https://x.example/3") == {"If-None-Match": '"3"'}
