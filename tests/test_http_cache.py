"""Tests for protor.http_cache module."""

import json
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
        """An index written by an older protor (body inline) still loads."""
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
