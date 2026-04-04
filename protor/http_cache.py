"""HTTP response cache using ETag and Last-Modified for conditional requests."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    """Cached HTTP response metadata."""

    etag: str | None = None
    last_modified: str | None = None
    body: str = ""
    status: int = 200
    timestamp: float = 0.0
    ttl: int = 3600

    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> dict[str, Any]:
        return {
            "etag": self.etag,
            "last_modified": self.last_modified,
            "body": self.body,
            "status": self.status,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        return cls(**data)


class HTTPCache:
    """Simple disk-backed HTTP cache using ETag/Last-Modified."""

    def __init__(self, cache_dir: str | Path | None = None, ttl: int = 3600) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "protor" / "http"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl
        self._index: dict[str, CacheEntry] = self._load_index()

    def _index_path(self) -> Path:
        return self._cache_dir / "index.json"

    def _load_index(self) -> dict[str, CacheEntry]:
        path = self._index_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {url: CacheEntry.from_dict(entry) for url, entry in data.items()}
        except (json.JSONDecodeError, KeyError):
            return {}

    def _save_index(self) -> None:
        data = {url: entry.to_dict() for url, entry in self._index.items()}
        self._index_path().write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get(self, url: str) -> CacheEntry | None:
        """Return cached entry for *url* if not expired."""
        entry = self._index.get(url)
        if entry and not entry.is_expired:
            return entry
        if entry and entry.is_expired:
            del self._index[url]
        return None

    def put(self, url: str, entry: CacheEntry) -> None:
        """Store a cache entry for *url*."""
        entry.timestamp = time.time()
        entry.ttl = self._ttl
        self._index[url] = entry
        self._save_index()

    def conditional_headers(self, url: str) -> dict[str, str]:
        """Return headers for a conditional request."""
        entry = self._index.get(url)
        if not entry:
            return {}
        headers: dict[str, str] = {}
        if entry.etag:
            headers["If-None-Match"] = entry.etag
        if entry.last_modified:
            headers["If-Modified-Since"] = entry.last_modified
        return headers

    def clear(self) -> None:
        """Clear all cached entries."""
        self._index.clear()
        if self._index_path().exists():
            self._index_path().unlink()
