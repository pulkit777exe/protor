"""
HTTP response cache using ETag and Last-Modified for conditional requests.

Response bodies are stored one file per URL and the index is written once per
run rather than after every insert. The previous version rewrote the entire
index -- every cached body included -- on each ``put``, which is quadratic:
60 pages of 50 KB wrote roughly 170 MB and spent 15 ms per page on the write
alone, all of it blocking the event loop.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    """Cached HTTP response metadata (the body lives in its own file)."""

    etag: str | None = None
    last_modified: str | None = None
    status: int = 200
    timestamp: float = 0.0
    ttl: int = 3600
    body: str = field(default="", repr=False)

    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> dict[str, Any]:
        return {
            "etag": self.etag,
            "last_modified": self.last_modified,
            "status": self.status,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        return cls(**data)


class HTTPCache:
    """Disk-backed HTTP cache using ETag/Last-Modified.

    Bodies live in ``bodies/<hash>``; only compact metadata is kept in
    ``index.json``. Call :meth:`flush` once at the end of a run to persist the
    index, or rely on the context manager, which flushes on exit.
    """

    def __init__(self, cache_dir: str | Path | None = None, ttl: int = 3600) -> None:
        self._cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".cache" / "protor" / "http"
        )
        self._bodies_dir = self._cache_dir / "bodies"
        self._bodies_dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl
        self._index: dict[str, CacheEntry] = self._load_index()
        self._dirty = False
        # An abandoned cache would otherwise keep its bytes forever.
        self.prune()

    # ── paths ────────────────────────────────────────────────────────────────

    def _index_path(self) -> Path:
        return self._cache_dir / "index.json"

    def _body_path(self, url: str) -> Path:
        return self._bodies_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.body"

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_index(self) -> dict[str, CacheEntry]:
        """
        Read cached metadata only — bodies stay on disk until requested.

        Loading every body up front made the cache fully resident in RAM (28 MB
        for 300 small pages, and unbounded beyond that), which defeats the
        point of a disk cache and can OOM a large crawl. Bodies are now read
        lazily by :meth:`get` and dropped again, so only what is actually
        requested is in memory.
        """
        path = self._index_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        entries: dict[str, CacheEntry] = {}
        for url, raw in data.items():
            try:
                entry = CacheEntry.from_dict(raw)
            except TypeError:
                continue
            if not self._body_path(url).exists():
                # Index points at a body that is gone; drop the entry rather
                # than serve an empty page.
                continue
            entries[url] = entry
        return entries

    def _read_body(self, url: str) -> str:
        try:
            return self._body_path(url).read_text(encoding="utf-8")
        except OSError:
            return ""

    def _drop_body(self, url: str) -> None:
        """Delete a URL's body file so the cache cannot grow without bound."""
        with contextlib.suppress(OSError):
            self._body_path(url).unlink()

    def prune(self) -> int:
        """
        Delete expired entries and their body files. Returns the number removed.

        Without this the body files were never deleted: ``get`` dropped the
        index entry but left the bytes on disk, so the cache grew forever.
        Called on open, so an abandoned cache still shrinks.
        """
        stale = [url for url, entry in self._index.items() if entry.is_expired]
        for url in stale:
            del self._index[url]
            self._drop_body(url)

        # Sweep orphaned bodies whose index entry no longer exists.
        live = {self._body_path(url).name for url in self._index}
        for body in self._bodies_dir.glob("*.body"):
            if body.name not in live:
                with contextlib.suppress(OSError):
                    body.unlink()

        if stale:
            self._dirty = True
        return len(stale)

    def flush(self) -> None:
        """Persist the index. Cheap and idempotent; safe to call repeatedly."""
        if not self._dirty:
            return
        data = {url: entry.to_dict() for url, entry in self._index.items()}
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Atomic replace so an interrupted run cannot leave a truncated index.
        fd, tmp = tempfile.mkstemp(dir=str(self._cache_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, self._index_path())
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        self._dirty = False

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> HTTPCache:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── lookups ──────────────────────────────────────────────────────────────

    def get(self, url: str) -> CacheEntry | None:
        """
        Return a cached entry for *url* if not expired, or None.

        The body is read from disk here and attached to the returned entry, so
        only the pages actually requested occupy memory.
        """
        entry = self._index.get(url)
        if entry is None:
            return None
        if entry.is_expired:
            del self._index[url]
            self._drop_body(url)
            self._dirty = True
            return None
        if not entry.body:
            entry.body = self._read_body(url)
        return entry

    def put(self, url: str, entry: CacheEntry) -> None:
        """Store a cache entry for *url* (body written once, index marked dirty)."""
        entry.timestamp = time.time()
        entry.ttl = self._ttl
        self._bodies_dir.mkdir(parents=True, exist_ok=True)
        self._body_path(url).write_text(entry.body, encoding="utf-8")
        # Keep the body out of the in-memory index copy that gets serialised.
        self._index[url] = entry
        self._dirty = True

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
        """Clear all cached entries and their body files."""
        self._index.clear()
        self._dirty = False
        with contextlib.suppress(OSError):
            self._index_path().unlink()
        for body in self._bodies_dir.glob("*.body"):
            with contextlib.suppress(OSError):
                body.unlink()

    def size_bytes(self) -> int:
        """Total bytes currently held on disk (index plus body files)."""
        return sum(p.stat().st_size for p in self._cache_dir.rglob("*") if p.is_file())
