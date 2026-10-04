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

#: Seconds a sweep of the bodies directory stays good for. The sweep is one
#: readdir plus one stat per body file, so its cost tracks the size of a cache
#: that only ever grows: 0.5 ms for 200 files, 4.8 ms for 2,000, 20 ms for
#: 8,000 — and it used to be paid by *every* construction, before any work was
#: done. An orphan can only appear when a run died between writing a body and
#: flushing the index, so nothing is lost by looking for them on a timer rather
#: than on every open.
DEFAULT_SWEEP_INTERVAL_S = 300.0

#: Zero-byte marker recording when the bodies directory was last swept. Its mtime
#: is the whole payload, which is why the schedule survives between processes —
#: each `protor` invocation is a fresh interpreter with no memory of the last.
_SWEEP_MARKER = ".last-sweep"


def _dir_has_entries(path: Path) -> bool:
    """True if *path* holds at least one entry, without listing them all."""
    try:
        with os.scandir(path) as entries:
            return next(entries, None) is not None
    except OSError:
        return False


@dataclass
class CacheEntry:
    """Cached HTTP response metadata (the body lives in its own file)."""

    etag: str | None = None
    last_modified: str | None = None
    status: int = 200
    timestamp: float = 0.0
    #: Seconds the entry may be served without revalidating.
    ttl: int = 3600
    #: Extra seconds the entry is *retained* after it goes stale, so its
    #: ETag/Last-Modified can still be sent. Without this window, expiry deletes
    #: the validators and every "conditional" request silently degrades into a
    #: full re-download.
    stale_ttl: int = 86_400
    #: Byte length of the body as stored. Recorded so a body file that vanished
    #: from disk is distinguishable from a response that really was empty:
    #: without it a missing file was served as a successful empty page.
    #: Indexes written before this field existed load with 0, which simply
    #: disables the check for them.
    nbytes: int = 0
    #: The response's own Content-Type, carried through the cache so a cached
    #: body can still be recognised as not-a-page. Cached entries from before
    #: this field existed load with "", which the sniffing path treats as unknown
    #: rather than as HTML.
    content_type: str = ""
    body: str = field(default="", repr=False)

    @property
    def age(self) -> float:
        return time.time() - self.timestamp

    @property
    def is_expired(self) -> bool:
        """True once too old to serve without revalidating."""
        return self.age > self.ttl

    @property
    def is_retained(self) -> bool:
        """True while still worth keeping for revalidation."""
        return self.age <= self.ttl + self.stale_ttl

    def to_dict(self) -> dict[str, Any]:
        return {
            "etag": self.etag,
            "last_modified": self.last_modified,
            "status": self.status,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "stale_ttl": self.stale_ttl,
            "nbytes": self.nbytes,
            "content_type": self.content_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        # Tolerate indexes written before stale_ttl existed.
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class HTTPCache:
    """Disk-backed HTTP cache using ETag/Last-Modified.

    Bodies live in ``bodies/<hash>``; only compact metadata is kept in
    ``index.json``. Call :meth:`flush` once at the end of a run to persist the
    index, or rely on the context manager, which flushes on exit.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        ttl: int = 3600,
        stale_ttl: int = 86_400,
        *,
        sweep_interval: float | None = None,
    ) -> None:
        self._cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".cache" / "protor" / "http"
        )
        self._bodies_dir = self._cache_dir / "bodies"
        self._bodies_dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl
        self._stale_ttl = stale_ttl
        self._paths: dict[str, Path] = {}
        # sweep_interval=0 restores the old sweep-on-every-open behaviour, which
        # is what the tests use to make the orphan path deterministic.
        self._sweep_interval = (
            DEFAULT_SWEEP_INTERVAL_S if sweep_interval is None else float(sweep_interval)
        )
        self._index: dict[str, CacheEntry] = self._load_index()
        self._dirty = False
        # An abandoned cache would otherwise keep its bytes forever. Entry expiry
        # is a walk of the in-memory index plus one unlink per dead entry, so it
        # still happens on every open; only the O(files) orphan sweep is deferred.
        self.prune(sweep_orphans=self._sweep_is_due())

    # ── paths ────────────────────────────────────────────────────────────────

    def _index_path(self) -> Path:
        return self._cache_dir / "index.json"

    def _body_path(self, url: str) -> Path:
        """
        Where *url*'s body lives, memoised for the life of this instance.

        The path is a pure function of the URL, but building it costs a SHA-256
        plus a pathlib join, and every lookup of the same URL — load, get, put,
        touch, entry_for, prune — needs it. Profiling a 2,000-entry open showed
        this alone at 11 ms of 27 ms, most of it pathlib argument parsing.
        """
        cached = self._paths.get(url)
        if cached is None:
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
            cached = self._bodies_dir / f"{digest}.body"
            self._paths[url] = cached
        return cached

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_index(self) -> dict[str, CacheEntry]:
        """
        Read cached metadata only — bodies stay on disk until requested.

        Loading every body up front made the cache fully resident in RAM (28 MB
        for 300 small pages, and unbounded beyond that), which defeats the
        point of a disk cache and can OOM a large crawl. Bodies are now read
        lazily by :meth:`get` and dropped again, so only what is actually
        requested is in memory.

        A damaged or non-object index yields an empty cache but sets
        ``self._index_readable = False``. That matters: an empty index is
        indistinguishable from a *truncated* one, and letting :meth:`prune`
        reconcile against a failed read deleted every body file on disk —
        one interrupted write destroyed the whole cache.
        """
        self._index_readable = True
        path = self._index_path()
        if not path.exists():
            return {}
        try:
            raw_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            self._index_readable = False
            return {}
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            self._index_readable = False
            return {}
        if not isinstance(data, dict):
            # Valid JSON of the wrong shape (null, a list, a string) is as
            # unusable as a parse error, and used to raise AttributeError.
            self._index_readable = False
            return {}
        entries: dict[str, CacheEntry] = {}
        # One directory read instead of a stat() per entry. The check is the
        # guard that stops a vanished body being served as an empty page, so it
        # stays — but asking the directory what it holds costs one syscall
        # sequence rather than N, which is most of a large cache's open time
        # (measured 7 ms of 27 ms at 2,000 entries).
        present = self._body_filenames()
        for url, raw_entry in data.items():
            if not isinstance(url, str) or not isinstance(raw_entry, dict):
                continue
            try:
                entry = CacheEntry.from_dict(raw_entry)
            except (TypeError, ValueError):
                continue
            if self._body_path(url).name not in present:
                # Index points at a body that is gone; drop the entry rather
                # than serve an empty page.
                continue
            entries[url] = entry
        return entries

    def _body_filenames(self) -> set[str]:
        """Names of the body files currently on disk."""
        try:
            with os.scandir(self._bodies_dir) as it:
                return {e.name for e in it if e.name.endswith(".body")}
        except OSError:
            return set()

    def _read_body(self, url: str) -> str:
        try:
            return self._body_path(url).read_text(encoding="utf-8")
        except OSError:
            return ""

    def _drop_body(self, url: str) -> None:
        """Delete a URL's body file so the cache cannot grow without bound."""
        with contextlib.suppress(OSError):
            self._body_path(url).unlink()

    def prune(self, *, sweep_orphans: bool = True) -> int:
        """
        Discard entries past their retention window, plus their body files.

        Stale-but-retained entries are kept so conditional requests still work;
        only entries older than ``ttl + stale_ttl`` go. Body files with no index
        entry are swept too, which is what previously let the cache grow without
        bound.

        The orphan sweep is skipped when the index could not be read: an
        unreadable index looks exactly like an empty one, and reconciling
        against it deleted every cached body on disk. Returns the number of
        entries removed.

        *sweep_orphans* defaults to True, so calling this does a full prune. The
        constructor is the caller that passes False, and the reason the sweep is
        rate-limited at all; see :meth:`_sweep_is_due`.
        """
        doomed = [url for url, entry in self._index.items() if not entry.is_retained]
        for url in doomed:
            del self._index[url]
            self._drop_body(url)

        if doomed:
            self._dirty = True

        if sweep_orphans and self._index_readable:
            self._sweep_orphan_bodies()
        return len(doomed)

    def _sweep_is_due(self) -> bool:
        """
        Whether this construction should sweep orphan bodies, or trust the last sweep.

        The sweep is one readdir plus one stat per body file, and it was being
        paid by every construction before any work happened — 0.5 ms at 200
        files, 4.8 ms at 2,000, 20 ms at 8,000, growing forever because the
        cache only grows. An orphan can only appear when a run died between
        writing a body and flushing the index, so deferring the search to once
        per ``sweep_interval`` bounds the leaked bytes just as tightly.

        A marker file carries the last sweep time because each run is a fresh
        process with no memory of the last.
        """
        # Nothing to reconcile against: an unreadable index is indistinguishable
        # from an empty one, so a sweep could not tell a live body from a ghost.
        if not self._index_readable:
            return False
        if self._sweep_interval <= 0:
            return True
        if not self._index and _dir_has_entries(self._bodies_dir):
            # Every body is unreachable — a lost index. Swept regardless of the
            # interval, since there is nothing here worth keeping.
            return True
        try:
            last = (self._cache_dir / _SWEEP_MARKER).stat().st_mtime
        except OSError:
            # No marker: this cache has never been swept, so sweep it now rather
            # than inherit an interval from nobody.
            return True
        return (time.time() - last) >= self._sweep_interval

    def _sweep_orphan_bodies(self) -> int:
        """Delete every body file with no index entry. Returns how many went."""
        live = {self._body_path(url).name for url in self._index}
        removed = 0
        for body in self._bodies_dir.glob("*.body"):
            if body.name not in live:
                with contextlib.suppress(OSError):
                    body.unlink()
                removed += 1
        self._mark_swept()
        return removed

    def _mark_swept(self) -> None:
        """
        Record the sweep time for the next open to read.

        Failure (a read-only cache directory, say) only means the next open
        sweeps again, which is the direction that leaks less.
        """
        with contextlib.suppress(OSError):
            (self._cache_dir / _SWEEP_MARKER).touch()

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
        Return a *fresh* cached entry for *url*, or None.

        An expired entry is deliberately kept, not discarded: its ETag and
        Last-Modified are still needed to revalidate, and deleting it here meant
        the conditional request could never be made — expiry silently degraded
        into a full re-download every time. Retention is bounded by
        :meth:`prune`, which discards entries past ``ttl + stale_ttl``.

        The body is read from disk on demand, so only requested pages are
        resident in memory.
        """
        entry = self._index.get(url)
        if entry is None or entry.is_expired:
            return None
        if not entry.body:
            entry.body = self._read_body(url)
        return entry

    def entry_for(self, url: str) -> CacheEntry | None:
        """
        Return the entry for *url*, fresh or expired, without discarding it.

        This is what a caller needs to choose between serving and revalidating:
        :meth:`get` returns None once stale, which is exactly the state a 304
        refers to.
        """
        entry = self._index.get(url)
        if entry is not None and not entry.body:
            entry.body = self._read_body(url)
        return entry

    #: Retained as an alias for :meth:`entry_for`.
    lookup = entry_for

    def touch(self, url: str) -> None:
        """Mark an entry fresh again, e.g. after the server confirms it with a 304."""
        entry = self._index.get(url)
        if entry is not None:
            entry.timestamp = time.time()
            self._dirty = True

    def put(self, url: str, entry: CacheEntry) -> None:
        """Store a cache entry for *url* (body written once, index marked dirty)."""
        entry.timestamp = time.time()
        entry.ttl = self._ttl
        entry.stale_ttl = self._stale_ttl
        entry.nbytes = len(entry.body.encode("utf-8"))
        self._bodies_dir.mkdir(parents=True, exist_ok=True)
        self._body_path(url).write_text(entry.body, encoding="utf-8")
        self._index[url] = entry
        self._dirty = True

    def conditional_headers(self, url: str) -> dict[str, str]:
        """Return headers for a conditional request, including for expired entries."""
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
        # Nothing is left to reconcile, so the next open must not inherit a
        # "already swept" verdict from a marker describing a fuller directory.
        self._mark_swept()

    def size_bytes(self) -> int:
        """Total bytes currently held on disk (index plus body files)."""
        return sum(p.stat().st_size for p in self._cache_dir.rglob("*") if p.is_file())
