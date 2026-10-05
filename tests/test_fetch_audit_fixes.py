"""Tests for defects found auditing the fetch path and the HTTP cache.

Each class below corresponds to one defect. They are written as reproductions
first: the comment at the top of each says what was broken and why it mattered,
so a future reader can tell a regression from a change of mind.

The audit findings share a shape. Every one of them is a case where the code
*looked* like it was guarding something and was not:

  - a cache write that could lose a page it had already fetched;
  - two ways of reading one store that disagreed about whether a missing body
    was an error;
  - a ``Content-Type`` read into a field and then ignored at the point of use;
  - a hook context advertising a key nothing ever read.

The failures are silent rather than loud in every case, which is why they
survived: nothing raised, nothing warned, and the output was merely wrong.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from protor.exceptions import FetchError
from protor.fetcher import fetch
from protor.http_cache import CacheEntry, HTTPCache

# ── HTTP doubles ──────────────────────────────────────────────────────────────


class _Resp:
    """The subset of ``aiohttp.ClientResponse`` that :func:`fetch` touches."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.status = spec.get("status", 200)
        self.headers = spec.get("headers", {})
        self.url = spec.get("url", "http://test.local/")
        self._body = spec.get("body", b"")

    async def __aenter__(self) -> _Resp:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def read(self) -> bytes:
        return self._body.encode("utf-8") if isinstance(self._body, str) else self._body


class Session:
    """A session that replays scripted responses and records what it was sent."""

    def __init__(self, *script: dict[str, Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Resp:
        self.calls.append({"url": url, **kwargs})
        return _Resp(self.script.pop(0) if self.script else {"status": 200, "body": b""})

    @property
    def headers_sent(self) -> dict[str, Any]:
        return self.calls[0]["headers"]


def page(body: str = "<html>ok</html>", **headers: str) -> dict[str, Any]:
    return {"status": 200, "body": body, "headers": headers}


def put(cache: HTTPCache, url: str, body: str, **kwargs: Any) -> None:
    """Store *body* and age it, so a stale entry can be revalidated on demand."""
    cache.put(url, CacheEntry(body=body, **kwargs))
    cache.flush()


def expire(cache: HTTPCache, url: str) -> None:
    cache._index[url].timestamp -= 7200


# ── 1. A cache-write failure must not destroy a fetched page ──────────────────


class TestCacheWriteFailureCannotLoseAPage:
    """
    ``cache.put`` sat inside ``fetch``'s try block but under no handler that
    could catch what it raises.

    It writes a body file, and that can fail: a read-only cache directory, a
    full disk, a path collision. What it raises is ``OSError``, which is neither
    ``TimeoutError`` nor ``aiohttp.ClientError``, so it escaped ``fetch``
    entirely — not even as the ``FetchError`` the docstring promises. The caller
    saw an exception for a page that had already been fetched successfully and
    written nowhere, and the engine recorded a hard failure for it.

    A cache is a pure optimisation. The cost of being unable to write one must
    be a refetch next time, never the page in hand.
    """

    @staticmethod
    def _cache_with_blocked_body_dir(tmp_path: Any) -> HTTPCache:
        """A cache whose body path is a directory, so ``write_text`` fails.

        Portable and not permission-based: ``IsADirectoryError`` is an
        ``OSError`` whatever uid the suite runs as, including root.
        """
        cache = HTTPCache(cache_dir=tmp_path / "c")
        cache._body_path("https://x.com/").mkdir(parents=True, exist_ok=True)
        return cache

    @pytest.mark.asyncio
    async def test_the_fetched_page_is_returned_not_raised(self, tmp_path):
        """The headline: a 200 already in hand must survive an unwritable cache."""
        cache = self._cache_with_blocked_body_dir(tmp_path)
        session = Session(page("<html>fetched fine</html>"))

        result = await fetch(session, "https://x.com/", cache=cache)

        assert result.text == "<html>fetched fine</html>"
        assert result.status == 200

    @pytest.mark.asyncio
    async def test_the_page_reports_its_real_size(self, tmp_path):
        """
        Not just non-raising but *right*.

        An ``OSError`` swallowed into a zero-byte or empty result would pass a
        "did it raise?" test while still reporting nothing scraped.
        """
        cache = self._cache_with_blocked_body_dir(tmp_path)
        body = "<html>" + "x" * 500 + "</html>"

        result = await fetch(Session(page(body)), "https://x.com/", cache=cache)

        assert result.text == body
        assert result.nbytes == len(body.encode("utf-8"))

    @pytest.mark.asyncio
    async def test_after_fetch_hooks_still_run(self, tmp_path):
        """
        The cache write is best-effort, so the hooks after it must still fire.

        Suppressing the ``OSError`` by abandoning the rest of the success path
        would trade one lost page for another.
        """
        cache = self._cache_with_blocked_body_dir(tmp_path)
        seen: list[dict[str, Any]] = []

        await fetch(
            Session(page("body")),
            "https://x.com/",
            cache=cache,
            hooks={"after_fetch": [lambda _u, ctx: seen.append(ctx)]},
        )

        assert [ctx["body"] for ctx in seen] == ["body"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root ignores directory permissions",
    )
    async def test_a_read_only_cache_directory_is_survivable(self, tmp_path):
        """The realistic form of the same failure: permissions, not a name clash."""
        cache = HTTPCache(cache_dir=tmp_path / "c")
        bodies = cache._bodies_dir
        mode = bodies.stat().st_mode
        os.chmod(bodies, 0o500)  # read + execute, no write
        try:
            result = await fetch(
                Session(page("<html>still here</html>")), "https://x.com/", cache=cache
            )
        finally:
            os.chmod(bodies, mode)

        assert result.text == "<html>still here</html>"

    @pytest.mark.asyncio
    async def test_the_next_fetch_still_works(self, tmp_path):
        """
        The failure is transient — a full disk gets emptied — so it must not
        poison the cache or wedge every later fetch of the same URL.
        """
        cache = self._cache_with_blocked_body_dir(tmp_path)
        url = "https://x.com/"

        first = await fetch(Session(page("first")), url, cache=cache)
        assert first.text == "first"

        cache._body_path(url).rmdir()  # the obstruction clears
        second = await fetch(Session(page("second")), url, cache=cache)

        assert second.text == "second"
        assert cache.get(url) is not None, "the write should succeed once it can"


# ── 2. A truncated body file is a cache problem, not a codec crash ────────────


class TestADamagedBodyFileIsNotACodecError:
    """
    ``_read_body`` caught ``OSError`` but not ``UnicodeDecodeError``, while
    ``_load_index`` caught both for ``index.json``.

    A body file truncated mid-multibyte-character — an interrupted write, a full
    disk, a process killed between the two — therefore raised a bare
    ``UnicodeDecodeError`` out of ``entry_for`` and ``get``, to a caller with no
    way to interpret it and no way to recover except by catching an exception
    the cache's own contract says it never raises.

    Damage to a body is a cache problem, and the cache answers it the same way
    it answers a missing body: there is nothing to serve, so the caller re-fetches.
    """

    URL = "https://x.com/"

    @staticmethod
    def _cache_with_corrupt_body(tmp_path: Any) -> HTTPCache:
        """An entry whose body file ends mid-character."""
        cache = HTTPCache(cache_dir=tmp_path / "c")
        put(cache, TestADamagedBodyFileIsNotACodecError.URL, "café — a whole page")
        # 'caf' plus the lead byte of 'é' with its continuation byte missing.
        cache._body_path(TestADamagedBodyFileIsNotACodecError.URL).write_bytes(b"caf\xc3")
        return cache

    def test_get_does_not_raise_a_codec_error(self, tmp_path):
        cache = self._cache_with_corrupt_body(tmp_path)
        assert cache.get(self.URL) is None

    def test_entry_for_does_not_raise_a_codec_error(self, tmp_path):
        cache = self._cache_with_corrupt_body(tmp_path)
        assert cache.entry_for(self.URL) is None

    def test_the_two_accessors_agree_about_damage(self, tmp_path):
        """
        The same disagreement as a *missing* body, in its other form.

        The index loader is forgiving about both ``OSError`` and
        ``UnicodeDecodeError``; the body reader forgave only the first. A cache
        that survives one kind of damage and crashes on the other is not
        predictable about which it will do.
        """
        cache = self._cache_with_corrupt_body(tmp_path)
        assert (cache.get(self.URL) is None) == (cache.entry_for(self.URL) is None)

    def test_the_index_and_the_body_are_held_to_the_same_standard(self, tmp_path):
        """Both readers now tolerate both failure modes. Pinning it explicitly."""
        cache = self._cache_with_corrupt_body(tmp_path)
        # The body file is still on disk, so the index still names it — it checks
        # presence, not readability. The damage shows up at read time instead,
        # and shows up as a miss rather than an exception.
        assert list(cache._index) == [self.URL]
        assert cache.get(self.URL) is None

        # A corrupt *index*, by contrast, makes the whole cache unusable, and
        # that too is survived rather than raised.
        cache._index_path().write_bytes(b"\xff\xfe\x00garbage")
        assert cache._load_index() == {}

    def test_a_genuinely_empty_body_is_still_served(self, tmp_path):
        """
        The guard must not swallow the legitimate case it resembles.

        A response that really was empty has a zero-byte body file. Reading it
        yields ``""``, which is not damage — treating it as a miss would turn
        every empty response into a refetch, forever.
        """
        cache = HTTPCache(cache_dir=tmp_path / "c")
        put(cache, self.URL, "")

        entry = cache.get(self.URL)

        assert entry is not None, "an empty response is not a missing one"
        assert entry is not None and entry.body == ""

    @pytest.mark.asyncio
    async def test_fetch_goes_to_the_network_for_a_corrupt_body(self, tmp_path):
        """End to end: the page is refetched rather than lost to an exception."""
        cache = self._cache_with_corrupt_body(tmp_path)
        session = Session(page("recovered"))

        result = await fetch(session, self.URL, cache=cache)

        assert result.text == "recovered"
        assert len(session.calls) == 1


# ── 3. A vanished body is an error, not an empty page ────────────────────────


class TestAVanishedBodyIsNotAnEntry:
    """
    ``get()`` returned a non-None entry whose ``.body`` was ``""`` when the body
    file had gone missing.

    ``CacheEntry.nbytes`` exists precisely to make that detectable — its own
    docstring says a missing file would otherwise "be served as a successful
    empty page" — but only ``fetch`` checked it. One store, two accessors, two
    answers to "is this entry intact?": the caller reading the cache directly
    was told a deleted page was an empty page, while the one going through
    ``fetch`` was told to re-download it.

    The direct accessor is the one that was wrong. An entry whose body cannot be
    read is not an entry, exactly as ``_load_index`` already decided at open
    time when it dropped such entries.
    """

    URL = "https://x.com/"

    @staticmethod
    def _cache_with_missing_body(tmp_path: Any) -> HTTPCache:
        cache = HTTPCache(cache_dir=tmp_path / "c")
        put(cache, TestAVanishedBodyIsNotAnEntry.URL, "REAL CONTENT")
        cache._body_path(TestAVanishedBodyIsNotAnEntry.URL).unlink()
        return cache

    def test_get_returns_none_rather_than_an_empty_body(self, tmp_path):
        cache = self._cache_with_missing_body(tmp_path)

        entry = cache.get(self.URL)

        assert entry is None, "a deleted body was served as an empty page"

    def test_entry_for_agrees_with_get(self, tmp_path):
        """
        The other half of the disagreement, and the reason to fix ``get``.

        ``fetch`` reached the same store through ``entry_for`` and did its own
        ``nbytes`` arithmetic to notice. Two accessors, one store, two verdicts.
        """
        cache = self._cache_with_missing_body(tmp_path)

        assert cache.entry_for(self.URL) is None

    def test_the_reopened_cache_agrees_too(self, tmp_path):
        """``_load_index`` has always dropped these; the accessors now match it."""
        cache = HTTPCache(cache_dir=tmp_path / "c")
        put(cache, self.URL, "REAL CONTENT")
        cache._body_path(self.URL).unlink()

        assert HTTPCache(cache_dir=tmp_path / "c").get(self.URL) is None

    def test_a_body_that_is_there_is_still_served(self, tmp_path):
        """The negative control: the fix must not turn every read into a miss."""
        cache = HTTPCache(cache_dir=tmp_path / "c")
        put(cache, self.URL, "REAL CONTENT")

        entry = cache.get(self.URL)

        assert entry is not None
        assert entry is not None and entry.body == "REAL CONTENT"

    @pytest.mark.asyncio
    async def test_fetch_refetches_instead_of_serving_an_empty_page(self, tmp_path):
        """
        This half already worked — it is pinned so the accessor fix above cannot
        quietly change it. ``fetch`` compared ``nbytes`` against the body it read
        and re-downloaded on a mismatch; now it does not have to.
        """
        cache = self._cache_with_missing_body(tmp_path)
        session = Session(page("REFRESHED"))

        result = await fetch(session, self.URL, cache=cache)

        assert len(session.calls) == 1
        assert result.text == "REFRESHED"

    @pytest.mark.asyncio
    async def test_a_304_cannot_serve_a_body_that_is_gone(self, tmp_path):
        """
        The 304 path had no such guard: it served whatever ``entry_for`` gave
        back, so a vanished body became an empty page reported as
        ``not_modified`` — a successful scrape of a page with nothing in it.

        With the accessor fixed, the honest answer is the same one the code
        already gives when there is no entry to satisfy a 304 with.
        """
        cache = HTTPCache(cache_dir=tmp_path / "c")
        put(cache, self.URL, "REAL CONTENT", etag='W/"1"')
        expire(cache, self.URL)
        cache._body_path(self.URL).unlink()

        session = Session({"status": 304})
        with pytest.raises(FetchError, match="304"):
            await fetch(session, self.URL, cache=cache)


# ── 4. put()'s contract with the caller's entry ──────────────────────────────


class TestPutDoesNotRewriteTheCallersEntry:
    """
    ``put`` overwrote ``entry.ttl`` and ``entry.stale_ttl`` with the cache-wide
    values and stamped ``timestamp`` and ``nbytes``, mutating four fields of an
    object the caller still holds — while copying it for the body, so it was
    inconsistent about ownership too.

    **Chosen contract: the cache's values always win, and the caller's object is
    left alone.** Two reasons.

    First, the alternative is not available. ``CacheEntry``'s fields default to
    3600/86400 and are not optional, so "per-entry values win" would mean every
    entry constructed without an explicit TTL ignores the one the cache was
    configured with. ``HTTPCache(cache_dir, ttl=1)`` would quietly become a
    no-op — and it is a real constructor argument, used by callers who want
    short-lived entries. Retention is a property of the *store*: ``prune``
    sweeps every entry against one window, so a per-entry window would make the
    same directory hold entries under two different policies.

    Second, a cache that rewrites the object it was handed is a cache the caller
    cannot reason about: it passes an entry, and afterwards its own copy says
    something the store never agreed to.

    So ``put`` builds the entry it stores and stamps the *copy*; the caller
    keeps exactly what it passed in.
    """

    @staticmethod
    def _cache(tmp_path: Any, **kwargs: Any) -> HTTPCache:
        return HTTPCache(cache_dir=tmp_path / "c", **kwargs)

    def test_the_callers_entry_is_untouched(self, tmp_path):
        """
        Before: four fields silently rewritten on the caller's object.

        This is the reproduction — ``put`` returned normally, and the caller's
        entry had a different TTL, a different retention window, a stamped
        timestamp and a byte count it never asked for.
        """
        cache = self._cache(tmp_path, ttl=99, stale_ttl=111)
        entry = CacheEntry(body="payload", ttl=5, stale_ttl=6)

        cache.put("https://x.com/", entry)

        assert entry.ttl == 5
        assert entry.stale_ttl == 6
        assert entry.timestamp == 0.0, "put stamped the caller's entry"
        assert entry.nbytes == 0, "put wrote a byte count onto the caller's entry"

    def test_the_stored_entry_carries_the_cache_policy(self, tmp_path):
        """
        The contract itself, pinned: a per-entry TTL does not beat the cache's.

        This passes before the fix too — that is the point. The behaviour is not
        the bug; the silent rewrite of the caller's object is. Pinning the
        chosen contract keeps the next person from "fixing" it the other way.
        """
        cache = self._cache(tmp_path, ttl=99, stale_ttl=111)

        cache.put("https://x.com/", CacheEntry(body="payload", ttl=5, stale_ttl=6))
        stored = cache._index["https://x.com/"]

        assert stored.ttl == 99
        assert stored.stale_ttl == 111

    def test_the_stored_entry_is_stamped_and_sized(self, tmp_path):
        """Stamp and byte count belong to the stored copy — they are bookkeeping."""
        cache = self._cache(tmp_path)
        before = time.time()

        cache.put("https://x.com/", CacheEntry(body="payload"))
        stored = cache._index["https://x.com/"]

        assert stored.timestamp >= before
        assert stored.nbytes == len("payload")

    def test_the_stored_entry_holds_no_body(self, tmp_path):
        """
        Ownership, applied consistently: the store keeps metadata, the caller
        keeps the body it passed in.
        """
        cache = self._cache(tmp_path)
        entry = CacheEntry(body="payload")

        cache.put("https://x.com/", entry)

        assert cache._index["https://x.com/"].body == ""
        assert entry.body == "payload"
        assert cache.get("https://x.com/").body == "payload"  # type: ignore[union-attr]

    def test_the_cache_policy_actually_governs_expiry(self, tmp_path):
        """
        The consequence that decides the contract: a ``ttl=1`` cache must expire.

        With per-entry values winning, the default 3600 on every
        ``CacheEntry()`` would make the constructor argument a no-op.
        """
        cache = self._cache(tmp_path, ttl=1, stale_ttl=3600)
        cache.put("https://x.com/", CacheEntry(body="payload"))

        assert cache.get("https://x.com/") is not None, "fresh entry must serve"
        cache._index["https://x.com/"].timestamp -= 10
        assert cache.get("https://x.com/") is None, "the cache's ttl governs"

    def test_reputting_the_same_entry_twice_works(self, tmp_path):
        """
        A caller reusing one entry object is a normal pattern, and it broke
        once ``put`` wrote into it: the second call inherited a stale timestamp
        and byte count from the first.
        """
        cache = self._cache(tmp_path)
        entry = CacheEntry(body="first")

        cache.put("https://x.com/", entry)
        cache.put("https://y.com/", entry)

        assert cache.get("https://x.com/").body == "first"  # type: ignore[union-attr]
        assert cache.get("https://y.com/").body == "first"  # type: ignore[union-attr]


# ── 5. The declared charset must be honoured ─────────────────────────────────


class TestTheDeclaredCharsetIsHonoured:
    """
    Every body was decoded as UTF-8 unconditionally, ignoring the ``charset=``
    parameter of the ``Content-Type`` header that ``fetch`` had already read
    into ``FetchResult.content_type``.

    A page served as ``text/html; charset=windows-1251`` or ``charset=shift_jis``
    had every non-ASCII byte replaced with U+FFFD — silently, since replacement
    never raises. The corruption then reached everything downstream: the saved
    HTML, the manifest, the extracted text, and non-ASCII link hrefs, so links
    dropped out of the crawl frontier and their pages were never discovered. It
    was written into the cache too, so every later read reproduced it. Meanwhile
    ``protor.robots.py`` had been decoding the same responses through
    ``resp.text()``, which honours the declared charset — the fetcher was the
    odd one out.

    The fix: honour the declared charset; fall back to UTF-8 with replacement
    when it is absent or unusable, which is the behaviour every existing caller
    relies on and the only one that cannot fail a fetch.
    """

    CYRILLIC = "<html><head><title>Статья</title></head><body>Привет, мир</body></html>"
    JAPANESE = "<html><head><title>日本語</title></head><body>テストページ</body></html>"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("charset", "text"),
        [("windows-1251", CYRILLIC), ("shift_jis", JAPANESE), ("koi8-r", CYRILLIC)],
    )
    async def test_a_declared_charset_is_decoded_with(self, charset, text):
        """
        The reproduction: every non-ASCII byte became U+FFFD, and the page
        looked fetched and parsed fine while containing no Russian or Japanese
        at all.
        """
        session = Session(
            page(text.encode(charset), **{"Content-Type": f"text/html; charset={charset}"})
        )

        result = await fetch(session, "https://x.com/")

        assert "�" not in result.text, "bytes were replaced instead of decoded"
        assert result.text == text

    @pytest.mark.asyncio
    async def test_the_content_type_is_still_reported_verbatim(self):
        """
        Decoding must not rewrite what the server said.

        ``content_type`` feeds ``looks_like_html`` and the "not a web page"
        reason in the engine; normalising the charset away would be a second,
        quieter corruption of the same header.
        """
        header = "text/html; charset=windows-1251"
        session = Session(page(self.CYRILLIC.encode("windows-1251"), **{"Content-Type": header}))

        result = await fetch(session, "https://x.com/")

        assert result.content_type == header

    @pytest.mark.asyncio
    async def test_a_quoted_charset_parameter_is_understood(self):
        """``charset="utf-8"`` is legal and common; the quotes are not the value."""
        session = Session(page("café", **{"Content-Type": 'text/html; charset="utf-8"'}))

        assert (await fetch(session, "https://x.com/")).text == "café"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "header",
        [
            "text/html; CHARSET=windows-1251",
            "text/html;charset=windows-1251",
            "text/html ; charset = windows-1251",
        ],
    )
    async def test_the_parameter_is_matched_loosely(self, header):
        """Parameter names are case-insensitive and whitespace is not significant."""
        session = Session(page(self.CYRILLIC.encode("windows-1251"), **{"Content-Type": header}))

        assert (await fetch(session, "https://x.com/")).text == self.CYRILLIC

    @pytest.mark.asyncio
    async def test_the_other_parameters_are_not_mistaken_for_the_charset(self):
        """``boundary`` is not a charset, and must not be decoded with."""
        session = Session(
            page("café", **{"Content-Type": 'multipart/form-data; boundary="charset=windows-1251"'})
        )

        assert (await fetch(session, "https://x.com/")).text == "café"

    @pytest.mark.asyncio
    async def test_no_declared_charset_still_means_utf8(self):
        """
        The absence of a declaration is not a licence to guess.

        There is no charset detector among protor's dependencies, and guessing
        wrong would be worse than the replacement characters: a latin-1 page
        decoded as cp1252 is silently *wrong* rather than visibly damaged.
        """
        session = Session(page("café ☕", **{"Content-Type": "text/html"}))

        assert (await fetch(session, "https://x.com/")).text == "café ☕"

    @pytest.mark.asyncio
    async def test_undecodable_bytes_are_replaced_not_fatal(self):
        """The pre-existing guarantee: bad bytes never fail a fetch."""
        session = Session(page(b"\xff\xfe bad bytes", **{"Content-Type": "text/html"}))

        result = await fetch(session, "https://x.com/")

        assert "bad bytes" in result.text

    @pytest.mark.asyncio
    async def test_an_unknown_charset_name_falls_back_instead_of_raising(self):
        """
        ``codecs.lookup`` raises ``LookupError`` on a name no codec answers to,
        and a server is free to send one. It must not take the page with it.
        """
        session = Session(page("café", **{"Content-Type": "text/html; charset=x-made-up"}))
        session.script[0]["body"] = "café".encode()

        assert (await fetch(session, "https://x.com/")).text == "café"

    @pytest.mark.asyncio
    async def test_a_server_that_lies_about_its_charset_still_yields_the_page(self, tmp_path):
        """
        Declaring a charset and then sending different bytes is common. The
        declared charset is tried first and the fallback takes over, so the
        result is the page rather than an exception.
        """
        session = Session(
            page("café ☕".encode(), **{"Content-Type": "text/html; charset=windows-1251"})
        )
        cache = HTTPCache(cache_dir=tmp_path / "c")

        result = await fetch(session, "https://x.com/", cache=cache)

        assert result.text == "café ☕"

    @pytest.mark.asyncio
    async def test_the_cache_stores_the_decoded_text_not_the_corruption(self, tmp_path):
        """
        The corruption was persisted. A second read reproduced it for the rest
        of the cache's life, so re-running the crawl could not recover the page
        even after the bug was fixed in the fetcher.
        """
        cache = HTTPCache(cache_dir=tmp_path / "c")
        url = "https://x.com/"
        header = {"Content-Type": "text/html; charset=windows-1251"}

        await fetch(Session(page(self.CYRILLIC.encode("windows-1251"), **header)), url, cache=cache)
        cached = cache.get(url)

        assert cached is not None
        assert cached is not None and cached.body == self.CYRILLIC
        assert cache._body_path(url).read_text(encoding="utf-8") == self.CYRILLIC

        # And the cache hit serves it back identically, not re-mangled.
        served = await fetch(Session(), url, cache=cache)
        assert served.text == self.CYRILLIC

    @pytest.mark.asyncio
    async def test_a_non_ascii_link_href_survives_into_the_frontier(self, tmp_path):
        """
        Why the silent U+FFFD substitution was worth fixing properly: a
        percent-free non-ASCII href no longer matched any page, so the link
        left the crawl frontier and the page behind it was never scraped.
        """
        from protor.parser import parse_html

        href = "/каталог/товары"
        html = f"<html><body><a href='{href}'>Каталог</a></body></html>"
        cache = HTTPCache(cache_dir=tmp_path / "c")

        result = await fetch(
            Session(
                page(
                    html.encode("windows-1251"),
                    **{"Content-Type": "text/html; charset=windows-1251"},
                )
            ),
            "https://x.com/",
            cache=cache,
        )
        _, parsed = parse_html(result.text, "https://x.com/")

        # Resolved to an absolute URL by the parser; the non-ASCII path is what
        # is being asserted, since a U+FFFD href matches no page at all.
        assert parsed.links == [f"https://x.com{href}"], f"the link was lost: {parsed.links}"
        assert "�" not in "".join(parsed.links)

    @pytest.mark.asyncio
    async def test_the_byte_count_is_the_bytes_received(self):
        """
        ``nbytes`` counts the wire, not the re-encoded text.

        For a windows-1251 page the two differ substantially, and the manifest
        and results table report transfer size.
        """
        raw = self.CYRILLIC.encode("windows-1251")
        session = Session(page(raw, **{"Content-Type": "text/html; charset=windows-1251"}))

        result = await fetch(session, "https://x.com/")

        assert result.nbytes == len(raw)
        assert result.nbytes != len(self.CYRILLIC.encode("utf-8"))


# ── 6. A hook's headers must reach the request ───────────────────────────────


class TestHookContextsAgreeAndAreHonoured:
    """
    ``hook_ctx`` advertised a ``headers`` key that the request was then built
    without, so a hook doing ``ctx["headers"]["Authorization"] = token`` got no
    error and no header. The only observable effect of setting a header was
    none.

    The two contexts also disagreed on shape: ``after_fetch`` got
    ``{"status", "body"}`` with no ``url``, while ``before_fetch`` carried
    ``url`` in its context. A hook that logs the request therefore saw the URL
    for one phase and not the other, for no stated reason.

    Both contexts now carry ``url``, and ``before_fetch``'s ``headers`` is the
    dict the request is actually built from.
    """

    URL = "https://x.com/"

    @pytest.mark.asyncio
    async def test_a_header_a_hook_sets_is_sent(self):
        """The reproduction: the header was written to a dict nothing read."""

        def authorise(_url: str, ctx: dict[str, Any]) -> None:
            ctx["headers"]["Authorization"] = "Bearer secret"

        session = Session(page())

        await fetch(session, self.URL, hooks={"before_fetch": [authorise]})

        assert session.headers_sent["Authorization"] == "Bearer secret"

    @pytest.mark.asyncio
    async def test_a_hook_may_replace_the_whole_header_mapping(self):
        """
        Mutating the dict is one way to use the key; assigning a new one is the
        other, and it used to be silently discarded too.
        """

        def replace_all(_url: str, ctx: dict[str, Any]) -> None:
            ctx["headers"] = {"X-Only": "1"}

        session = Session(page())

        await fetch(session, self.URL, hooks={"before_fetch": [replace_all]})

        assert session.headers_sent == {"X-Only": "1"}

    @pytest.mark.asyncio
    async def test_the_conditional_validators_survive_the_hook(self, tmp_path):
        """
        Handing over the real dict must not cost the cache its headers.

        A hook that adds one header used to be a way to lose ``If-None-Match``
        as well, which silently turned every conditional request into a full
        re-download.
        """
        cache = HTTPCache(cache_dir=tmp_path / "c")
        put(cache, self.URL, "original", etag='W/"abc"')
        expire(cache, self.URL)

        def add_token(_url: str, ctx: dict[str, Any]) -> None:
            ctx["headers"]["X-Token"] = "t"

        session = Session({"status": 304})

        await fetch(session, self.URL, cache=cache, hooks={"before_fetch": [add_token]})

        assert session.headers_sent["If-None-Match"] == 'W/"abc"'
        assert session.headers_sent["X-Token"] == "t"

    @pytest.mark.asyncio
    async def test_a_hook_that_raises_does_not_lose_the_page(self):
        """
        Hooks are user code, and they are already isolated: one raising must not
        cost the page.

        Worth re-pinning now that the hook's ``headers`` mapping is the real one.
        A hook that raised *after* clearing it has genuinely cleared it — that is
        the hook's decision, not a fault to roll back — so this asserts the
        isolation that does exist: the exception does not escape and the body
        still comes back.
        """
        def boom(_url: str, ctx: dict[str, Any]) -> None:
            ctx["headers"]["X-Partial"] = "1"
            raise RuntimeError("bad hook")

        session = Session(page("survived"))

        result = await fetch(session, self.URL, hooks={"before_fetch": [boom]})

        assert result.text == "survived"

    @pytest.mark.asyncio
    async def test_a_hook_that_replaces_headers_with_a_non_mapping_is_ignored(self):
        """
        The context is a ``dict[str, Any]``, so a hook can put anything in it.

        Whatever it puts where ``headers`` goes, the request must still be built
        from a real mapping — aiohttp's own failure for a bad header type would
        be an obscure crash in the middle of the retry loop, where it looks like
        a transport problem rather than a bad hook.
        """
        def replace_with_junk(_url: str, ctx: dict[str, Any]) -> None:
            ctx["headers"] = "not a mapping"

        session = Session(page())

        result = await fetch(session, self.URL, hooks={"before_fetch": [replace_with_junk]})

        assert "User-Agent" in session.headers_sent
        assert result.text == "<html>ok</html>"

    @pytest.mark.asyncio
    async def test_no_hooks_still_sends_the_user_agent(self):
        """The negative control: the header dict is built the same way."""
        session = Session(page())

        await fetch(session, self.URL)

        assert "User-Agent" in session.headers_sent

    @pytest.mark.asyncio
    async def test_after_fetch_receives_the_url(self):
        """The half of the inconsistency that made the contexts disagree."""
        seen: list[dict[str, Any]] = []

        await fetch(
            Session(page("body")), self.URL, hooks={"after_fetch": [lambda _u, c: seen.append(c)]}
        )

        assert seen[0]["url"] == self.URL

    @pytest.mark.asyncio
    async def test_both_contexts_carry_the_url(self):
        """
        The consistency pin: whatever else each phase offers, both name the URL.

        ``after_fetch`` keeps ``status`` and ``body``; ``before_fetch`` keeps
        ``headers``. Neither drops ``url``.
        """
        before: list[dict[str, Any]] = []
        after: list[dict[str, Any]] = []

        await fetch(
            Session(page("body")),
            self.URL,
            hooks={
                "before_fetch": [lambda _u, c: before.append(c)],
                "after_fetch": [lambda _u, c: after.append(c)],
            },
        )

        assert before[0]["url"] == self.URL
        assert after[0]["url"] == self.URL
        assert before[0]["headers"] is not None
        assert after[0]["status"] == 200
        assert after[0]["body"] == "body"

    @pytest.mark.asyncio
    async def test_a_fresh_cache_hit_still_runs_no_hooks(self, tmp_path):
        """
        Documenting the existing boundary: hooks are request-scoped, so a
        served-from-cache page fires neither. Pinned so nobody reads the header
        fix as having changed it.
        """
        cache = HTTPCache(cache_dir=tmp_path / "c")
        put(cache, self.URL, "cached body")
        seen: list[str] = []

        result = await fetch(
            Session(page("fresh")),
            self.URL,
            cache=cache,
            hooks={"after_fetch": [lambda url, _ctx: seen.append(url)]},
        )

        assert result.text == "cached body"
        assert seen == [], "a cache hit must not report a fetch that did not happen"
