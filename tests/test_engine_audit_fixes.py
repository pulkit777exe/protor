"""
Three defects in :mod:`protor.engine`, each reproduced before it was fixed.

They share a shape worth naming: every one of them is a claim the engine
documents that the code does not keep.

* ``max_targets`` is documented as a ceiling on pages *attempted*, and the
  ceiling was really a ceiling on pages that ended in scraped/error/blocked — so
  a site whose pages link to PDFs walked straight past it.
* "One identity for both the question and the request", and every script went
  out with the session's default User-Agent, unthrottled.
* The JS reservation table was read with ``candidate in by_url.values()``, a
  linear scan per script, over a dict nothing bounded.

Each class below documents what was broken, why it mattered, and what the fix
rests on. The cost assertions are in :class:`TestJsReservationIsNotLinear`
because the results there were already pinned by ``tests/test_js_downloads.py``;
a results-only test cannot tell a linear scan from a set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from protor.engine import CrawlEngine, RecursiveSource, StaticQueue, StaticSource

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

# ── doubles ───────────────────────────────────────────────────────────────────


class _GrowingQueue(StaticQueue):
    """A batch queue that also takes links, so the frontier never empties."""

    def __init__(self, seeds: list[str]) -> None:
        super().__init__(seeds)
        self.seen = set(seeds)
        self.extra: list[str] = []

    def dequeue(self) -> str | None:
        if self.extra:
            return self.extra.pop(0)
        return super().dequeue()

    def enqueue(self, url: str, priority: int = 0) -> bool:
        if url not in self.seen:
            self.seen.add(url)
            self.extra.append(url)
            return True
        return False

    @property
    def empty(self) -> bool:
        return not self.extra and not self._urls


class _RecordingRateLimiter:
    """Stands in for :class:`DomainRateLimiter`, recording every domain asked for."""

    def __init__(self) -> None:
        self.waited: list[str] = []

    async def wait(self, domain: str) -> None:
        self.waited.append(domain)

    def tracked_domains(self) -> int:
        return len(set(self.waited))


class _NoLinks:
    def discover(self, url: str, page: object) -> list[str]:
        return []


def _engine_for_js(tmp_path: Path) -> CrawlEngine:
    """An engine with only its filename bookkeeping exercised."""

    class _EmptyQueue:
        def dequeue(self) -> str | None:
            return None

        def enqueue(self, url: str, priority: int = 0) -> bool:
            return False

        def mark_visited(self, url: str, success: bool = True, *, attempted: bool = True) -> None:
            pass

        @property
        def empty(self) -> bool:
            return True

    return CrawlEngine(
        queue=_EmptyQueue(),
        link_source=_NoLinks(),
        output_dir=tmp_path,
        max_targets=0,
    )


# ── 1. max_targets is not a ceiling ───────────────────────────────────────────


class TestMaxTargetsBoundsRequests:
    """
    `--max-pages` bounds requests, not just outcomes.

    The spawn loop compared ``stats.total`` — scraped + errors + blocked —
    against the ceiling. ``_skip()`` advances none of those, so a URL that was
    fetched and then *declined* cost a request and nothing else: the ceiling
    never closed. A docs site whose index links 30 PDFs under ``--max-pages 4``
    fetched all 30 and reported "1 page scraped", and nothing in the output said
    the budget had been spent five times over.

    The invariant the old comment rested on — "nothing can skip, because the
    parser yields same-host links only" — is false. A same-host ``<a href>`` to
    a PDF is skipped just as readily as an off-domain link is, and that is the
    common case on exactly the sites a crawl is pointed at.

    The fix counts dispatches, which is what the parameter has always been
    documented as bounding. ``CrawlStats.total`` deliberately stays
    scraped + errors + blocked: a skip is not a failure and must not be
    reported as one, which is the other half of ``tests/test_regressions.py``
    (``stats.total == 0`` for a dispatched PDF). Those two tests only looked
    contradictory because both were being read as claims about the same
    counter; one is about the ceiling, the other about the summary.
    """

    @staticmethod
    def _pdf_farm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, ceiling: int, links: int):
        """An index linking *links* PDFs; the PDFs are skipped, the index is not."""
        import protor.engine as engine_mod
        from protor.fetcher import FetchResult

        index = (
            "<html><body><h1>Manual</h1>"
            + "".join(
                f'<p><a href="https://ex.com/manual{i}.pdf">chapter {i}</a></p>'
                for i in range(links)
            )
            + "</body></html>"
        )
        requested: list[str] = []

        async def fake_fetch(session: Any, url: str, **kwargs: Any) -> FetchResult:
            requested.append(url)
            if url.endswith(".pdf"):
                return FetchResult(
                    text="%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>",
                    nbytes=42,
                    status=200,
                    content_type="application/pdf",
                )
            return FetchResult(text=index, nbytes=len(index), status=200, content_type="text/html")

        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)
        engine = CrawlEngine(
            queue=_GrowingQueue(["https://ex.com/"]),
            link_source=RecursiveSource(),
            output_dir=tmp_path / str(ceiling),
            max_targets=ceiling,
            allowed_domain="ex.com",
            concurrency=4,
        )
        return engine, requested

    async def test_a_page_of_pdfs_does_not_walk_past_the_ceiling(self, tmp_path, monkeypatch):
        engine, requested = self._pdf_farm(tmp_path, monkeypatch, ceiling=4, links=30)

        stats = await engine.arun()

        assert len(requested) <= 4, f"a ceiling of 4 issued {len(requested)} requests: {requested}"
        assert stats.dispatched <= 4, f"a ceiling of 4 dispatched {stats.dispatched} tasks"
        assert stats.scraped == 1, "the index itself is a real page and must still be saved"

    async def test_the_whole_frontier_is_skipped_work_but_not_free_work(
        self, tmp_path, monkeypatch
    ):
        """
        The counter the ceiling reads is dispatches, not outcomes.

        A skip is not a success and not a failure, so folding it into
        ``CrawlStats.total`` would report PDFs as pages the run had handled.
        The ceiling is therefore over ``dispatched``, and this pins that the two
        are genuinely separate numbers rather than the same one twice.
        """
        engine, _requested = self._pdf_farm(tmp_path, monkeypatch, ceiling=4, links=30)

        stats = await engine.arun()

        assert stats.dispatched == 4, stats.dispatched
        assert stats.total == 1, (
            "a skipped URL must not be tallied as a scraped page, an error or a block"
        )
        assert stats.scraped + stats.errors + stats.blocked == stats.total

    async def test_a_run_with_nothing_to_skip_still_honours_the_ceiling(
        self, tmp_path, monkeypatch
    ):
        """
        The guard on the fix: ordinary crawling is unchanged.

        A recursive crawl whose pages are all HTML is the case every existing
        ceiling test covers, and it must still stop at the ceiling on the
        server's request log.
        """
        import protor.engine as engine_mod
        from protor.fetcher import FetchResult

        body = (
            "<html><body>"
            + "".join(f'<p><a href="https://ex.com/p{i}">x</a></p>' for i in range(30))
            + "</body></html>"
        )
        requested: list[str] = []

        async def fake_fetch(session: Any, url: str, **kwargs: Any) -> FetchResult:
            requested.append(url)
            return FetchResult(text=body, nbytes=len(body), status=200, content_type="text/html")

        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)

        for ceiling in (1, 2, 5):
            engine = CrawlEngine(
                queue=StaticQueue(["https://ex.com/"]),
                link_source=RecursiveSource(),
                output_dir=tmp_path / f"plain-{ceiling}",
                max_targets=ceiling,
                allowed_domain="ex.com",
                concurrency=4,
            )
            stats = await engine.arun()
            assert stats.dispatched <= ceiling, (
                f"ceiling {ceiling} spawned {stats.dispatched} tasks"
            )
            assert len(requested) <= ceiling, f"ceiling {ceiling} issued {len(requested)} requests"


# ── 2. script downloads bypass the limiter and the User-Agent ─────────────────


class TestScriptDownloadsArePolite:
    """
    A script is a request like any other, and was not treated as one.

    ``_process_one`` picks a User-Agent and awaits ``rate_limiter.wait(domain)``
    before it fetches the page, and ``download_file`` was then called for every
    ``<script src>`` with neither. So on the batch default (``--download-js`` is
    on) a page fetched politely was immediately followed by up to
    ``MAX_JS_FILES`` unthrottled requests to CDN hosts nobody had budgeted for
    — and the robots check that was performed against *this* agent said nothing
    about what those requests then claimed to be.

    ``allow_internal_redirects`` had the same gap in the other direction: the
    engine threaded it into ``fetch`` and left it at its ``False`` default for
    scripts, even though ``download_file`` has accepted it all along.

    The User-Agent half cannot be fixed from here: ``download_file`` takes no
    ``user_agent``/``headers`` argument, so there is nothing for the engine to
    pass. That is left as a pinned requirement below rather than quietly
    dropped.
    """

    @staticmethod
    def _page_with_scripts(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        import protor.engine as engine_mod

        html = (
            "<html><head><title>T</title>"
            "<script src='https://cdn.example.com/a.js'></script>"
            "<script src='https://static.example.com/b.js'></script>"
            "</head><body><p>hi</p></body></html>"
        )
        seen: list[dict[str, Any]] = []

        async def fake_fetch(session: Any, url: str, **kwargs: Any) -> Any:
            from protor.fetcher import FetchResult

            return FetchResult(text=html, nbytes=len(html), status=200, content_type="text/html")

        async def fake_download(session: Any, url: str, dest: Any, **kwargs: Any) -> bool:
            seen.append({"url": url, "dest": str(dest), **kwargs})
            return True

        monkeypatch.setattr(engine_mod, "fetch", fake_fetch)
        monkeypatch.setattr(engine_mod, "download_file", fake_download)
        return seen

    async def _scrape_with_scripts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs):
        seen = self._page_with_scripts(monkeypatch)
        limiter = _RecordingRateLimiter()
        engine = CrawlEngine(
            queue=StaticQueue(["https://ex.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=1,
            download_js=True,
            rate_limiter=limiter,
            **kwargs,
        )
        stats = await engine.arun()
        return seen, limiter, stats

    async def test_a_script_request_waits_on_the_rate_limiter(self, tmp_path, monkeypatch):
        """
        The defect: the limiter was consulted once, for the page, and never again.

        ``--download-js`` is the batch default, so most of a run's traffic was
        unthrottled — and the CDN hosts scripts come from are precisely the ones
        a run has no prior relationship with.
        """
        seen, limiter, stats = await self._scrape_with_scripts(tmp_path, monkeypatch)

        assert stats.scraped == 1
        assert len(seen) == 2, f"no scripts were downloaded: {seen}"
        for download in seen:
            host = download["url"].split("/")[2]
            assert host in limiter.waited, (
                f"{download['url']} went out unthrottled; limiter saw {limiter.waited}"
            )
        assert "ex.com" in limiter.waited, "the page itself stopped being throttled"

    async def test_a_script_request_honours_the_internal_redirect_switch(
        self, tmp_path, monkeypatch
    ):
        """
        The same flag, the same guarantee, for both kinds of request.

        ``--allow-internal-redirects`` is documented as covering every request
        protor makes. Left at its default for scripts, a ``<script src>``
        answering ``302 Location: http://169.254.169.254/...`` was refused on
        the default path while the page path was not — the page path is the one
        that is easy to test, which is presumably why it was the one wired up.
        """
        seen, _limiter, _stats = await self._scrape_with_scripts(
            tmp_path, monkeypatch, allow_internal_redirects=True
        )

        assert len(seen) == 2
        for download in seen:
            assert download.get("allow_internal_redirects") is True, (
                f"{download['url']} ignored --allow-internal-redirects"
            )

    async def test_a_script_keeps_the_page_identity(self, tmp_path, monkeypatch):
        """
        The engine asks robots.txt about one identity and sends that identity.

        ``download_file`` took no ``user_agent``, so every script request carried
        the session's default headers while the page went out as the agent the
        run had just asked ``check_robots`` about. On the default path
        (``--download-js``) that is most of a run's traffic, and the two strings
        a site sees for one page are not the same — the guarantee the engine's
        own comment states ("one identity for both the question and the
        request") was silently false for scripts.

        This test was left as a non-strict xfail naming the signature change it
        needed, rather than dropped, so the gap stayed visible. It now passes for
        the same reason it was written.
        """
        seen = self._page_with_scripts(monkeypatch)
        engine = CrawlEngine(
            queue=StaticQueue(["https://ex.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=1,
            download_js=True,
        )
        await engine.arun()

        assert len(seen) == 2
        for download in seen:
            assert download.get("user_agent"), (
                "download_file has no user_agent parameter, so the script went out "
                "with the session default rather than the agent the run asked robots.txt about"
            )


# ── 3. the JS reservation table ───────────────────────────────────────────────


class TestJsReservationIsNotLinear:
    """
    One linear scan per script, over a dict nothing bounded.

    ``_reserve_js_filename`` asked ``candidate in by_url.values()`` — every
    filename already reserved for that site, compared one at a time — once per
    newly-seen script URL. With ``MAX_JS_FILES = 15`` and *N* pages on one
    domain that is ~112·N² value comparisons: 5,000 pages is about 2.8e9, spent
    on what should be a set lookup. The table itself never shrank, so it also
    grew without limit for the life of the engine.

    Both of the properties that make the table worth having are preserved: the
    same URL still gets the same filename, and two different URLs sharing a
    basename still get different ones. Past the bound, a URL is named from a
    digest of itself — deterministic, so the same URL still matches, and
    distinct, so a basename clash still separates.

    The cost assertion is the point. ``tests/test_js_downloads.py`` pins the
    *results* of this function, and a linear scan and a set produce identical
    results; only a comparison count can tell them apart.
    """

    #: Reserving this many scripts must stay linear, not quadratic.
    _N = 200

    class _CountingValues(dict):  # type: ignore[type-arg]
        """
        A dict whose ``.values()`` counts every *element* comparison made.

        Counting calls to ``__contains__`` would not do: the real comparison is
        one C-level scan of a view, so a Python-level counter sees one call per
        reservation whether that call touched one name or a thousand. The view
        returned here therefore compares element by element in Python, which is
        what makes the quadratic cost visible at all.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.comparisons = 0

        def values(self) -> Any:  # type: ignore[override]
            items = list(super().values())
            counter = self

            class _View:
                def __contains__(self, item: Any) -> bool:
                    for existing in items:
                        counter.comparisons += 1
                        if item == existing:
                            return True
                    return False

                def __iter__(self) -> Any:
                    return iter(items)

            return _View()

    def test_reserving_many_scripts_does_not_compare_quadratically(self, tmp_path):
        engine = _engine_for_js(tmp_path)
        by_url = self._CountingValues()
        engine._js_names["site"] = by_url

        n = self._N
        for i in range(n):
            engine._reserve_js_filename("site", i, f"https://x.com/static/s{i}.js")

        # A set lookup per reservation, plus at most one re-check. The linear
        # scan spent 1 + 2 + ... + n; a quadratic bound could not pass this.
        assert by_url.comparisons <= 2 * n, (
            f"{n} reservations cost {by_url.comparisons} filename comparisons; "
            "the membership test is still a scan"
        )

    def test_the_table_is_bounded(self, tmp_path):
        """
        Growth, not just cost.

        Unbounded bookkeeping is its own defect: 40,000 pages x 15 scripts is
        600,000 permanent entries of two strings each, retained for a run that
        only ever reads the most recent few pages' worth.
        """
        engine = _engine_for_js(tmp_path)
        limit = engine._js_reservation_limit()

        for i in range(limit * 2):
            engine._reserve_js_filename("site", i, f"https://x.com/static/s{i}.js")

        stored = len(engine._js_names["site"])
        assert stored <= limit, f"the reservation table holds {stored} URLs past a bound of {limit}"

    def test_the_properties_survive_the_bound(self, tmp_path):
        """
        A bound that gives up the two guarantees is not a fix.

        Both are asserted past the bound deliberately: that is where a fix that
        simply forgets to store anything would show up.
        """
        engine = _engine_for_js(tmp_path)
        limit = engine._js_reservation_limit()
        for i in range(limit + 50):
            engine._reserve_js_filename("site", i, f"https://x.com/static/app.js?v={i}")

        assert engine._reserve_js_filename("site", 0, "https://x.com/static/app.js?v=0") == (
            engine._reserve_js_filename("site", 1, "https://x.com/static/app.js?v=0")
        ), "the same URL got two names past the bound"

        names = {
            engine._reserve_js_filename("site", i, f"https://x.com/static/app.js?v={i}")
            for i in range(limit, limit + 50)
        }
        assert len(names) == 50, (
            f"{len(names)} distinct URLs sharing a basename collapsed onto one file"
        )

    def test_the_bound_is_per_site(self, tmp_path):
        """Two sites have two ``js/`` directories; neither may starve the other."""
        engine = _engine_for_js(tmp_path)
        limit = engine._js_reservation_limit()

        for i in range(limit):
            engine._reserve_js_filename("site-a", i, "https://a.com/static/app.js")

        assert (
            engine._reserve_js_filename("site-b", 0, "https://b.com/static/app.js") == "app.js"
        ), "a full site table pushed another site onto hashed names"


# ── item 3, unchanged behaviour the fix must not disturb ──────────────────────


class TestJsReservationStillBehaves:
    """
    The two guarantees, re-asserted at the scale ``tests/test_js_downloads.py``
    uses, so the optimisation is visibly not changing them.
    """

    def test_the_same_url_is_stored_once(self, tmp_path):
        engine = _engine_for_js(tmp_path)
        names = {
            engine._reserve_js_filename("site", 0, "https://x.com/static/vendor.js")
            for _ in range(50)
        }
        assert len(names) == 1, f"the same URL was stored under {len(names)} names"

    def test_a_shared_basename_still_separates(self, tmp_path):
        engine = _engine_for_js(tmp_path)
        names = [
            engine._reserve_js_filename("site", i, url)
            for i, url in enumerate(
                [
                    "https://x.com/static/app.js",
                    "https://x.com/static/app.js?v=2",
                    "https://x.com/static/app.js?v=3",
                ]
            )
        ]
        assert len(set(names)) == 3, names

    def test_a_free_name_is_still_preferred(self, tmp_path):
        engine = _engine_for_js(tmp_path)
        assert engine._reserve_js_filename("site", 0, "https://x.com/a/jquery.js") == "jquery.js"


# ── 4. the same scan, in the page table ───────────────────────────────────────


class TestPageReservationIsNotLinear:
    """
    The JS twin of :class:`TestJsReservationIsNotLinear`, in the page table.

    ``_reserve_page_filename`` asked ``name in by_url.values()`` — every page
    filename reserved for that site, compared one at a time — once per page.
    Exactly N²/2 comparisons for N pages: measured 7,998,000 for 4,000 pages and
    646 ms of pure CPU, extrapolating to ~65 s over a 40,000-page crawl, spent
    deciding whether a page needed a hash. Worse than the JS case in one respect
    — no ``MAX_JS_FILES`` multiplier, it is per *page* — and it sat directly
    beside the fixed JS table, which is how it survived the audit: the same
    defect, one function over, applied to the thing the crawl actually writes.

    Unlike the JS table this one is deliberately **not** bounded. It holds one
    entry per page saved, so its size is the run's own page count, already capped
    by ``max_targets``. Bounding it would mean forgetting which filenames are in
    use, and forgetting that is what lets two pages share one file.
    """

    def test_reserving_many_pages_does_not_compare_quadratically(self):
        import time

        class CountingValues(dict):
            """A dict whose ``.values()`` compares in Python.

            A counting spy would not work: the real check is one C-level scan of a
            view either way, so a counter sees a single call regardless of how many
            elements it walks. Making the walk happen in Python is what makes the
            cost visible at all.
            """

            comparisons = 0

            def values(self):
                for value in super().values():
                    CountingValues.comparisons += 1
                    yield value

        engine = CrawlEngine.__new__(CrawlEngine)
        engine._page_names = {"site": CountingValues()}
        engine._page_taken_names = {}

        started = time.perf_counter()
        for i in range(2000):
            engine._reserve_page_filename("site", f"https://ex.com/a/page{i}.html", f"page{i}.html")
        elapsed = time.perf_counter() - started

        assert CountingValues.comparisons == 0, (
            f"{CountingValues.comparisons} filename comparisons for 2000 pages: "
            f"the lookup went back to a scan"
        )
        # Linear work at ~0.4us/page is under 50ms here; the quadratic form took
        # ~160ms for this count. A loose bound, because this is a wall-clock
        # assertion and the comparison count above is the real one.
        assert elapsed < 0.05, f"2000 reservations took {elapsed * 1e3:.0f} ms"

    def test_the_cost_does_not_grow_with_the_crawl(self):
        """The property that matters: per-page cost flat as the crawl deepens."""
        import time

        def us_per_page(n: int) -> float:
            engine = CrawlEngine.__new__(CrawlEngine)
            engine._page_names = {}
            engine._page_taken_names = {}
            started = time.perf_counter()
            for i in range(n):
                engine._reserve_page_filename(
                    "s", f"https://ex.com/a/page{i}.html", f"page{i}.html"
                )
            return (time.perf_counter() - started) * 1e6 / n

        small = us_per_page(1000)
        large = us_per_page(8000)
        assert large < small * 4, (
            f"per-page cost grew from {small:.2f}us at 1,000 pages to "
            f"{large:.2f}us at 8,000 -- that is the quadratic shape"
        )

    def test_a_collided_page_keeps_its_extension(self):
        """
        The digest belongs between the stem and the extension.

        ``a-b.1f3c.html``, not ``a-b.1f3chtml``: dropping the dot leaves a file
        nothing recognises as HTML, which is a new failure introduced alongside the
        fix for the old one. Caught by printing the actual output rather than by
        mutation — a mutation of the lookup leaves this behaviour untouched, so
        nothing would have flagged it.
        """
        engine = CrawlEngine.__new__(CrawlEngine)
        engine._page_names = {}
        engine._page_taken_names = {}

        first = engine._reserve_page_filename("s", "https://ex.com/a/b.html", "a-b.html")
        second = engine._reserve_page_filename("s", "https://ex.com/a-b.html", "a-b.html")

        assert first != second, "the collision was not resolved"
        assert second.endswith(".html"), f"the extension was lost: {second}"
        assert second.count(".") == 2, f"the digest was not separated: {second}"

    def test_a_collision_without_an_extension_still_resolves(self):
        """The no-suffix branch is a separate code path, and was written separately."""
        engine = CrawlEngine.__new__(CrawlEngine)
        engine._page_names = {}
        engine._page_taken_names = {}

        names = {
            engine._reserve_page_filename("s", f"https://ex.com/{p}", "a-b") for p in ("a/b", "a-b")
        }
        assert len(names) == 2, names

    def test_the_same_url_still_gets_the_same_name(self):
        """The property the table exists for, re-asserted against the new lookup."""
        engine = CrawlEngine.__new__(CrawlEngine)
        engine._page_names = {}
        engine._page_taken_names = {}

        first = engine._reserve_page_filename("s", "https://ex.com/a/b.html", "a-b.html")
        again = engine._reserve_page_filename("s", "https://ex.com/a/b.html", "a-b.html")
        assert first == again
        assert first == "a-b.html", "the first claimant should keep the plain name"

    def test_two_sites_do_not_collide_with_each_other(self):
        """The set is keyed per site, like the dict it mirrors."""
        a = CrawlEngine.__new__(CrawlEngine)
        b = CrawlEngine.__new__(CrawlEngine)
        for engine in (a, b):
            engine._page_names = {}
            engine._page_taken_names = {}

        assert (
            a._reserve_page_filename("one.com", "https://one.com/a/b.html", "a-b.html")
            == b._reserve_page_filename("two.com", "https://two.com/a/b.html", "a-b.html")
            == "a-b.html"
        )
