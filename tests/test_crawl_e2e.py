"""End-to-end crawl tests against a real HTTP server.

Every other crawler test drives :class:`protor.crawler.Crawler` with the engine's
``fetch`` replaced or the network patched out. That is the right way to test the
queue arithmetic, the log rows and the checkpoint cadence — and it is exactly why
the resume path was untested: nothing in a stubbed run can show whether a second
``--resume`` puts bytes on the wire again, or whether ``--max-pages`` still holds
once two runs have added to the same directory. Those are properties of a real
run, so these tests run one.

The server below is a real ``aiohttp`` listener on an ephemeral loopback port.
It records every path it is asked for, which turns "the crawl discovered the
links" and "the resumed crawl fetched nothing" into facts about observed traffic
rather than about what a mock was asked to return.

Which resume behaviour is asserted
----------------------------------
The strict form: **a resumed crawl issues no requests at all.** Reading
``protor/crawler.py`` and ``protor/engine.py`` to decide that:

* ``_CrawlQueue.enqueue`` refuses any URL with a ``visited`` row of
  ``success = 1``, and ``Crawler.__init__`` seeds the start URL through
  ``enqueue``, so a page the previous run finished is never re-admitted.
* Once a page has been dispatched it is gone from ``queue``, and its links are
  only re-harvested by scraping it again. A completed run therefore leaves a
  queue with nothing in it, ``CrawlEngine._start``'s ``spawn()`` sees
  ``queue.empty`` and dispatches nothing — so no robots check, no fetch, no
  traffic of any kind.

``--resume`` does not revalidate either: ``HTTPCache`` is not wired into
``Crawler._run``, and ``_SEEN_SQL``'s ``scraped_at >= ?`` clause deliberately
*excludes* rows that failed in an earlier run, so those are eligible for a retry
rather than a 304. The one case where a resumed run does hit the network is a
seed URL that itself failed, covered by ``TestResume.test_a_seed_url_that_failed_is_retried``.

Where the counts must be exact
------------------------------
Most tests crawl :data:`TREE`, a five-page site in which every page links to
exactly one other page. That shape is what makes a request count equal to a page
count, which is what lets ``--max-pages`` be asserted as a number. The cyclic
:func:`link_graph` — twelve edges between five pages, every child linking back
to the index — is used where the point is traffic rather than arithmetic.

A bug this file was written to catch
-----------------------------------
``TestPlainCrawl.test_a_cycle_produces_no_duplicate_request`` used to fail: a
page that was dequeued and in flight was in neither the queue nor ``visited``,
so a concurrent page linking to it re-admitted it. That fetched 8 requests for
a 5-page site, charged the duplicates against ``--max-pages``, and wrote a
checkpoint saying ``scraped: 8, visited: 5``. On a cyclic site each duplicate
rediscovers the same links, so the frontier multiplies until the queue table
outgrows the crawl — a five-page site produced a 7.8 GB queue database.
``CrawlEngine._start`` now tracks in-flight URLs and refuses to re-admit one, so
the test is unmarked and asserts the fixed behaviour.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import TYPE_CHECKING

import pytest
from aiohttp import web

from protor.config import CHECKPOINT_FILENAME
from protor.crawler import Crawler
from protor.robots import clear_cache
from protor.utils import safe_filename

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

#: Opens a loopback socket. Deliberately not ``slow``: a five-page crawl is a
#: couple of seconds, and the point of these tests is that they run by default.
pytestmark = pytest.mark.integration


# ── the server ────────────────────────────────────────────────────────────────


def _page(title: str, links: Sequence[str] = ()) -> str:
    """One HTML page. Links live in plain paragraphs so the noise filter keeps them."""
    items = "".join(f'<p><a href="{href}">{href}</a></p>' for href in links)
    return (
        "<!DOCTYPE html><html><head>"
        f"<title>{title}</title></head>"
        f"<body><h1>{title}</h1><p>{title} body.</p>{items}</body></html>"
    )


class _Site:
    """
    A real site, plus the log of every request it received.

    ``requests`` is the whole log, robots.txt included; ``page_requests`` drops
    the robots policy fetch, which is a question about the host rather than a
    page of the site.
    """

    def __init__(self) -> None:
        self.pages: dict[str, str] = {}
        self.errors: dict[str, int] = {}
        self.delays: dict[str, float] = {}
        self.robots = "User-agent: *\nAllow: /\n"
        #: Body served at /sitemap.xml. Empty means "no sitemap here", which is
        #: the common case and must not stop a crawl.
        self.sitemap: str = ""
        self.requests: list[str] = []
        self.page_requests: list[str] = []
        self.robots_requests: list[str] = []
        self._port = 0

    # ── address ──

    def attach(self, port: int) -> None:
        self._port = port

    @property
    def netloc(self) -> str:
        return f"127.0.0.1:{self._port}"

    @property
    def base_url(self) -> str:
        return f"http://{self.netloc}/"

    def url(self, path: str) -> str:
        """The URL the crawler holds for *path* — its canonical form."""
        return f"{self.base_url}{path.lstrip('/')}"

    # ── content ──

    def add(self, path: str, title: str, links: Sequence[str] = ()) -> None:
        self.pages[path] = _page(title, links)

    def fail(self, path: str, status: int) -> None:
        self.errors[path] = status

    def heal(self, path: str) -> None:
        """Stop failing *path*, so a later run gets the page it is asking for."""
        self.errors.pop(path, None)

    def forbid(self, path: str) -> None:
        """Add a robots.txt rule disallowing *path*."""
        self.robots = f"User-agent: *\nDisallow: {path}\n"

    def slow(self, path: str, seconds: float) -> None:
        """
        Hold *path* open, so another page finishes while it is still in flight.

        That window is the only place the crawl can decide whether a page it
        already has in hand gets fetched a second time.
        """
        self.delays[path] = seconds

    # ── the log ──

    def count(self, path: str) -> int:
        return self.page_requests.count(path)

    def forget(self) -> None:
        """Drop the request log, so the next run's traffic can be read on its own."""
        self.requests.clear()
        self.page_requests.clear()
        self.robots_requests.clear()

    # ── handlers ──

    async def sitemap_handler(self, request: web.Request) -> web.Response:
        self.requests.append(request.path)
        if not self.sitemap:
            return web.Response(status=404, text="no sitemap")
        return web.Response(text=self.sitemap, content_type="application/xml")

    async def robots_handler(self, request: web.Request) -> web.Response:
        self.requests.append(request.path)
        self.robots_requests.append(request.path)
        return web.Response(text=self.robots, content_type="text/plain")

    async def page_handler(self, request: web.Request) -> web.Response:
        path = request.path
        self.requests.append(path)
        self.page_requests.append(path)
        delay = self.delays.get(path)
        if delay:
            await asyncio.sleep(delay)
        if path in self.errors:
            return web.Response(status=self.errors[path], text=f"{path} is broken")
        body = self.pages.get(path)
        if body is None:
            return web.Response(status=404, text="not found")
        return web.Response(text=body, content_type="text/html")


@pytest.fixture(autouse=True)
def _isolated_robots():
    """
    Give every test a cold robots cache.

    ``protor.robots`` keeps one process-wide policy cache; without this, a port
    reused by a later test would inherit the previous test's rules.
    """
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
async def site():
    """Run a real HTTP server for the duration of a test."""
    state = _Site()
    app = web.Application()
    app.router.add_get("/robots.txt", state.robots_handler)
    app.router.add_get("/sitemap.xml", state.sitemap_handler)
    app.router.add_get("/{tail:.*}", state.page_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    server = web.TCPSite(runner, "127.0.0.1", 0)
    await server.start()
    state.attach(runner.addresses[0][1])
    try:
        yield state
    finally:
        await runner.cleanup()


# ── driving a crawl ───────────────────────────────────────────────────────────


async def _crawl(
    site: _Site,
    output_dir: Path,
    *,
    max_pages: int,
    resume: bool = False,
    use_sitemaps: bool = False,
) -> None:
    """
    Run the crawl the way the CLI does, against the live server.

    ``Crawler.crawl()`` owns its event loop via ``asyncio.run()``, so it cannot
    run on the loop the test server is on; and the queue's SQLite connection
    belongs to the thread that opened it, so the ``Crawler`` has to be built
    there too. A worker thread satisfies both, and is a faithful reproduction
    rather than a workaround: the CLI is a separate process with its own loop
    talking to this one over TCP.

    ``crawl()`` rather than ``_run()`` because it is what also writes the
    checkpoint and closes the queue — the two things a resumed run depends on.
    ``live=False`` for the reason every crawler test turns rendering off: the
    behaviour under test is traffic and files, not the display.
    """

    def run() -> None:
        Crawler(
            site.base_url,
            max_pages=max_pages,
            output_dir=output_dir,
            resume=resume,
            live=False,
            use_sitemaps=use_sitemaps,
        ).crawl()

    await asyncio.to_thread(run)


# ── reading what the crawl left behind ────────────────────────────────────────


def _visited(output_dir: Path) -> dict[str, int]:
    """``{url: success}`` as the queue database holds it — the crawl's state of record."""
    conn = sqlite3.connect(str(output_dir / "crawl_queue.db"))
    try:
        return {url: success for url, success in conn.execute("SELECT url, success FROM visited")}
    finally:
        conn.close()


def _summary(output_dir: Path) -> dict:
    return json.loads((output_dir / CHECKPOINT_FILENAME).read_text(encoding="utf-8"))


def _site_dir(output_dir: Path, site: _Site) -> Path:
    return output_dir / safe_filename(site.netloc)


def _manifests(output_dir: Path, site: _Site) -> list[str]:
    """Names of the manifests the crawl wrote. The root page's is ``manifest.json``."""
    return sorted(p.name for p in _site_dir(output_dir, site).glob("*.json"))


# ── the sites ─────────────────────────────────────────────────────────────────

#: Five pages, one outgoing link each. A tree, so nothing is ever linked twice:
#: a request count and a page count are the same number.
TREE = ["/", "/about.html", "/blog.html", "/pricing.html", "/contact.html"]


def _tree(site: _Site) -> None:
    """index → about → blog → pricing → contact. No URL is ever handed to the crawler."""
    site.add("/", "Index", ["/about.html"])
    site.add("/about.html", "About", ["/blog.html"])
    site.add("/blog.html", "Blog", ["/pricing.html"])
    site.add("/pricing.html", "Pricing", ["/contact.html"])
    site.add("/contact.html", "Contact")


#: The same five pages wired into a cycle: the index links to every child, and
#: every child links back to the index and on to one sibling. Twelve edges.
CYCLE = ["/", "/about.html", "/blog.html", "/pricing.html", "/contact.html"]


def link_graph(site: _Site) -> None:
    site.add("/", "Index", CYCLE[1:])
    for i, path in enumerate(CYCLE[1:], start=1):
        name = path.strip("/").removesuffix(".html").title()
        site.add(path, name, ["/", CYCLE[1 + i] if i < len(CYCLE) - 1 else CYCLE[1]])


# ── a plain crawl ─────────────────────────────────────────────────────────────


class TestPlainCrawl:
    async def test_the_crawl_reaches_the_pages_it_was_not_given(self, site, tmp_path):
        """
        Link discovery, end to end.

        None of these four URLs is seeded anywhere: they exist only as
        ``<a href>`` in the index the server serves. If the harvest, the
        canonicalisation or the domain filter broke, the server would never be
        asked for them.
        """
        link_graph(site)
        out = tmp_path / "crawl"

        await _crawl(site, out, max_pages=10)

        assert set(site.page_requests) == set(CYCLE)
        assert site.robots_requests, "the crawl must consult robots.txt before fetching"
        assert set(site.requests) == set(CYCLE) | {"/robots.txt"}, (
            "the crawl talked to the host about nothing else"
        )

    async def test_a_cycle_produces_no_duplicate_request(self, site, tmp_path):
        """
        Five pages, twelve edges, five requests.

        Every child links back to the index and on to a sibling, so a crawler
        that re-enqueued on discovery alone would request the index four more
        times. Counting requests rather than pages is what makes the backlinks
        visible; per-path counts alone would still pass with the index fetched
        once and its siblings twice.
        """
        link_graph(site)
        out = tmp_path / "crawl"

        await _crawl(site, out, max_pages=10)

        assert len(site.page_requests) == len(CYCLE), site.page_requests
        assert all(site.count(path) == 1 for path in CYCLE)
        assert site.count("/") == 1, "four backlinks to the index, one request"

    async def test_the_crawl_leaves_the_artefacts_the_cli_promises(self, site, tmp_path):
        """
        What ``protor crawl -o DIR`` leaves behind: the queue database, the
        checkpoint summary, and a per-page HTML file plus manifest under a
        directory named for the host.

        The names come from the code that writes them — ``CHECKPOINT_FILENAME``,
        ``safe_filename``, ``page_filename``, ``manifest_filename`` — rather than
        being spelled out here, so this asserts the CLI's contract instead of a
        copy of it.
        """
        _tree(site)
        out = tmp_path / "crawl"

        await _crawl(site, out, max_pages=10)

        assert (out / "crawl_queue.db").is_file()

        summary = _summary(out)
        assert summary["start_url"] == site.base_url
        assert summary["max_pages"] == 10
        assert summary["scraped"] == len(TREE)
        assert summary["visited"] == len(TREE)
        assert summary["queued"] == 0

        site_dir = _site_dir(out, site)
        index_html = site_dir / "index.html"
        assert index_html.is_file()
        assert "<title>Index</title>" in index_html.read_text(encoding="utf-8")

        index = json.loads((site_dir / "manifest.json").read_text(encoding="utf-8"))
        assert index["url"] == site.base_url
        assert index["metadata"]["title"] == "Index"
        assert index["bytes_received"] > 0

        for child in TREE[1:]:
            stem = child.strip("/").removesuffix(".html")
            saved = site_dir / f"{stem}.html"
            assert saved.is_file(), f"{child} was scraped but not saved"
            assert f"<title>{stem.title()}</title>" in saved.read_text(encoding="utf-8")
            manifest = json.loads((site_dir / f"{stem}.manifest.json").read_text(encoding="utf-8"))
            assert manifest["url"] == site.url(child)
            assert manifest["metadata"]["title"] == stem.title()

        assert _manifests(out, site) == [
            "about.manifest.json",
            "blog.manifest.json",
            "contact.manifest.json",
            "manifest.json",
            "pricing.manifest.json",
        ]


# ── resume ────────────────────────────────────────────────────────────────────


class TestResume:
    async def test_a_resumed_crawl_issues_no_requests_at_all(self, site, tmp_path):
        """
        The core gap: nothing had ever driven ``--resume`` against a server.

        A completed crawl leaves an empty queue and a full ``visited`` table, so a
        resumed run has nothing to admit and must not touch the network. The
        robots cache is cleared first, because a real ``protor crawl --resume`` is
        a new process and starts cold: that makes ``site.requests`` here a
        faithful record of what a fresh resumed run would put on the wire, with
        no allowance for a policy re-fetch hiding a page fetch.

        The cyclic site is deliberate — the first run is the messiest one the
        crawler can produce, and the resumed run must still be silent after it.
        """
        link_graph(site)
        out = tmp_path / "crawl"

        await _crawl(site, out, max_pages=10)
        assert set(site.page_requests) == set(CYCLE)

        site.forget()
        clear_cache()
        await _crawl(site, out, max_pages=10, resume=True)

        assert site.requests == [], "a page that already succeeded must not be re-scraped"
        # Nothing changed on disk either: the same five rows, still successful.
        assert _visited(out) == {site.url(p): 1 for p in CYCLE}

    async def test_a_resumed_crawl_keeps_reporting_the_whole_crawl(self, site, tmp_path):
        """
        The headline number belongs to the crawl, not to the run: the summary a
        resumed run writes still has to report every page the crawl has scraped,
        so a user who re-runs ``protor crawl`` after a crash is not told the
        crash undid the work.
        """
        _tree(site)
        out = tmp_path / "crawl"

        await _crawl(site, out, max_pages=10)
        site.forget()
        await _crawl(site, out, max_pages=10, resume=True)

        summary = _summary(out)
        assert summary["scraped"] == len(TREE)
        assert summary["visited"] == len(TREE)
        assert _visited(out) == {site.url(p): 1 for p in TREE}

    async def test_a_seed_url_that_failed_is_retried(self, site, tmp_path):
        """
        The complement of the rule above, and the one case where a resumed run
        legitimately does hit the network.

        ``_CrawlQueue._SEEN_SQL`` treats a row that failed in an *earlier* run as
        unseen, so retrying it is the point; and ``Crawler.__init__`` seeds the
        start URL through ``enqueue`` on every start. A seed that 404'd is
        therefore re-requested, and is still recorded as a failure. The
        queue-level half of this is covered in ``test_regressions.py``; what only
        a real run shows is that the retry reaches the wire — and that a 404 is
        not retried within the same run.
        """
        site.fail("/", 404)
        out = tmp_path / "crawl"

        await _crawl(site, out, max_pages=5)
        assert site.page_requests == ["/"], "404 is not retryable, so one request"
        assert _visited(out) == {site.base_url: 0}

        site.forget()
        clear_cache()
        await _crawl(site, out, max_pages=5, resume=True)

        assert site.page_requests == ["/"], "a page that failed must be retried"
        assert _visited(out) == {site.base_url: 0}, "and still recorded as a failure"


# ── --max-pages across runs ───────────────────────────────────────────────────


class TestMaxPagesAcrossRuns:
    async def test_a_resumed_crawl_cannot_push_the_total_past_max_pages(self, site, tmp_path):
        """
        ``--max-pages`` is a ceiling on the crawl, not on each run.

        ``Crawler._run`` prices the engine's budget as
        ``max_pages - state.scraped``. Handing the engine a fresh ``max_pages``
        instead let a resumed crawl finish with twice the requested pages — the
        regression this file exists to catch, visible here as observed traffic
        rather than as a constructor argument.
        """
        _tree(site)
        out = tmp_path / "crawl"
        ceiling = 3

        await _crawl(site, out, max_pages=ceiling)
        first_pass = list(site.page_requests)
        assert len(first_pass) == ceiling

        site.forget()
        clear_cache()
        await _crawl(site, out, max_pages=ceiling, resume=True)

        assert len(first_pass) + len(site.page_requests) <= ceiling
        assert len(_visited(out)) == ceiling, "the crawl went past its own ceiling"
        assert _summary(out)["scraped"] == ceiling

    async def test_a_raised_budget_finishes_the_crawl_without_rescraping(self, site, tmp_path):
        """
        The same ceiling read from the other side: raising the budget lets the
        resume spend the difference and stop at the new limit, with every page
        still fetched exactly once across the two runs.
        """
        _tree(site)
        out = tmp_path / "crawl"

        await _crawl(site, out, max_pages=2)
        first_pass = list(site.page_requests)
        assert len(first_pass) == 2

        site.forget()
        clear_cache()
        await _crawl(site, out, max_pages=5, resume=True)

        everything = [*first_pass, *site.page_requests]
        assert len(everything) == 5, "the crawl must stop at the new ceiling"
        assert len(set(everything)) == 5, "a page was scraped by both runs"
        assert set(everything) <= set(TREE)
        assert len(_visited(out)) == 5
        assert _summary(out)["scraped"] == 5


# ── pages that do not come back ───────────────────────────────────────────────


class TestFailingPages:
    async def test_a_500_is_recorded_as_an_error_and_the_crawl_keeps_going(self, site, tmp_path):
        """
        A 500 must cost a page its status, not its place in the run.

        The chain is index → alpha → bravo → charlie, with the 500 hanging off
        alpha. ``charlie`` is reachable only through a link found on a page
        scraped *after* the failure was recorded, so a crawl that stopped, or
        swallowed the exception out of the loop, would never ask for it. That the
        server saw the request at all is the evidence.
        """
        site.add("/", "Index", ["/alpha.html"])
        site.add("/alpha.html", "Alpha", ["/bravo.html", "/broken.html"])
        site.add("/bravo.html", "Bravo", ["/charlie.html"])
        site.add("/charlie.html", "Charlie")
        site.fail("/broken.html", 500)
        out = tmp_path / "crawl"

        await _crawl(site, out, max_pages=10)

        assert "/charlie.html" in site.page_requests, "the crawl stopped at the failure"
        # 500 is retryable, so the server saw it more than once — and the run
        # still finished rather than spinning on it.
        assert site.count("/broken.html") >= 2

        rows = _visited(out)
        assert rows[site.url("/broken.html")] == 0, "a 500 is not a successful scrape"
        assert rows[site.url("/charlie.html")] == 1
        assert sum(rows.values()) == 4, "index, alpha, bravo, charlie"

        site_dir = _site_dir(out, site)
        assert not (site_dir / "broken.html").exists(), "a failed page must leave no HTML"
        assert not (site_dir / "broken.manifest.json").exists()
        assert _manifests(out, site) == [
            "alpha.manifest.json",
            "bravo.manifest.json",
            "charlie.manifest.json",
            "manifest.json",
        ]

        summary = _summary(out)
        assert summary["scraped"] == 4
        assert summary["visited"] == 5

    async def test_a_404_is_an_error_row_and_leaves_no_artefact(self, site, tmp_path):
        """
        404 is not in ``RETRYABLE_STATUS``, so the server must be asked exactly
        once — a crawl that retried a missing page would triple its traffic
        against a host that has already answered definitively.
        """
        site.add("/", "Index", ["/here.html", "/gone.html"])
        site.add("/here.html", "Here")
        site.fail("/gone.html", 404)
        out = tmp_path / "crawl"

        await _crawl(site, out, max_pages=10)

        assert site.count("/gone.html") == 1, "404 must not be retried"
        assert site.count("/here.html") == 1, "and must not cost its neighbour"

        rows = _visited(out)
        assert rows[site.url("/gone.html")] == 0
        assert rows[site.url("/here.html")] == 1

        site_dir = _site_dir(out, site)
        assert (site_dir / "here.html").is_file()
        assert not (site_dir / "gone.html").exists()
        assert _manifests(out, site) == ["here.manifest.json", "manifest.json"]
        assert _summary(out)["scraped"] == 2

    async def test_a_robots_disallowed_page_is_blocked_and_never_requested(self, site, tmp_path):
        """
        The other half of "recorded as an error row": a page the host forbids is
        blocked *before* the request, so the server log is the proof — the URL
        never appears in it. It is still recorded, and not as a success, so the
        crawl's own accounting agrees with the host's rules rather than quietly
        counting the URL as done.

        Recorded as "not attempted" rather than "failed" (``-1`` rather than
        ``0``): nothing was asked for, so a resume retrying the failures must not
        put it back at the head of the queue to be refused identically.
        """
        site.forbid("/private.html")
        site.add("/", "Index", ["/public.html", "/private.html"])
        site.add("/public.html", "Public")
        site.add("/private.html", "Private")
        out = tmp_path / "crawl"

        await _crawl(site, out, max_pages=10)

        assert "/private.html" not in site.page_requests, "a disallowed URL must not be fetched"
        assert site.count("/public.html") == 1

        rows = _visited(out)
        assert rows[site.url("/private.html")] == -1, "blocked is recorded, not scraped"
        assert rows[site.url("/public.html")] == 1

        site_dir = _site_dir(out, site)
        assert not (site_dir / "private.html").exists()
        assert _manifests(out, site) == ["manifest.json", "public.manifest.json"]
        assert _summary(out)["scraped"] == 2


class TestFreshVersusResume:
    """
    A plain `protor crawl URL` means "crawl it"; `--resume` means "carry on".

    The queue database is opened unconditionally, so an earlier run's rows used
    to suppress a second crawl entirely — the user asked for a crawl and got
    zero requests and no explanation. Every crawl here completes the whole tree,
    so "resume has nothing to do" is a statement about the mode and not about
    leftover budget.
    """

    async def test_a_second_plain_crawl_actually_crawls(self, site, tmp_path):
        _tree(site)
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path)
        assert site.page_requests, "the first crawl must have fetched something"

        site.forget()
        clear_cache()
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path)
        assert site.page_requests, (
            "a fresh crawl over a populated output directory did no work at all"
        )

    async def test_resume_still_does_nothing_when_there_is_nothing_left(self, site, tmp_path):
        _tree(site)
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path)
        site.forget()
        clear_cache()

        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path, resume=True)
        assert site.page_requests == [], "resume had nothing left to do"

    async def test_the_two_modes_disagree_on_the_same_directory(self, site, tmp_path):
        """One crawl, then the same command with and without --resume."""
        _tree(site)
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path)

        site.forget()
        clear_cache()
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path, resume=True)
        resumed = len(site.page_requests)

        site.forget()
        clear_cache()
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path)
        fresh = len(site.page_requests)

        assert resumed == 0, f"resume should find nothing to do, made {resumed} requests"
        assert fresh == len(TREE), f"a fresh crawl should re-crawl, made {fresh} requests"

    async def test_a_page_that_failed_is_retried_on_resume(self, site, tmp_path):
        """
        The successes are done by definition, so a retry is all a resume has left.

        A page that failed in an earlier run used to be unreachable again: it sat
        in ``visited``, so no admission check would re-admit it, and the only way
        back was for some other page to link to it. A 502 that has since healed
        therefore stayed a failure forever.
        """
        _tree(site)
        site.fail("/about.html", 502)
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path)
        assert _visited(tmp_path)[site.url("/about.html")] == 0, "it failed first time"

        site.heal("/about.html")
        site.forget()
        clear_cache()
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path, resume=True)

        assert "/about.html" in site.page_requests, f"never retried: {site.page_requests}"
        assert _visited(tmp_path)[site.url("/about.html")] == 1, "the retry did not take"

    async def test_a_failure_is_retried_on_the_next_run_and_not_in_a_loop(self, site, tmp_path):
        """
        One retry on the following run, then the crawl moves on.

        A 404 rather than a 502 on purpose — 502 is in the fetcher's retryable
        set, so one crawl-level attempt would already be three requests on the
        wire and this would be measuring the fetcher's policy, not the queue's.

        The loop half is the point: within the resumed run the failure is
        recorded again and nothing links it back in, so it is requested once and
        the crawl finishes. Without the run cutoff in ``_SEEN_SQL`` a page that
        links to itself would be re-queued by every page that links to it.
        """
        _tree(site)
        site.fail("/about.html", 404)
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path)
        assert site.page_requests.count("/about.html") == 1

        site.forget()
        clear_cache()
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path, resume=True)

        assert site.page_requests.count("/about.html") == 1, site.page_requests
        assert _visited(tmp_path)[site.url("/about.html")] == 0
        assert site.page_requests == ["/about.html"], "and nothing else was re-fetched"

    async def test_resume_does_not_refetch_the_pages_that_worked(self, site, tmp_path):
        """The retry is for failures, not a second pass over the successes."""
        _tree(site)
        site.fail("/about.html", 404)
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path)

        site.forget()
        clear_cache()
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path, resume=True)

        assert site.page_requests == ["/about.html"], site.page_requests

    async def test_pages_from_the_earlier_crawl_are_still_on_disk(self, site, tmp_path):
        """Resetting the crawl state must not delete what was already saved."""
        _tree(site)
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path)
        site_dir = _site_dir(tmp_path, site)
        before = sorted(p.name for p in site_dir.glob("*.html"))
        assert before, "the first crawl wrote pages"

        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path)
        after = sorted(p.name for p in site_dir.glob("*.html"))
        assert after == before, "a fresh crawl must not delete what was already saved"


class TestInflightDeduplication:
    """
    A page already in flight must not be fetched again because of how a link spelled it.

    A dispatched page leaves the queue, so it is in neither `queue` nor `visited`
    while it is being fetched. The engine keeps its own set of in-flight URLs to
    close that window; the set holds the canonical URLs the queue handed out, so
    the comparison has to be canonical too. `/docs/index.html` and `/docs/` are
    the same page, and a raw string comparison found no overlap between them —
    an extra request, and a crawl reporting one more page than the site has.
    """

    async def test_one_page_spelled_two_ways_is_fetched_once(self, site, tmp_path):
        site.add("/", "Index", ["/docs/", "/other/"])
        site.add("/docs/", "Docs")
        site.add("/other/", "Other", ["/docs/index.html"])
        # Long enough that /other/ finishes while /docs/ is still in flight.
        site.slow("/docs/", 0.3)

        await _crawl(site, max_pages=10, output_dir=tmp_path)

        assert site.count("/docs/") == 1, (
            f"/docs/ was fetched {site.count('/docs/')} times: {site.page_requests}"
        )
        assert site.count("/docs/index.html") == 0, site.page_requests
        assert len(site.page_requests) == len(set(site.page_requests)), (
            f"a page was requested twice: {site.page_requests}"
        )


class TestDomainFilter:
    """A host is case-insensitive, and the queue canonicalises it as such."""

    async def test_a_mixed_case_host_is_not_treated_as_off_domain(self, site, tmp_path):
        """
        ``canonicalize_url`` lowercases the host; the allowed domain came from
        ``urlparse`` of the URL as typed, so it kept the case.

        Compared raw, a seed of ``https://EXAMPLE.com/`` was measured against its
        own canonical form, rejected as off-domain, and the crawl reported zero
        pages scraped beside a single "off-domain" row — the seed skipping itself,
        with nothing to suggest why.
        """
        site.add("/", "Index", ["/about.html"])
        site.add("/about.html", "About")

        start = f"http://{site.netloc.upper()}/"

        def run() -> None:
            # Off the event loop, as _crawl does: crawl() owns its own.
            Crawler(start, max_pages=5, output_dir=tmp_path, live=False).crawl()

        await asyncio.to_thread(run)

        summary = _summary(tmp_path)
        assert summary["scraped"] == 2, f"the seed skipped itself: {summary}"


class TestResumeDoesNotRequeueFilteredUrls:
    """
    A resume retries what was *asked for* and failed.

    Off-domain links, robots-refusals and ad-blocked URLs were never requested,
    so a retry is refused identically. Re-queueing them put the whole filtered
    set at the head of the queue on every resumed run — dispatched, skipped, and
    re-skipped — while the budget they should have been fetching with went
    unspent. They are recorded distinctly, as not attempted.
    """

    async def test_a_blocked_page_is_not_retried(self, site, tmp_path):
        site.forbid("/private.html")
        site.add("/", "Index", ["/public.html", "/private.html"])
        site.add("/public.html", "Public", ["/private.html"])
        site.add("/private.html", "Private")

        await _crawl(site, max_pages=10, output_dir=tmp_path)
        assert _visited(tmp_path)[site.url("/private.html")] == -1

        site.forget()
        clear_cache()
        await _crawl(site, max_pages=10, output_dir=tmp_path, resume=True)

        assert "/private.html" not in site.page_requests, (
            f"the filtered URL was re-queued: {site.page_requests}"
        )

    async def test_a_fetch_failure_is_still_retried(self, site, tmp_path):
        """The counterpart: a real failure must not lose its retry."""
        _tree(site)
        site.fail("/about.html", 502)
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path)
        assert _visited(tmp_path)[site.url("/about.html")] == 0, "a fetch failure is 0"

        site.heal("/about.html")
        site.forget()
        clear_cache()
        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path, resume=True)

        assert "/about.html" in site.page_requests, "the fetch failure lost its retry"


class TestRobotsIsEvaluatedAgainstTheSentAgent:
    """
    The rules are asked about the identity the request then uses.

    ``check_robots`` documents that the ``user_agent`` it is given must be the
    string the request actually sends: evaluating the ``*`` group while
    transmitting a browser User-Agent asks the site about a policy it never
    agreed to. urllib reduces the argument to the token before the first "/", so
    the full browser string scores as ``mozilla`` — matching a ``User-agent:
    Mozilla`` group, never a ``User-agent: Googlebot`` one, and leaving a
    ``User-agent: * Disallow:`` site gated by the wrong group.

    The agent is pinned rather than drawn from the rotation pool: fifteen real
    browser strings would make the outcome depend on which one the crawl picked.
    """

    async def test_a_group_specific_disallow_is_honoured(self, site, tmp_path, monkeypatch):
        import protor.engine as engine_mod

        monkeypatch.setattr(engine_mod, "random_user_agent", lambda: "TestAgent/1.0")
        # The site allows its own named agent nothing and everyone else /.
        site.robots = "User-agent: TestAgent\nDisallow: /private.html\n\nUser-agent: *\nAllow: /\n"
        site.add("/", "Index", ["/private.html"])
        site.add("/private.html", "Private")

        await _crawl(site, max_pages=10, output_dir=tmp_path)

        assert "/private.html" not in site.page_requests, (
            f"fetched despite the rule for the agent it sent: {site.page_requests}"
        )

    async def test_a_page_the_sent_agent_is_allowed_is_fetched(self, site, tmp_path, monkeypatch):
        """The guard must not have become a blanket refusal."""
        import protor.engine as engine_mod

        monkeypatch.setattr(engine_mod, "random_user_agent", lambda: "TestAgent/1.0")
        site.robots = "User-agent: TestAgent\nAllow: /\n"
        site.add("/", "Index", ["/public.html"])
        site.add("/public.html", "Public")

        await _crawl(site, max_pages=10, output_dir=tmp_path)

        assert "/public.html" in site.page_requests, site.page_requests


class TestMaxPagesBoundsRequests:
    """
    `--max-pages` bounds requests, not just successes.

    The spawn loop compares ``stats.total`` — scraped + errors + blocked — against
    the ceiling, so a *skipped* URL would not advance it and the ceiling would
    bound nothing. Nothing can skip: the parser yields same-host links only, so a
    recursive crawl never hands the domain filter a URL to reject. That is why
    this holds, and it is an invariant rather than luck — loosen the parser's host
    check and the ceiling quietly becomes a suggestion with nothing to say so.

    Asserted on the server's own request log, which is the only count no internal
    counter can flatter.
    """

    async def test_a_never_ending_frontier_still_stops_at_the_ceiling(self, site, tmp_path):
        # Every page links to every page, so the frontier never empties and the
        # crawl can only stop because it hit the ceiling.
        pages = ["/"] + [f"/p{i}.html" for i in range(12)]
        for path in pages:
            site.add(path, path, [p for p in pages if p != path])

        await _crawl(site, max_pages=4, output_dir=tmp_path)

        assert len(site.page_requests) <= 4, (
            f"ceiling of 4 issued {len(site.page_requests)} requests: {site.page_requests}"
        )
        assert _summary(tmp_path)["scraped"] <= 4

    async def test_a_failed_page_still_counts_against_the_ceiling(self, site, tmp_path):
        """
        Failures count too: the point is to bound work, not successes.

        A 404 rather than a 500, because 5xx is in the fetcher's retryable set —
        one page attempt then costs three wire requests, which is deliberate and
        is why the ceiling is documented over *pages* and not over HTTP requests.
        """
        pages = ["/"] + [f"/p{i}.html" for i in range(12)]
        for path in pages:
            site.add(path, path, [p for p in pages if p != path])
        for path in pages[1:4]:
            site.fail(path, 404)

        await _crawl(site, max_pages=3, output_dir=tmp_path)

        assert len(site.page_requests) <= 3, (
            f"3 failures should have ended the crawl, not {len(site.page_requests)} requests"
        )
        assert _summary(tmp_path)["scraped"] <= 1, "the failures are not successes"

    async def test_the_ceiling_is_over_pages_not_wire_requests(self, site, tmp_path):
        """
        One page attempt may cost several requests, by design.

        A 502 is retried up to ``MAX_RETRIES`` before the page is recorded as
        failed, so the request log exceeds ``--max-pages`` while the number of
        *pages* attempted does not. Documented over pages for exactly this
        reason; asserted here so the distinction cannot quietly change.
        """
        pages = ["/"] + [f"/p{i}.html" for i in range(12)]
        for path in pages:
            site.add(path, path, [p for p in pages if p != path])
        # The first page the crawl reaches, so a ceiling of 2 is sure to include it.
        site.fail("/p0.html", 502)

        await _crawl(site, max_pages=2, output_dir=tmp_path)

        visited = _visited(tmp_path)
        assert len(visited) <= 2, f"more pages attempted than the ceiling: {visited}"
        assert len(site.page_requests) > len(visited), (
            "expected the retried page to cost more than one request"
        )


class TestSitemapSeeding:
    """
    A link-walk only reaches what pages happen to link to.

    On a documentation site that is the sidebar; on a shop it is the top nav. The
    pages in a sitemap and nothing else — an unlisted changelog, a product retired
    from the nav last year — are exactly the ones a link-walk structurally cannot
    find, and they are a large share of what a site has.
    """

    def _sitemap(self, site: _Site, *paths: str) -> None:
        site.sitemap = (
            '<?xml version="1.0"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            + "".join(f"<url><loc>{site.url(p)}</loc></url>" for p in paths)
            + "</urlset>"
        )

    async def test_sitemap_pages_the_homepage_never_links_to_are_crawled(self, site, tmp_path):
        site.add("/", "Index", ["/about.html"])
        site.add("/about.html", "About")
        site.add("/changelog.html", "Changelog")
        site.add("/legacy/old-product.html", "Old product")
        self._sitemap(site, "/changelog.html", "/legacy/old-product.html")

        await _crawl(site, max_pages=10, output_dir=tmp_path, use_sitemaps=True)

        assert "/changelog.html" in site.page_requests, "sitemap page was missed"
        assert "/legacy/old-product.html" in site.page_requests
        assert _summary(tmp_path)["scraped"] == 4

    async def test_without_the_flag_the_same_site_is_not_reached(self, site, tmp_path):
        """The control: the flag is what changes the outcome, not the sitemap."""
        site.add("/", "Index", ["/about.html"])
        site.add("/about.html", "About")
        site.add("/changelog.html", "Changelog")
        self._sitemap(site, "/changelog.html")

        await _crawl(site, max_pages=10, output_dir=tmp_path, use_sitemaps=False)

        assert "/changelog.html" not in site.page_requests

    async def test_a_site_with_no_sitemap_still_crawls(self, site, tmp_path):
        """A sitemap is an optimisation; its absence must not stop the walk."""
        _tree(site)

        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path, use_sitemaps=True)

        assert _summary(tmp_path)["scraped"] == len(TREE), "the crawl stopped without a sitemap"

    async def test_a_malformed_sitemap_does_not_stop_the_crawl(self, site, tmp_path):
        _tree(site)
        site.sitemap = "<html>not xml at all"

        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path, use_sitemaps=True)

        assert _summary(tmp_path)["scraped"] == len(TREE)

    async def test_off_domain_sitemap_entries_are_dropped(self, site, tmp_path):
        """A sitemap can list a sibling property or a CDN; those are not ours."""
        _tree(site)
        site.add("/about.html", "About")
        site.sitemap = (
            '<?xml version="1.0"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            "<url><loc>https://elsewhere.example/page</loc></url>"
            f"<url><loc>{site.url('/about.html')}</loc></url>"
            "</urlset>"
        )

        await _crawl(site, max_pages=len(TREE), output_dir=tmp_path, use_sitemaps=True)

        assert all("elsewhere.example" not in p for p in site.page_requests), site.page_requests

    async def test_a_sitemap_cannot_overrun_the_page_ceiling(self, site, tmp_path):
        """5,000 listed URLs and --max-pages 4 is four requests, not five thousand."""
        site.add("/", "Index", [])
        pages = [f"/p{i}.html" for i in range(5000)]
        self._sitemap(site, *pages)
        for path in pages:
            site.add(path, path)

        await _crawl(site, max_pages=4, output_dir=tmp_path, use_sitemaps=True)

        assert len(site.page_requests) <= 4, f"{len(site.page_requests)} requests for --max-pages 4"
