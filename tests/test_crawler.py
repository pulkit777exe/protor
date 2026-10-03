"""Unit tests for protor.crawler module"""

import contextlib
import json
import os
import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from protor.config import CHECKPOINT_FILENAME
from protor.crawler import Crawler, _CrawlLog, _CrawlQueue, _render, _State
from protor.engine import CrawlEngine
from protor.fetcher import FetchResult
from protor.utils import canonicalize_url

#: Disable in-place rendering: these tests cover crawl behaviour, not
#: display, and driving the real no-live path keeps them honest about
#: piped/CI output.
NO_LIVE = patch.dict(os.environ, {"PROTOR_NO_LIVE": "1"})

ROBOTS_PATCH = patch("protor.engine.check_robots", new_callable=AsyncMock, return_value=True)


class _NoDelayLimiter:
    """Stand-in for the polite per-domain limiter used by the big-crawl tests."""

    async def wait(self, domain):
        return


#: The real limiter spaces same-domain requests CRAWLER_DELAY apart, which is 100
#: seconds of sleeping for a 400-page crawl and says nothing about checkpoints.
NO_DELAY = patch("protor.crawler.DomainRateLimiter", lambda *a, **k: _NoDelayLimiter())


@contextlib.contextmanager
def stubbed_network(fetched):
    """Stub out everything a crawl would otherwise reach out for."""
    page = "<html><body>hi</body></html>"

    async def fake_fetch(session, url, **kwargs):
        fetched.append(url)
        return FetchResult(text=page, nbytes=len(page))

    with (
        patch("protor.engine.aiohttp.ClientSession"),
        patch("protor.engine.fetch", new_callable=AsyncMock, side_effect=fake_fetch),
        NO_LIVE,
        ROBOTS_PATCH,
        NO_DELAY,
    ):
        yield


def seed(queue, count, start=1):
    """Queue *count* pages for the next run to chew through."""
    for i in range(start, start + count):
        queue.enqueue(f"https://example.com/p{i}")


class TestCrawlLog:
    def test_defaults(self):
        log = _CrawlLog(status="ok", domain="example.com")
        assert log.status == "ok"
        assert log.domain == "example.com"
        assert log.note == ""

    def test_with_note(self):
        log = _CrawlLog(status="err", domain="example.com", note="timeout")
        assert log.note == "timeout"


class TestState:
    def test_defaults(self):
        s = _State()
        assert s.scraped == 0
        assert s.errors == 0
        assert s.current == ""
        assert s.queue_n == 0
        assert s.max_pages == 10
        assert list(s.log) == []
        assert s.log_total == 0

    def test_custom_max_pages(self):
        s = _State(max_pages=50)
        assert s.max_pages == 50

    def test_log_is_independent(self):
        s1 = _State()
        s2 = _State()
        s1.log.append(_CrawlLog("ok", "a.com"))
        assert len(s2.log) == 0


class TestRender:
    def test_render_returns_group(self):
        state = _State(scraped=2, max_pages=10, queue_n=5)
        result = _render(state, "/tmp/output")
        assert result is not None

    def test_render_with_errors(self):
        state = _State(
            scraped=3,
            max_pages=10,
            errors=2,
            current="https://example.com/page",
            queue_n=3,
            log=[
                _CrawlLog("ok", "example.com"),
                _CrawlLog("err", "fail.com", note="timeout"),
            ],
        )
        result = _render(state, "/tmp/output")
        assert result is not None

    def test_render_empty_state(self):
        state = _State()
        result = _render(state, "/tmp/output")
        assert result is not None

    def test_render_with_blocked(self):
        state = _State(
            scraped=1,
            max_pages=5,
            blocked=2,
            log=[_CrawlLog("blocked", "tracker.com")],
        )
        result = _render(state, "/tmp/output")
        assert result is not None


class TestCrawlQueue:
    def test_enqueue_and_dequeue(self, tmp_path):
        q = _CrawlQueue(tmp_path / "test.db")
        q.enqueue("https://example.com/")
        url = q.dequeue()
        assert url == "https://example.com/"
        assert q.dequeue() is None
        q.close()

    def test_requeue_failed_only_requeues_failures(self, tmp_path):
        q = _CrawlQueue(tmp_path / "test.db")
        q.mark_visited("https://example.com/good", success=True)
        q.mark_visited("https://example.com/bad", success=False)

        assert q.requeue_failed() == 1
        assert q.dequeue() == "https://example.com/bad"
        assert q.dequeue() is None, "the page that worked is not queued again"
        q.close()

    def test_requeue_failed_returns_zero_when_nothing_failed(self, tmp_path):
        q = _CrawlQueue(tmp_path / "test.db")
        q.mark_visited("https://example.com/good", success=True)
        assert q.requeue_failed() == 0
        q.close()

    def test_requeue_failed_is_idempotent(self, tmp_path):
        """Called on a queue that already holds them must not double the rows."""
        q = _CrawlQueue(tmp_path / "test.db")
        q.mark_visited("https://example.com/bad", success=False)
        assert q.requeue_failed() == 1
        assert q.requeue_failed() == 0, "already queued"
        assert q.queue_size == 1
        q.close()

    def test_requeue_failed_fixes_the_queue_size_counter(self, tmp_path):
        """
        The counter answers ``queue_size`` on every admission check, so a bulk
        insert that does not move it leaves the live render and the budget
        disagreeing with the table.
        """
        q = _CrawlQueue(tmp_path / "test.db")
        for i in range(3):
            q.mark_visited(f"https://example.com/{i}", success=False)
        assert q.queue_size == 0

        assert q.requeue_failed() == 3
        assert q.queue_size == 3
        q.close()

    def test_a_requeued_failure_is_admitted_once_per_run(self, tmp_path):
        """
        One retry, not a loop.

        A failure from an earlier run is re-admitted; the one this run just
        recorded still sits behind ``_run_started``, so rediscovery cannot spin
        on it.
        """
        path = tmp_path / "test.db"
        q = _CrawlQueue(path)
        q.mark_visited("https://example.com/bad", success=False)
        q.close()

        again = _CrawlQueue(path)
        again.requeue_failed()
        assert again.dequeue() == "https://example.com/bad"
        again.mark_visited("https://example.com/bad", success=False)
        assert again.enqueue("https://example.com/bad") is False, "retried twice in one run"
        again.close()

    def test_deduplication(self, tmp_path):
        q = _CrawlQueue(tmp_path / "test.db")
        assert q.enqueue("https://example.com/") is True
        assert q.enqueue("https://example.com/") is False
        q.close()

    def test_canonical_variants_dedup(self, tmp_path):
        q = _CrawlQueue(tmp_path / "test.db")
        assert q.enqueue("https://example.com/index.html") is True
        assert q.enqueue("https://example.com/") is False
        assert q.enqueue("https://example.com/about#top") is True
        assert q.enqueue("https://example.com/about") is False
        q.close()

    def test_a_visited_url_is_never_queued_again(self, tmp_path):
        """Resume safety: a page the crawl already finished must not be re-queued,
        or it gets scraped a second time."""
        q = _CrawlQueue(tmp_path / "test.db")
        q.mark_visited("https://example.com/done", success=True)
        assert q.enqueue("https://example.com/done") is False
        assert q.enqueue("https://example.com/done#section") is False
        assert q.queue_size == 0
        q.close()

    def test_enqueue_canonicalises_once_and_probes_with_one_statement(self, tmp_path, monkeypatch):
        """
        The admission check runs once per discovered link, so it is the hottest
        path in the crawler. It used to canonicalise the URL three times and then
        probe the two tables with a SELECT each — all of it blocking sqlite on the
        event loop.
        """
        q = _CrawlQueue(tmp_path / "test.db")
        canonicalised = []
        monkeypatch.setattr(
            "protor.crawler.canonicalize_url",
            lambda url: canonicalised.append(url) or canonicalize_url(url),
        )
        statements = []
        q._conn.set_trace_callback(statements.append)

        def selects():
            return [s for s in statements if s.lstrip().upper().startswith("SELECT")]

        assert q.enqueue("https://example.com/Index.html#top") is True
        assert len(canonicalised) == 1
        assert len(selects()) == 1, "the duplicate check must be a single round trip"
        assert "visited" in selects()[0] and "queue" in selects()[0]

        # A known URL short-circuits after that same one statement.
        statements.clear()
        assert q.enqueue("https://example.com/") is False
        assert len(canonicalised) == 2
        assert len(selects()) == 1
        q.close()

    def test_visited_tracking(self, tmp_path):
        q = _CrawlQueue(tmp_path / "test.db")
        q.enqueue("https://example.com/")
        q.dequeue()
        q.mark_visited("https://example.com/", success=True)
        assert q.is_visited("https://example.com/")
        assert q.visited_count == 1
        assert q.success_count == 1
        q.close()

    def test_queue_size(self, tmp_path):
        q = _CrawlQueue(tmp_path / "test.db")
        q.enqueue("https://a.com/")
        q.enqueue("https://b.com/")
        q.enqueue("https://c.com/")
        assert q.queue_size == 3
        q.dequeue()
        assert q.queue_size == 2
        q.close()

    def test_repeat_marks_do_not_inflate_the_counter(self, tmp_path):
        """
        ``INSERT OR REPLACE`` always reports a change, so marking a page again
        counted it as another page: the live counter drifted away from COUNT(*),
        which is what reopening the database recomputes.
        """
        db = tmp_path / "test.db"
        q = _CrawlQueue(db)
        q.mark_visited("https://example.com/a", success=True)
        q.mark_visited("https://example.com/a", success=True)
        q.mark_visited("https://example.com/a", success=True)
        q.mark_visited("https://example.com/b", success=False)
        assert q.visited_count == 2
        assert q.success_count == 1
        q.close()

        reopened = _CrawlQueue(db)
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM visited").fetchone()[0]
        assert reopened.visited_count == rows, "the counter must match the rows"
        assert reopened.success_count == 1
        reopened.close()

    def test_a_repeat_mark_keeps_the_latest_outcome(self, tmp_path):
        """A page that failed and later succeeded still counts as a success."""
        q = _CrawlQueue(tmp_path / "test.db")
        q.mark_visited("https://example.com/a", success=False)
        q.mark_visited("https://example.com/a", success=True)
        assert q.visited_count == 1
        assert q.success_count == 1
        q.close()

    def test_queue_state_survives_reopening(self, tmp_path):
        """
        The database is the crawl's state of record: reopening it is all a resumed
        run needs, with no checkpoint file in sight.
        """
        db = tmp_path / "queue.db"
        q = _CrawlQueue(db)
        q.enqueue("https://example.com/")
        q.enqueue("https://example.com/about")
        assert q.dequeue() == "https://example.com/"
        q.mark_visited("https://example.com/", success=True)
        q.close()

        q2 = _CrawlQueue(db)
        assert q2.visited_count == 1
        assert q2.success_count == 1
        assert q2.queue_size == 1
        assert q2.is_visited("https://example.com/")
        assert not q2.is_queued("https://example.com/")
        assert q2.enqueue("https://example.com/about") is False, "already queued"
        assert q2.enqueue("https://example.com/") is False, "already scraped"
        q2.close()


class TestCheckpointCadence:
    @pytest.mark.asyncio
    async def test_interval_scales_with_the_page_budget(self, tmp_path):
        """
        A checkpoint is written from the crawl loop, so a fixed interval made the
        event-loop stall grow with the crawl: more writes, and each one dearer as
        the history behind it grew. The budget now decides.
        """
        seen = {}
        real_engine = CrawlEngine

        def spy(*args, **kwargs):
            seen["max_targets"] = kwargs["max_targets"]
            seen["checkpoint_interval"] = kwargs["checkpoint_interval"]
            return real_engine(*args, **kwargs)

        big = Crawler("https://example.com", max_pages=4000, output_dir=str(tmp_path))
        with patch("protor.crawler.CrawlEngine", spy), stubbed_network([]):
            await big._run()
        assert seen["max_targets"] == 4000
        assert seen["checkpoint_interval"] == 200

        small = Crawler("https://example.com", max_pages=8, output_dir=str(tmp_path / "small"))
        with patch("protor.crawler.CrawlEngine", spy), stubbed_network([]):
            await small._run()
        assert seen["checkpoint_interval"] == 5, "small crawls keep the old cadence"
        big._queue.close()
        small._queue.close()

    @pytest.mark.asyncio
    async def test_a_large_crawl_writes_a_bounded_number_of_checkpoints(self, tmp_path):
        """
        Counted, not timed: the number of writes has to be a function of the
        budget rather than of the page count. At the old fixed interval of 5 this
        crawl wrote 80 summaries; a 40,000-page one wrote ~8,000.
        """
        pages = 400
        c = Crawler("https://example.com", max_pages=pages, output_dir=str(tmp_path))
        seed(c._queue, pages - 1)

        writes = []
        real_save = c._save_checkpoint

        def spy():
            writes.append(1)
            real_save()

        c._save_checkpoint = spy
        fetched = []
        with stubbed_network(fetched):
            await c._run()
        c._queue.close()

        assert len(fetched) == pages
        assert c._state.scraped == pages
        assert len(writes) <= 20, "checkpoint count must not scale with the crawl"
        assert len(writes) >= 5, "checkpointing still has to happen"


class TestCheckpointFile:
    def test_the_summary_does_not_grow_with_the_crawl(self, tmp_path):
        """
        The crawler used to mirror the whole queue into JSON *and* SQLite — two
        stores holding the same rows, both written from the crawl loop. The
        summary is a report now, so its size cannot track the pages seen.
        """
        c = Crawler("https://example.com", max_pages=5, output_dir=str(tmp_path))
        for i in range(5):
            c._queue.mark_visited(f"https://example.com/p{i}", success=True)
        c._save_checkpoint()
        before = (tmp_path / CHECKPOINT_FILENAME).stat().st_size

        for i in range(2000):
            c._queue.mark_visited(f"https://example.com/many/{i}", success=True)
        c._save_checkpoint()
        after = (tmp_path / CHECKPOINT_FILENAME).stat().st_size
        c._queue.close()

        # Only the digits of the counters may move; the old summary carried one
        # line per URL and grew by ~60 kB over the same 2,000 pages.
        assert after - before < 64

    def test_the_summary_reports_the_crawl(self, tmp_path):
        c = Crawler("https://example.com", max_pages=5, output_dir=str(tmp_path))
        assert c._queue.dequeue() == "https://example.com/"
        c._queue.mark_visited("https://example.com/", success=True)
        seed(c._queue, 2)
        c._state.scraped = 1
        c._save_checkpoint()
        c._queue.close()

        summary = json.loads((tmp_path / CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
        assert summary["start_url"] == "https://example.com"
        assert summary["max_pages"] == 5
        assert summary["scraped"] == 1
        assert summary["queued"] == 2
        assert summary["visited"] == 1
        assert "https://example.com/p1" not in json.dumps(summary), "rows live in the database"


class TestCrawlerInit:
    def test_default_values(self, tmp_path):
        c = Crawler("https://example.com", output_dir=str(tmp_path))
        assert c.start_url == "https://example.com"
        assert c.max_pages == 10
        assert c._queue.queue_size == 1
        assert c._queue.is_queued("https://example.com")
        c._queue.close()

    def test_custom_values(self, tmp_path):
        c = Crawler("https://example.com", max_pages=50, output_dir=str(tmp_path / "crawl"))
        assert c.max_pages == 50
        assert c.output_dir == tmp_path / "crawl"
        c._queue.close()

    def test_output_dir_defaults_to_home_downloads(self):
        c = Crawler("https://example.com")
        assert "Downloads" in str(c.output_dir)
        assert "protor" in str(c.output_dir)
        c._queue.close()

    def test_resume_flag(self, tmp_path):
        c = Crawler("https://example.com", resume=True, output_dir=str(tmp_path))
        assert c.resume is True
        c._queue.close()

    def test_a_scraped_seed_url_is_not_requeued(self, tmp_path):
        """The seed is enqueued on every start, so on a *resume* it has to stay
        rejected once the crawl has seen it — otherwise --resume re-scrapes
        page one."""
        c = Crawler("https://example.com", output_dir=str(tmp_path))
        assert c._queue.dequeue() == "https://example.com/"
        c._queue.mark_visited("https://example.com/", success=True)
        c._queue.close()

        again = Crawler("https://example.com", max_pages=3, output_dir=str(tmp_path), resume=True)
        assert again._queue.queue_size == 0
        again._queue.close()

    def test_a_fresh_crawl_forgets_the_previous_one(self, tmp_path):
        """
        A plain `protor crawl URL` means "crawl it".

        The queue database is opened whether or not --resume was passed, so an
        earlier run's rows used to make a second crawl do nothing at all — zero
        requests, no explanation. Continuing is what --resume is for.
        """
        c = Crawler("https://example.com", output_dir=str(tmp_path))
        c._queue.mark_visited("https://example.com/", success=True)
        c._queue.close()

        fresh = Crawler("https://example.com", output_dir=str(tmp_path))
        assert fresh._queue.visited_count == 0, "previous crawl state must not survive"
        assert fresh._queue.queue_size == 1, "the seed is queued again"
        assert fresh._queue.dequeue() == "https://example.com/"
        fresh._queue.close()

    def test_a_fresh_crawl_keeps_the_pages_already_on_disk(self, tmp_path):
        """Only the crawl state is reset; artefacts a user may be using stay."""
        page = tmp_path / "crawler" / "example.com" / "index.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("<html>earlier run</html>")

        c = Crawler("https://example.com", output_dir=str(tmp_path / "crawler"))
        c._queue.mark_visited("https://example.com/", success=True)
        c._queue.close()

        Crawler("https://example.com", output_dir=str(tmp_path / "crawler"))._queue.close()
        assert page.exists()
        assert page.read_text() == "<html>earlier run</html>"

    def test_resume_leaves_the_state_alone(self, tmp_path):
        """The counterpart: --resume must never clear anything."""
        c = Crawler("https://example.com", output_dir=str(tmp_path))
        c._queue.mark_visited("https://example.com/", success=True)
        c._queue.close()

        resumed = Crawler("https://example.com", output_dir=str(tmp_path), resume=True)
        assert resumed._queue.visited_count == 1
        assert resumed._state.scraped == 1, "and the budget is priced from it"
        resumed._queue.close()

    def test_has_state_is_false_for_a_new_database(self, tmp_path):
        """The queue, not the Crawler: the crawler enqueues its seed by then."""
        from protor.crawler import _CrawlQueue

        q = _CrawlQueue(tmp_path / "q.db")
        assert q.has_state() is False
        q.enqueue("https://example.com/a")
        assert q.has_state() is True
        assert q.clear_state() == 0, "nothing had been visited yet"
        assert q.has_state() is False
        q.close()

    @pytest.mark.asyncio
    async def test_resume_does_not_reset_the_page_ceiling(self, tmp_path):
        """
        --max-pages is a ceiling for the crawl, not for each run.

        The engine was handed a fresh budget of max_pages on every run, so a
        resumed crawl finished with up to twice the requested pages while the
        report still claimed the limit.
        """
        seen = []
        real_engine = CrawlEngine

        def spy(*args, **kwargs):
            seen.append(kwargs["max_targets"])
            return real_engine(*args, **kwargs)

        c = Crawler("https://example.com", max_pages=3, output_dir=str(tmp_path), resume=True)
        # Pretend an earlier run already scraped two of the three pages.
        c._state.scraped = 2
        c._queue.mark_visited("https://example.com/a", success=True)
        c._queue.mark_visited("https://example.com/b", success=True)

        with patch("protor.crawler.CrawlEngine", spy), stubbed_network([]):
            await c._run()
        assert seen == [1], "only the one remaining page may be fetched"
        c._queue.close()

    def test_auto_scale_flag(self, tmp_path):
        c = Crawler("https://example.com", auto_scale=True, output_dir=str(tmp_path))
        assert c.auto_scale is True
        c._queue.close()


class TestResume:
    @pytest.mark.asyncio
    async def test_crash_between_checkpoints_never_rescrapes(self, tmp_path):
        """
        The crawl was interrupted before it wrote a summary, leaving nothing to
        resume from but the queue rows. The next run continues from them instead
        of starting over.
        """
        first_run = []
        c1 = Crawler("https://example.com", max_pages=4, output_dir=str(tmp_path))
        seed(c1._queue, 9)
        with stubbed_network(first_run):
            await c1._run()
        assert len(first_run) == 4
        # A crash: no summary is written, and the database is all that is left.
        c1._queue.close()
        assert not (tmp_path / CHECKPOINT_FILENAME).exists()

        second_run = []
        c2 = Crawler("https://example.com", max_pages=10, output_dir=str(tmp_path), resume=True)
        assert c2._state.scraped == 4, "resume counts the pages the database already has"
        with stubbed_network(second_run):
            await c2._run()
        assert c2._queue.visited_count == 10
        c2._queue.close()

        assert set(first_run).isdisjoint(second_run), "a page was scraped twice"
        assert len(second_run) == 6, "the crawl must finish the remaining budget"

    @pytest.mark.asyncio
    async def test_queue_rows_outrank_a_stale_summary(self, tmp_path):
        """
        A page that finished after the last summary was written is known to the
        database and not to the file. The rows have to win, or a resumed crawl
        re-scrapes work the database says is done.
        """
        c1 = Crawler("https://example.com", max_pages=2, output_dir=str(tmp_path))
        seed(c1._queue, 2)
        with stubbed_network([]):
            await c1._run()
        c1._save_checkpoint()
        c1._queue.close()

        summary = json.loads((tmp_path / CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
        assert summary["visited"] == 2

        # A page scraped after that write, then a crash before the next one.
        q = _CrawlQueue(tmp_path / "crawl_queue.db")
        q.mark_visited("https://example.com/late", success=True)
        q.close()

        c2 = Crawler("https://example.com", max_pages=10, output_dir=str(tmp_path), resume=True)
        assert c2._state.scraped == 3, "the database, not the stale file, prices the budget"
        c2._queue.close()

    def test_damaged_summary_is_reported_and_the_crawl_still_starts(self, tmp_path):
        """A corrupt summary used to be swallowed, leaving no clue why --resume
        appeared to do nothing."""
        (tmp_path / CHECKPOINT_FILENAME).write_text("{ not json", encoding="utf-8")
        printed = []

        with patch("protor.crawler.console") as fake_console:
            fake_console.print.side_effect = lambda *a, **k: printed.append(" ".join(map(str, a)))
            c = Crawler("https://example.com", output_dir=str(tmp_path), resume=True)

        assert any("Could not resume from checkpoint" in line for line in printed)
        assert c._queue.is_queued("https://example.com")
        c._queue.close()

    @pytest.mark.asyncio
    async def test_resumed_crawl_reports_the_whole_crawl(self, tmp_path):
        """The headline number belongs to the crawl, not to the run."""
        c1 = Crawler("https://example.com", max_pages=3, output_dir=str(tmp_path))
        seed(c1._queue, 9)
        with stubbed_network([]):
            await c1._run()
        c1._queue.close()

        c2 = Crawler("https://example.com", max_pages=6, output_dir=str(tmp_path), resume=True)
        with stubbed_network([]):
            await c2._run()
        assert c2._state.scraped == 6
        assert c2._queue.success_count == 6
        c2._queue.close()

    def test_resume_says_how_much_it_resumed(self, tmp_path):
        q = _CrawlQueue(tmp_path / "crawl_queue.db")
        for i in range(3):
            q.mark_visited(f"https://example.com/p{i}", success=True)
        q.close()

        printed = []
        with patch("protor.crawler.console") as fake_console:
            fake_console.print.side_effect = lambda *a, **k: printed.append(" ".join(map(str, a)))
            c = Crawler("https://example.com", max_pages=6, output_dir=str(tmp_path), resume=True)
        assert any("Resumed from checkpoint — 3 pages already scraped" in line for line in printed)
        c._queue.close()

    def test_a_fresh_directory_resumes_silently(self, tmp_path):
        """Nothing scraped yet: no "resumed" claim to make."""
        printed = []
        with patch("protor.crawler.console") as fake_console:
            fake_console.print.side_effect = lambda *a, **k: printed.append(" ".join(map(str, a)))
            c = Crawler("https://example.com", output_dir=str(tmp_path), resume=True)
        assert not any("Resumed" in line for line in printed)
        c._queue.close()


class TestStatusEvents:
    def test_domain_comes_from_the_row_the_engine_already_parsed(self, tmp_path):
        """The engine parses every URL it dispatches; re-parsing it for each
        status event bought nothing."""
        c = Crawler("https://example.com", output_dir=str(tmp_path))
        row = {"domain": "example.com"}
        with patch("protor.crawler.urlparse", side_effect=AssertionError("re-parsed the URL")):
            c._on_status("fetching", "https://example.com/page", row)
            c._on_status("done", "https://example.com/page", row)
        assert [entry.domain for entry in c._state.log] == ["example.com"]
        assert c._state.scraped == 1
        c._queue.close()

    def test_falls_back_to_parsing_a_row_without_a_domain(self, tmp_path):
        c = Crawler("https://example.com", output_dir=str(tmp_path))
        c._on_status("fetching", "https://example.com/page", {})
        assert c._state.log[-1].domain == "example.com"
        c._queue.close()

    @pytest.mark.asyncio
    async def test_log_rows_carry_the_dispatched_domain(self, tmp_path):
        c = Crawler("https://example.com", max_pages=2, output_dir=str(tmp_path))
        seed(c._queue, 1)
        with stubbed_network([]):
            await c._run()
        assert {entry.domain for entry in c._state.log} == {"example.com"}
        c._queue.close()


class TestCrawlerCrawl:
    @pytest.mark.asyncio
    async def test_crawl_single_page_success(self, tmp_path):
        c = Crawler("https://example.com", max_pages=1, output_dir=str(tmp_path))

        with stubbed_network([]):
            await c._run()

            assert c._state.scraped == 1
            assert c._state.errors == 0
        c._queue.close()

    @pytest.mark.asyncio
    async def test_crawl_handles_errors(self, tmp_path):
        c = Crawler("https://example.com", max_pages=1, output_dir=str(tmp_path))

        with (
            patch("protor.engine.aiohttp.ClientSession"),
            patch("protor.engine.fetch", new_callable=AsyncMock) as mock_fetch,
            NO_LIVE,
            ROBOTS_PATCH,
            NO_DELAY,
        ):
            mock_fetch.side_effect = Exception("Connection refused")

            await c._run()

            assert c._state.scraped == 0
            assert c._state.errors == 1
            assert c._state.log[-1].status == "err"
            assert "Connection refused" in c._state.log[-1].note
        c._queue.close()

    @pytest.mark.asyncio
    async def test_crawl_respects_max_pages(self, tmp_path):
        c = Crawler("https://example.com", max_pages=2, output_dir=str(tmp_path))
        # Pre-enqueue extra URLs
        seed(c._queue, 3)

        with stubbed_network([]):
            await c._run()

            assert c._state.scraped == 2
        c._queue.close()

    @pytest.mark.asyncio
    async def test_crawl_discovers_links(self, tmp_path):
        c = Crawler("https://example.com", max_pages=3, output_dir=str(tmp_path))

        with (
            patch("protor.engine.aiohttp.ClientSession"),
            patch("protor.engine.fetch", new_callable=AsyncMock) as mock_fetch,
            NO_LIVE,
            ROBOTS_PATCH,
            NO_DELAY,
        ):
            mock_fetch.return_value = FetchResult(
                text=('<html><body><a href="/about">A</a><a href="/contact">B</a></body></html>'),
                nbytes=200,
            )

            await c._run()

            assert c._queue.is_queued("https://example.com/about") or c._queue.is_visited(
                "https://example.com/about"
            )
            assert c._queue.is_queued("https://example.com/contact") or c._queue.is_visited(
                "https://example.com/contact"
            )
        c._queue.close()

    @pytest.mark.asyncio
    async def test_crawl_skips_visited(self, tmp_path):
        c = Crawler("https://example.com", max_pages=5, output_dir=str(tmp_path))

        with stubbed_network([]):
            await c._run()

            # Only 1 page should be scraped despite duplicate enqueue attempts
            assert c._state.scraped == 1
        c._queue.close()

    def test_crawl_method_writes_a_summary(self, tmp_path):
        c = Crawler("https://example.com", max_pages=1, output_dir=str(tmp_path))

        with patch("protor.crawler.asyncio.run") as mock_run:
            c.crawl()
            assert mock_run.called
            # Close the coroutine the mock discarded, so it does not resurface
            # later as a "coroutine was never awaited" RuntimeWarning.
            mock_run.call_args.args[0].close()
        assert (tmp_path / CHECKPOINT_FILENAME).exists()
        c._queue.close()


class TestRequeueOnlyAttempts:
    """
    `requeue_failed` retries what was asked for and failed — nothing else.

    A URL the crawl filtered out (robots.txt, the ad list, the domain filter) was
    never requested, so a retry is refused identically. Re-queueing it puts the
    whole filtered set at the head of every resumed run: dispatched, skipped,
    re-skipped, while the budget it should have fetched with goes unspent.
    """

    def test_a_filtered_url_is_not_requeued(self, tmp_path):
        from protor.crawler import _CrawlQueue

        q = _CrawlQueue(tmp_path / "q.db")
        q.mark_visited("https://example.com/scraped", success=True)
        q.mark_visited("https://example.com/failed", success=False)  # attempted
        q.mark_visited("https://example.com/blocked", success=False, attempted=False)
        q.close()

        again = _CrawlQueue(tmp_path / "q.db")
        assert again.requeue_failed() == 1, "only the attempted failure is retried"
        assert again.dequeue() == "https://example.com/failed"
        assert again.dequeue() is None, "the filtered URL was re-queued"
        again.close()

    def test_a_filtered_url_is_still_deduplicated(self, tmp_path):
        """
        Recording it as "not attempted" must not make it re-discoverable.

        The row is still there and still recent, so ``_SEEN_SQL`` refuses the URL
        for the rest of the run. If it were admitted again, one off-domain link
        in a nav footer would be re-fetched by every page that links to it.
        """
        from protor.crawler import _CrawlQueue

        q = _CrawlQueue(tmp_path / "q.db")
        q.mark_visited("https://example.com/blocked", success=False, attempted=False)
        assert q.enqueue("https://example.com/blocked") is False
        q.close()

    def test_only_success_counts_as_scraped(self, tmp_path):
        """The sentinel must not be mistaken for a page that worked."""
        from protor.crawler import _CrawlQueue

        q = _CrawlQueue(tmp_path / "q.db")
        q.mark_visited("https://example.com/ok", success=True)
        q.mark_visited("https://example.com/failed", success=False)
        q.mark_visited("https://example.com/blocked", success=False, attempted=False)
        assert q.success_count == 1
        assert q.visited_count == 3, "all three are recorded"
        q.close()
