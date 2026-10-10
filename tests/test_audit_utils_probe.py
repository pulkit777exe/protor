"""THROWAWAY audit probes, round 7. Not kept."""

from __future__ import annotations


def test_R1c_resume_never_rewalks_a_scraped_page(tmp_path, monkeypatch):
    """
    Isolate the frontier loss from the commit window: commit everything first,
    then delete p1 from the queue the way a fetch that never finished does.
    """
    import protor.engine as engine_mod
    import protor.robots as robots_mod
    from protor.fetcher import FetchResult
    from protor.crawler import Crawler, _CrawlQueue

    seen: list[str] = []

    async def fake_fetch(session, url, **kwargs):  # noqa: ANN001, ARG001
        seen.append(url)
        # p0 is the only in-link to p1.
        body = "<html><body><a href='/p1'>p1</a></body></html>"
        return FetchResult(text=body, nbytes=len(body))

    async def fake_robots(url, session, user_agent=None):  # noqa: ANN001
        return True

    monkeypatch.setattr(engine_mod, "fetch", fake_fetch)
    monkeypatch.setattr(robots_mod, "check_robots", fake_robots)
    monkeypatch.setattr(engine_mod, "check_robots", fake_robots)

    out = tmp_path / "crawler"
    out.mkdir()

    q = _CrawlQueue(out / "crawl_queue.db")
    q.mark_visited("https://a.example/p0", success=True)
    q.enqueue("https://a.example/p1")
    q.close()  # committed

    q2 = _CrawlQueue(out / "crawl_queue.db")
    print("p1 dispatched, then the process died:", q2.dequeue())
    q2.close()  # the DELETE is committed

    b = Crawler("https://a.example/p0", 10, out, resume=True, live=False)
    print("run 2 queue size:", b._queue.queue_size)
    b.crawl()
    print("run 2 fetched   :", seen)
    print("p1 recovered?   :", "https://a.example/p1" in seen)

    # And a second --resume does not help either.
    c = Crawler("https://a.example/p0", 10, out, resume=True, live=False)
    c.crawl()
    print("run 3 fetched   :", seen, " still missing p1:", "https://a.example/p1" not in seen)


def test_R4_enqueue_refuses_a_scraped_page(tmp_path):
    from protor.crawler import _CrawlQueue

    q = _CrawlQueue(tmp_path / "crawl_queue.db")
    q.mark_visited("https://a.example/p0", success=True)
    print("re-enqueue a scraped page:", q.enqueue("https://a.example/p0"))
    print("queue size              :", q.queue_size)
    q.close()