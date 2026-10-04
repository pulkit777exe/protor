"""
Downloaded scripts must not overwrite each other across a site's pages.

Found by running the *default* pipeline against a real site rather than by
reading code. Every earlier check of this area ran with `--no-js`, which is the
flag a user reaches for when something is slow — so the whole script-download
path had never been exercised against real input.

Two pages of one site loading `/static/app.js` and `/static/app.js?v=2` left
**one** file on disk. Both manifests listed a script and both claimed success;
the first page's copy was unrecoverable. `js_dir` is per-*domain* while the set of
filenames already used was per-*page*, so the second download wrote over the
first.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from protor.engine import CrawlEngine

PAGE_A_SCRIPT = b"console.log('script from PAGE A');"
PAGE_B_SCRIPT = b"console.log('script from PAGE B, different content entirely');"

_UNSET = object()


def _page(title: str, src: str) -> bytes:
    return (
        f"<!DOCTYPE html><html><head><title>{title}</title>"
        f"<script src='{src}'></script></head><body><h1>{title}</h1></body></html>"
    ).encode()


class _Site:
    """A site whose two pages load scripts that share a basename on purpose."""

    def __init__(self) -> None:
        pages = {
            "/a": _page("Page A", "/static/app.js"),
            "/b": _page("Page B", "/static/app.js?v=2"),
        }
        scripts = {
            "/static/app.js": PAGE_A_SCRIPT,
            "/static/app.js?v=2": PAGE_B_SCRIPT,
        }

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                body = scripts.get(self.path) or pages.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                ctype = (
                    "application/javascript" if self.path.startswith("/static/") else "text/html"
                )
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._server.handle_error = lambda *_: None
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def site():
    s = _Site()
    yield s
    s.close()


class _FakeQueue:
    """The smallest thing CrawlEngine accepts where it expects a queue."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def __len__(self) -> int:
        return 0

    def pop(self):
        raise IndexError("empty")

    def put(self, url: str) -> None:
        self.urls.append(url)

    def __contains__(self, url: object) -> bool:
        return False


class _NoLinks:
    def discover(self, url: str, page: object) -> list[str]:
        return []


def _engine(tmp_path) -> CrawlEngine:
    """A CrawlEngine with only the filename bookkeeping exercised."""
    return CrawlEngine(
        queue=_FakeQueue(),
        link_source=_NoLinks(),
        output_dir=tmp_path,
        max_targets=0,
    )


class TestAcrossPages:
    @pytest.mark.integration
    def test_two_pages_loading_the_same_basename_keep_both_scripts(self, site, tmp_path):
        """The measured defect, driven over real HTTP."""
        from protor.scraper import scrape_multiple

        scrape_multiple([f"{site.base}/a", f"{site.base}/b"], output_dir=tmp_path, live=False)

        saved = sorted(tmp_path.rglob("*.js"))
        assert len(saved) == 2, f"expected both scripts, found {[p.name for p in saved]}"

        bodies = {p.read_bytes() for p in saved}
        assert PAGE_A_SCRIPT in bodies, "page A's script was overwritten"
        assert PAGE_B_SCRIPT in bodies, "page B's script is missing"

    @pytest.mark.integration
    def test_both_manifests_still_describe_a_file_that_exists(self, site, tmp_path):
        from protor.scraper import scrape_multiple

        scrape_multiple([f"{site.base}/a", f"{site.base}/b"], output_dir=tmp_path, live=False)
        records = json.loads((tmp_path / "sites_index.json").read_text(encoding="utf-8"))
        assert len(records) == 2
        for record in records:
            assert record["js_files"], "a script was downloaded but not recorded"
            assert record["success"] is True

    @pytest.mark.integration
    def test_the_two_downloads_really_are_different_resources(self, site):
        """
        The premise of the fix, checked against the server rather than assumed.

        If both URLs returned the same bytes there would be nothing to preserve,
        and a test asserting two files would be asserting an implementation
        detail rather than a requirement.
        """
        import urllib.request

        a = urllib.request.urlopen(f"{site.base}/static/app.js", timeout=5).read()
        b = urllib.request.urlopen(f"{site.base}/static/app.js?v=2", timeout=5).read()
        assert a != b


class TestSharedScripts:
    def test_the_same_url_is_stored_once_however_many_pages_reference_it(self, tmp_path):
        """
        The common case, and the one a naive fix breaks.

        Every page of a real site loads the same vendor bundle. Keying only by
        basename would store one copy per page under one hashed name each;
        keying by URL stores one and points every manifest at it.
        """
        engine = _engine(tmp_path)
        names = {
            engine._reserve_js_filename("site", 0, "https://x.com/static/vendor.js")
            for _ in range(50)
        }
        assert len(names) == 1, f"the same URL was stored under {len(names)} names"

    def test_different_urls_sharing_a_basename_get_different_files(self, tmp_path):
        engine = _engine(tmp_path)
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

    def test_separate_sites_do_not_interfere(self, tmp_path):
        """
        Two domains have two `js/` directories, so the same basename in each is
        not a collision and neither should be renamed.
        """
        engine = _engine(tmp_path)
        a = engine._reserve_js_filename("site-a", 0, "https://a.com/static/app.js")
        b = engine._reserve_js_filename("site-b", 0, "https://b.com/static/app.js")
        assert a == "app.js"
        assert b == "app.js"

    def test_reservation_happens_before_any_download(self, tmp_path):
        """
        Two pages fetched concurrently must not both be handed the same name.

        The reservation is what prevents that: if it happened after the download
        was scheduled, both pages would pick `app.js` before either had stored
        anything.
        """
        engine = _engine(tmp_path)
        first = engine._reserve_js_filename("site", 0, "https://x.com/a/app.js")
        second = engine._reserve_js_filename("site", 1, "https://x.com/b/app.js")
        assert first != second

    def test_the_original_name_is_preferred_when_it_is_free(self, tmp_path):
        """Hashing is a last resort; readable names are the common case."""
        engine = _engine(tmp_path)
        assert engine._reserve_js_filename("site", 0, "https://x.com/a/jquery.js") == "jquery.js"


class TestCrawlJsFlag:
    @staticmethod
    def _command(name: str):
        from protor.cli import _build_parser

        root = _build_parser()
        sub = next(a for a in root._actions if getattr(a, "choices", None))
        return sub.choices[name]

    def test_crawl_offers_a_js_switch(self):
        """`crawl` never downloaded scripts and `crawl --help` never said so."""
        assert any(a.dest == "js" for a in self._command("crawl")._actions)

    def test_the_js_help_names_the_default_and_the_reason(self):
        action = next(a for a in self._command("crawl")._actions if a.dest == "js")
        assert action.default is False, "crawl should not start downloading scripts"
        assert "off by default" in action.help, action.help

    def test_scrape_still_downloads_scripts_by_default(self):
        no_js = next(a for a in self._command("scrape")._actions if a.dest == "no_js")
        assert no_js.default is False, "scrape's existing default must not change"

    def test_crawl_passes_the_flag_through_to_the_engine(self, tmp_path, monkeypatch):
        # `cli` imports Crawler into its own namespace, so patching the module
        # it was defined in would leave the real class in place — which is what
        # happened the first time this test was written.
        import protor.cli as cli_mod

        seen: dict[str, bool] = {}

        class Spy:
            def __init__(self, *a, download_js: bool = False, **kw):
                seen["download_js"] = download_js

            def crawl(self):
                return "done"

        monkeypatch.setattr(cli_mod, "Crawler", Spy)
        from protor.cli import _cmd_crawl

        _cmd_crawl(argparse_ns(js=True, output=str(tmp_path)))
        assert seen["download_js"] is True

        _cmd_crawl(argparse_ns(js=False, output=str(tmp_path)))
        assert seen["download_js"] is False


def argparse_ns(**kwargs):
    """Every attribute `_cmd_crawl` reads, so a missing one fails loudly."""
    import argparse

    base = {
        "url": "https://x.com/",
        "output": None,
        "max_pages": 1,
        "resume": False,
        "auto_scale": False,
        "no_live": True,
        "allow_internal_redirects": False,
        "js": False,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)
