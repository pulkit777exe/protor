"""
Redirect targets a scraper must refuse to follow.

A fetcher that follows redirects unconditionally will fetch whatever a remote
server tells it to. These cover the decision itself (:mod:`protor.netguard`)
and, over real sockets, the behaviour it exists to produce.

Loopback and RFC 1918 are deliberately *allowed*: protor scrapes local and
intranet sites, and a user who types an internal URL has already decided that.
What is refused is never a legitimate scrape destination.
"""

from __future__ import annotations

import http.server
import threading

import pytest

from protor.netguard import describe_block, is_blocked_redirect, is_metadata_host


class TestRedirectTargets:
    @pytest.mark.parametrize(
        "url,reason",
        [
            ("http://169.254.169.254/latest/meta-data/", "metadata"),
            ("http://[fe80::1]/x", "link-local"),
            ("http://169.254.170.2/computeMetadata/v1/", "metadata"),
            ("http://metadata.google.internal/computeMetadata/v1/", "metadata"),
            ("http://metadata/computeMetadata/v1/", "metadata"),
            ("file:///etc/passwd", "scheme"),
            ("ftp://example.com/x", "scheme"),
            ("gopher://example.com/", "scheme"),
            ("data:text/html,<h1>x", "scheme"),
        ],
    )
    def test_blocked_targets(self, url, reason):
        blocked = describe_block(url)
        assert blocked is not None, f"{url} should be refused"
        assert reason in blocked, f"unhelpful reason for {url}: {blocked!r}"

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/",
            "http://example.com/page",
            # protor exists to scrape local and intranet sites.
            "http://127.0.0.1:8000/",
            "http://localhost:11434/",
            "http://10.0.0.5/admin",
            "http://192.168.1.10/",
            "http://[::1]/",
            "http://wiki.internal/page",
        ],
    )
    def test_allowed_targets(self, url):
        assert describe_block(url) is None, f"{url} is legitimate and must not be blocked"

    def test_metadata_host_recognised_by_name_and_by_literal(self):
        assert is_metadata_host("metadata.google.internal")
        assert is_metadata_host("METADATA.GOOGLE.INTERNAL")  # case-insensitive
        assert is_metadata_host("metadata.google.internal.")  # trailing dot
        assert is_metadata_host("169.254.169.254")
        assert not is_metadata_host("example.com")
        assert not is_metadata_host("")

    def test_a_link_local_address_is_blocked_even_when_it_is_not_the_metadata_ip(self):
        """
        The whole 169.254.0.0/16 range is refused, not just the one well-known
        metadata address, so a cloud on a different address is covered too. The
        two checks are separate on purpose: ``is_metadata_host`` names a secret,
        the range check names a destination that is never a web page.
        """
        assert is_metadata_host("169.254.1.1") is False, "not the metadata IP itself"
        assert describe_block("http://169.254.1.1/") is not None, "but still link-local"

    def test_malformed_url_is_refused_rather_than_crashing(self):
        assert is_blocked_redirect("http://[not-an-ip]/x") in (True, False)
        assert describe_block("") is not None

    def test_a_host_that_looks_like_a_metadata_name_is_not_matched_by_suffix(self):
        """`notmetadata.google.internal` is a real registrable domain."""
        assert is_metadata_host("notmetadata.google.internal") is False


def _serve(handler_cls, *args):
    """Run *handler_cls* on loopback; returns (port, shutdown)."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    server.daemon_threads = True
    server.handle_error = lambda *_: None  # type: ignore[method-assign]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server.server_address[1], lambda: (server.shutdown(), server.server_close())


def _redirector(location: str):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

    return H


def _body(text: str):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            payload = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return H


async def _fetch(url: str, **kw):
    import aiohttp

    from protor.fetcher import fetch

    async with aiohttp.ClientSession() as session:
        return await fetch(session, url, **kw)


@pytest.mark.integration
class TestRedirectsOverRealSockets:
    """The guard is only useful if it holds on a real connection."""

    async def test_a_redirect_to_metadata_is_refused(self):
        from protor.exceptions import FetchError

        port, stop = _serve(_redirector("http://169.254.169.254/latest/meta-data/"))
        try:
            with pytest.raises(FetchError) as excinfo:
                await _fetch(f"http://127.0.0.1:{port}/totally-normal-page")
            assert "metadata" in str(excinfo.value)
        finally:
            stop()

    async def test_a_redirect_to_a_file_url_is_refused(self):
        from protor.exceptions import FetchError

        port, stop = _serve(_redirector("file:///etc/passwd"))
        try:
            with pytest.raises(FetchError) as excinfo:
                await _fetch(f"http://127.0.0.1:{port}/page")
            assert "scheme" in str(excinfo.value)
        finally:
            stop()

    async def test_a_loopback_redirect_is_still_followed(self):
        """Blocking this would break scraping a local site that redirects."""
        target, stop_target = _serve(_body("REAL PAGE"))
        port, stop = _serve(_redirector(f"http://127.0.0.1:{target}/final"))
        try:
            result = await _fetch(f"http://127.0.0.1:{port}/start")
            assert result.text == "REAL PAGE"
        finally:
            stop()
            stop_target()

    async def test_a_relative_redirect_resolves_against_the_page(self):
        """`Location: /second` must resolve against the current URL, not fail."""

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/first":
                    self.send_response(302)
                    self.send_header("Location", "/second")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = b"SECOND PAGE"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        port, stop = _serve(H)
        try:
            result = await _fetch(f"http://127.0.0.1:{port}/first")
            assert result.text == "SECOND PAGE"
        finally:
            stop()

    async def test_the_opt_in_flag_lifts_the_refusal(self):
        port, stop = _serve(_redirector("http://169.254.169.254/latest/"))
        try:
            # Nothing listens on link-local here, so this cannot succeed; what is
            # being checked is that the guard, not the network, is what refused it.
            from protor.exceptions import FetchError

            with pytest.raises(FetchError) as excinfo:
                await _fetch(f"http://127.0.0.1:{port}/page", allow_internal_redirects=True)
            assert "refused to follow" not in str(excinfo.value)
        finally:
            stop()

    async def test_a_redirect_loop_terminates(self):
        """A pair of pages that redirect to each other must not spin."""
        from protor.exceptions import FetchError

        ports: dict[str, int] = {}

        def make(name: str):
            class H(http.server.BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"

                def log_message(self, *a):
                    pass

                def do_GET(self):
                    other = "b" if name == "a" else "a"
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{ports[other]}/")
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            return H

        servers = []
        for name in ("a", "b"):
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make(name))
            server.daemon_threads = True
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
            ports[name] = server.server_address[1]

        try:
            with pytest.raises(FetchError, match="redirects"):
                await _fetch(f"http://127.0.0.1:{ports['a']}/")
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
