"""
The `protor extract` command, end to end.

`extract` is a whole user-facing command that no test drove: it validates a URL,
loads and compiles a schema, fetches a page, applies the selectors, writes a JSON
file and previews the records. A regression anywhere in that chain surfaced only
when a user ran it against a real site.

These drive the real command over a real page served from loopback. The schema is
compiled for real, so a selector typo is rejected the way it would be in
production rather than being stubbed away.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

PAGE = """<!DOCTYPE html>
<html><head><title>Widgets Ltd</title></head>
<body>
  <main>
    <section class="product" data-sku="A1">
      <h2 class="name">Blue Widget</h2>
      <span class="price">$10</span>
      <a class="link" href="/p/blue">Details</a>
    </section>
    <section class="product" data-sku="B2">
      <h2 class="name">Red Widget</h2>
      <span class="price">$20</span>
      <a class="link" href="/p/red">Details</a>
    </section>
  </main>
</body></html>
"""

#: ``base_selector`` defines what one *record* is: every field selector is
#: resolved inside each matching container, which is how two products on a page
#: become two records rather than one.
SCHEMA = {
    "name": "products",
    "base_selector": "section.product",
    "fields": [
        {"name": "name", "selector": ".name", "type": "text"},
        {"name": "price", "selector": ".price", "type": "text"},
        # `href` resolves against the page URL; `attribute` returns the raw value.
        {"name": "link", "selector": "a.link", "type": "href"},
        {"name": "raw_href", "selector": "a.link", "type": "attribute", "attribute": "href"},
    ],
}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def site():
    """Serve PAGE on loopback for the duration of one test."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}/index.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _write_schema(tmp_path, payload: dict, name: str = "schema.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _run(argv: list[str]) -> None:
    """Invoke the CLI, letting SystemExit propagate."""
    import sys

    from protor.cli import cli

    old = sys.argv
    sys.argv = ["protor", *argv]
    try:
        cli()
    finally:
        sys.argv = old


class TestExtractCommand:
    def test_records_are_extracted_and_written(self, site, tmp_path):
        """The whole chain: schema compiled, page fetched, records written."""
        schema = _write_schema(tmp_path, SCHEMA)
        out = tmp_path / "out"

        _run(["extract", site, schema, "--output", str(out)])

        written = out / "products_index.json"
        assert written.exists(), "no output file was written"

        records = json.loads(written.read_text(encoding="utf-8"))
        assert len(records) == 2, "base_selector should yield one record per product"
        assert records[0]["name"] == "Blue Widget"
        assert records[0]["price"] == "$10"
        assert records[1]["name"] == "Red Widget"
        assert records[1]["link"].endswith("/p/red")

    def test_href_fields_resolve_against_the_page_url(self, site, tmp_path):
        """
        A relative href saved verbatim is unusable in the output file: nothing
        downstream of the scraper knows which page it came from.
        """
        schema = _write_schema(tmp_path, SCHEMA)
        out = tmp_path / "out"
        _run(["extract", site, schema, "--output", str(out)])

        records = json.loads((out / "products_index.json").read_text(encoding="utf-8"))
        assert records[0]["link"] == site.rsplit("/", 1)[0] + "/p/blue"
        # `attribute` is the escape hatch for the raw value, and stays raw.
        assert records[0]["raw_href"] == "/p/blue"

    def test_a_schema_whose_selectors_match_nothing_says_so(self, site, tmp_path):
        """
        No matches used to be indistinguishable from success in the saved file.
        The command must say nothing matched rather than write an empty result.
        """
        schema = _write_schema(
            tmp_path,
            {
                "name": "none",
                "base_selector": "section.product",
                "fields": [{"name": "x", "selector": ".no-such-class"}],
            },
        )
        out = tmp_path / "out"
        # Non-zero, so `protor extract ... && next-step` does not run on a schema
        # that matched nothing. The message is unchanged; only the code is.
        with pytest.raises(SystemExit) as excinfo:
            _run(["extract", site, schema, "--output", str(out)])

        assert excinfo.value.code == 1, "an empty extraction reported success"
        assert not (out / "none_index.json").exists(), "wrote a file for no matches"

    def test_a_missing_schema_file_is_reported(self, site, tmp_path):

        with pytest.raises(SystemExit):
            _run(["extract", site, str(tmp_path / "absent.json")])

    def test_a_malformed_schema_is_rejected_before_any_fetch(self, site, tmp_path):
        """
        A schema with a broken selector used to be applied field by field, so
        every field came back None and the page still reported as scraped.
        """
        schema = _write_schema(tmp_path, "{not json")
        with pytest.raises(SystemExit):
            _run(["extract", site, schema])

    def test_an_invalid_selector_is_rejected_before_fetching(self, site, tmp_path, capsys):
        """
        A selector that does not parse is a schema bug, found when the schema
        loads, naming the field. Applied field by field instead, every value
        came back None and the page still reported as scraped.
        """
        schema = _write_schema(
            tmp_path,
            {
                "name": "bad",
                "base_selector": "section.product",
                "fields": [{"name": "broken", "selector": "[[[bad"}],
            },
        )
        with pytest.raises(SystemExit) as excinfo:
            _run(["extract", site, schema])
        assert excinfo.value.code == 1
        assert "broken" in capsys.readouterr().out

    def test_a_schema_matching_containers_but_no_fields_writes_nothing(self, site, tmp_path):
        """
        The failure-as-success case one level above selector syntax: the
        container matched, the field selectors did not, and the result was a
        file of all-null records reported as a successful extraction. That is
        what a stale CSS selector looks like after a site redesigns its markup.
        """
        schema = _write_schema(
            tmp_path,
            {
                "name": "stale",
                "base_selector": "section.product",
                "fields": [{"name": "x", "selector": ".renamed-away"}],
            },
        )
        out = tmp_path / "out"
        with pytest.raises(SystemExit) as excinfo:
            _run(["extract", site, schema, "--output", str(out)])

        assert excinfo.value.code == 1, "all-null records reported success"
        assert not (out / "stale_index.json").exists(), "wrote records whose every value was None"

    def test_a_url_without_a_scheme_is_rejected(self, tmp_path, capsys):
        """Rejected before any fetch, with a message naming what was wrong."""
        schema = _write_schema(tmp_path, SCHEMA)
        with pytest.raises(SystemExit) as excinfo:
            _run(["extract", "example.com/no-scheme", schema])
        assert excinfo.value.code == 1
        assert "scheme" in capsys.readouterr().out
