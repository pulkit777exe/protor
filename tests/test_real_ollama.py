"""
protor against a real Ollama runtime.

Everything else in the suite either stubs the HTTP layer or talks to a server the
test itself started. Both are useful, and neither can tell you whether the real
thing works: a stub decides where SSE lines begin, and a hand-written server
reproduces the wire format *as understood by whoever wrote it*. A real runtime
is the only thing that can be wrong in a way nobody modelled.

These tests are skipped unless an Ollama is reachable, and additionally require a
model, because a runtime with nothing loaded answers every request with an error
and a test that asserts on that error is not testing anything:

    ollama serve
    ollama pull qwen2.5:0.5b     # any small chat model will do
    pytest tests/test_real_ollama.py

Marked ``ollama`` so the default run leaves them out. They need no network
beyond loopback once the model is present.
"""

from __future__ import annotations

import json
import socket
from typing import ClassVar

import pytest

from protor.config import OLLAMA_BASE

pytestmark = pytest.mark.ollama


def _ollama_reachable(base: str, timeout: float = 1.0) -> bool:
    """True when something is listening on *base*'s host and port."""
    from urllib.parse import urlparse

    parsed = urlparse(base)
    host, port = parsed.hostname or "localhost", parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _first_model(base: str) -> str | None:
    import requests

    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception:
        return None
    models = resp.json().get("models") or []
    return str(models[0]["name"]) if models else None


BASE = OLLAMA_BASE.rstrip("/")
MODEL = _first_model(BASE) if _ollama_reachable(BASE) else None

pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(MODEL is None, reason="no Ollama runtime with a model loaded"),
]


@pytest.fixture(scope="module")
def model() -> str:
    assert MODEL is not None, "unreachable: the skipif above should have caught this"
    return MODEL


class TestRealModelListing:
    def test_models_lists_what_is_actually_loaded(self, model):
        """The size and timestamp come from the runtime, not from a fixture."""
        from protor.llm_backends import OllamaBackend

        backend = OllamaBackend(model=model, base_url=BASE)
        assert backend.check_available() is True

        models = backend.list_models()
        assert models, "a loaded model was not listed"
        assert model in [m.name for m in models]
        assert all(m.size_bytes for m in models), "the runtime reported no size"


class TestRealStreaming:
    def test_a_completion_streams_text(self, model):
        from protor.llm_backends import OllamaBackend

        backend = OllamaBackend(model=model, base_url=BASE)
        chunks = list(backend.stream("Reply with exactly the word: acknowledged"))
        text = "".join(chunks)

        assert text.strip(), "the real runtime streamed nothing"
        assert len(chunks) > 1, f"expected a stream of chunks, got {len(chunks)}"

    def test_a_prompt_is_actually_answered(self, model):
        """
        The weakest possible check that the round trip is real: ask a question
        whose answer is checkable, and confirm the response reflects the prompt.
        """
        from protor.llm_backends import OllamaBackend

        backend = OllamaBackend(model=model, base_url=BASE)
        text = "".join(backend.stream("What is 2 plus 2? Reply with just the number."))
        assert "4" in text, f"the model did not answer the prompt: {text[:120]!r}"

    def test_an_unloaded_model_is_reported_as_such(self, model):
        """
        The typed error must survive a real 404, which is what the runtime
        actually returns for a model it has not pulled.
        """
        from protor.exceptions import OllamaModelNotFoundError
        from protor.llm_backends import OllamaBackend

        backend = OllamaBackend(model="no-such-model-9d2f", base_url=BASE)
        with pytest.raises(OllamaModelNotFoundError) as excinfo:
            list(backend.stream("hi"))
        assert "no-such-model-9d2f" in str(excinfo.value)
        assert "ollama pull" in str(excinfo.value), "the message must say how to fix it"


class TestRealAnalyze:
    SITES: ClassVar[list[dict]] = [
        {
            "url": "https://alpha.example/",
            "domain": "alpha.example",
            "html_file": "index.html",
            "metadata": {"title": "Alpha", "description": "Sells widgets"},
            "text_content": (
                "Alpha sells blue widgets for 10 dollars with next-day delivery. "
                "The widget is made of recycled aluminium."
            ),
            "js_files": [],
            "js_count": 0,
            "success": True,
        },
        {
            "url": "https://beta.example/",
            "domain": "beta.example",
            "html_file": "index.html",
            "metadata": {"title": "Beta", "description": "Sells gadgets"},
            "text_content": (
                "Beta sells red gadgets for 20 dollars with slow delivery. "
                "The gadget is made of imported plastic."
            ),
            "js_files": [],
            "js_count": 0,
            "success": True,
        },
    ]

    def test_analyze_produces_a_report_from_real_output(self, tmp_path, model):
        """The whole path, with nothing stubbed."""
        from protor.analyzer import analyze

        result = analyze(
            self.SITES,
            model=model,
            focus="general",
            output_dir=tmp_path,
            backend="ollama",
            base_url=BASE,
        )

        assert result.sites_analyzed == 2
        assert result.analysis.strip(), "the report is empty"

        saved = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
        assert saved["analysis"].strip()
        assert (tmp_path / "analysis.md").read_text(encoding="utf-8").strip()

    def test_the_report_reflects_the_data_it_was_given(self, tmp_path, model):
        """
        A tiny model will produce a poor report; what matters is that it is about
        *these* sites. A report that ignores the prompt means the context never
        reached the model — the failure mode that makes a scraper's output
        worthless while looking entirely normal.
        """
        from protor.analyzer import analyze

        result = analyze(
            self.SITES,
            model=model,
            focus="general",
            output_dir=tmp_path,
            backend="ollama",
            base_url=BASE,
        )
        answer = result.analysis.lower()
        assert any(term in answer for term in ("alpha", "beta", "widget", "gadget", "aluminium")), (
            f"the report mentions neither site: {result.analysis[:200]!r}"
        )

    def test_a_single_site_batch_works(self, tmp_path, model):
        from protor.analyzer import analyze

        result = analyze(
            self.SITES[:1],
            model=model,
            focus="content",
            output_dir=tmp_path,
            backend="ollama",
            base_url=BASE,
        )
        assert result.sites_analyzed == 1
        assert result.analysis.strip()


class TestRealCli:
    def test_the_models_command_works_against_the_runtime(self, capsys):
        """`protor models` is the first thing anyone runs. Drive it for real."""
        import sys

        from protor.cli import cli

        old = sys.argv
        sys.argv = ["protor", "models", "--backend", "ollama"]
        try:
            cli()
        finally:
            sys.argv = old

        out = capsys.readouterr().out
        assert "Model" in out
        assert "GB" in out or "MB" in out, "sizes were not rendered"

    def test_the_run_command_scrapes_then_analyzes(self, tmp_path):
        """`protor run` is the recommended path: one command, end to end."""
        import sys
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        page = (
            b"<!DOCTYPE html><html><head><title>Alpha</title>"
            b'<meta name="description" content="Sells widgets"></head>'
            b"<body><h1>Alpha</h1><p>Sells blue widgets for 10 dollars, "
            b"made of recycled aluminium, delivered next day.</p></body></html>"
        )

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

        server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        server.daemon_threads = True
        server.handle_error = lambda *_: None
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]

        try:
            from protor.cli import cli

            old = sys.argv
            sys.argv = [
                "protor",
                "run",
                f"http://127.0.0.1:{port}/",
                "--backend",
                "ollama",
                "--model",
                MODEL or "",
                "--output",
                str(tmp_path),
                "--no-js",
            ]
            try:
                cli()
            finally:
                sys.argv = old
        finally:
            server.shutdown()
            server.server_close()

        index = tmp_path / "sites_index.json"
        assert index.exists(), "the scrape produced no index"
        sites = json.loads(index.read_text(encoding="utf-8"))
        assert sites and sites[0]["success"] is True

        # `run` keeps the scrape and the analysis side by side under one output
        # directory: the index at the top, the report in `analysis/`.
        report = tmp_path / "analysis" / "analysis.json"
        assert report.exists(), f"the analysis produced no report under {tmp_path}"
        assert json.loads(report.read_text(encoding="utf-8"))["analysis"].strip()
