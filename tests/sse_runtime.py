"""
A real OpenAI-compatible HTTP server, for exercising the LLM path over a socket.

Every other test of the streaming code hands it a fake response object. That
cannot show whether the framing survives a real connection: chunk boundaries,
CRLF line endings, keepalive comments arriving mid-body, a stream that ends
without ``[DONE]``. Those are transport-level facts, and a stub has none of
them — the stub decides where lines begin, which is exactly the thing in
question.

The wire behaviour here is modelled on llama.cpp's server rather than on the
OpenAI spec, because the spec is not what these clients receive:

- ``Content-Type: text/event-stream`` with ``data:`` framing
- CRLF line endings, as a real HTTP server emits
- chunks flushed one at a time, so the client sees partial reads
- ``:`` comment lines as keepalives, which runtimes emit during long generations
- a ``data: null`` keepalive frame, which several gateways send
- ``reasoning_content`` alongside ``content`` on reasoning models
- a ``[DONE]`` sentinel — optionally omitted, to model a dropped connection

The reply is assembled from a template so a test can assert on exactly what the
model "said" without parsing prose.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["FakeRuntime", "RuntimeSpec", "runtime_server"]


@dataclass
class RuntimeSpec:
    """How the fake runtime should behave for one test."""

    #: The assistant's reply, streamed word by word.
    reply: str = "ANALYSIS: the site looks healthy."
    #: Emit ``:`` comment keepalives between chunks.
    keepalives: bool = True
    #: Emit ``data: null`` frames, which runtimes send during long generations.
    null_keepalives: bool = True
    #: Include ``reasoning_content`` alongside ``content``.
    reasoning: bool = False
    #: Close the stream without the ``[DONE]`` sentinel.
    omit_done: bool = False
    #: Send CRLF line endings, as a real server does.
    crlf: bool = True
    #: Send ``content`` as an array of fragments instead of a string.
    fragmented_content: bool = False
    #: Model ids ``/v1/models`` reports.
    models: tuple[str, ...] = ("test-model",)
    #: When set, ``/v1/chat/completions`` answers with this status instead.
    chat_status: int | None = None
    #: When set, ``/v1/models`` answers with this status instead.
    models_status: int | None = None
    #: Every request body received, for asserting on what was actually sent.
    received: list[dict] = field(default_factory=list)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Silence the default stderr logging; a test that prints a traceback per
    # request is unreadable.
    def log_message(self, fmt: str, *args: object) -> None:
        pass

    @property
    def spec(self) -> RuntimeSpec:
        return self.server.spec  # type: ignore[attr-defined]

    def _endings(self) -> str:
        return "\r\n" if self.spec.crlf else "\n"

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            if self.spec.models_status is not None:
                self._json(self.spec.models_status, {"error": "unavailable"})
                return
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": name, "object": "model", "created": 1_700_000_000}
                        for name in self.spec.models
                    ],
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            self.spec.received.append(json.loads(raw.decode("utf-8")))
        except json.JSONDecodeError:
            self.spec.received.append({"_unparsed": raw.decode("utf-8", "replace")})

        if self.spec.chat_status is not None:
            self._json(self.spec.chat_status, {"error": "rejected"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        for line in self._frames():
            self._write_chunk(line)
        if not self.spec.omit_done:
            self._write_chunk(f"data: [DONE]{self._endings()}")
        self._write_chunk("")  # terminating chunk

    def _frames(self) -> list[str]:
        """Build the SSE frames for one response."""
        eol = self._endings()
        frames: list[str] = []
        words = self.spec.reply.split(" ")
        for index, word in enumerate(words):
            if self.spec.keepalives and index:
                frames.append(f": keepalive{eol}{eol}")
            if self.spec.null_keepalives and index == 1:
                # Runtimes send this while a long generation is in progress.
                frames.append(f"data: null{eol}{eol}")
            frame: dict = {"choices": [{"index": 0, "delta": {}, "finish_reason": None}]}
            if self.spec.reasoning:
                frame["choices"][0]["delta"]["reasoning_content"] = f"thinking {index}"
            # A real stream splits mid-sentence, so the client must reassemble
            # the reply exactly. Each piece but the last carries the separator
            # that followed it in the original text.
            piece = word + (" " if index < len(words) - 1 else "")
            if self.spec.fragmented_content:
                frame["choices"][0]["delta"]["content"] = [{"type": "text", "text": piece}]
            else:
                frame["choices"][0]["delta"]["content"] = piece
            frames.append(f"data: {json.dumps(frame)}{eol}{eol}")
        return frames

    def _write_chunk(self, text: str) -> None:
        """Write one HTTP chunk, so the client sees genuinely partial reads."""
        payload = text.encode("utf-8")
        self.wfile.write(f"{len(payload):X}\r\n".encode())
        self.wfile.write(payload)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeRuntime:
    """A running fake runtime. Use as a context manager."""

    def __init__(self, spec: RuntimeSpec | None = None) -> None:
        self.spec = spec or RuntimeSpec()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.spec = self.spec  # type: ignore[attr-defined]
        # A client that stops reading mid-stream is expected here — the point is
        # to exercise partial reads — so don't print a traceback for each one.
        self._server.handle_error = lambda *_a: None  # type: ignore[method-assign]
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeRuntime:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def runtime_server(spec: RuntimeSpec | None = None) -> FakeRuntime:
    """Build a fake runtime; call it as a context manager to run it."""
    return FakeRuntime(spec)
