"""
Regressions for a six-defect audit of the LLM and runtime layers.

Each class below covers one defect that was confirmed by hand, in the order the
audit found them. The tests drive the code the way a user reaches it — a real
``requests`` exception raised by the transport, a base URL typed with a trailing
slash, a stream that carries an error frame instead of text — rather than
asserting on implementation details, because the defects were all "the wrong
thing reaches the user" and a test that mocks past the wrong thing cannot see
them.

The recurring theme is the module contract in ``protor.llm_backends``:
*Every user-facing failure ... is raised as a :class:`ProtorError` subclass*.
``cli.cli()`` catches ``ProtorError`` and ``ValueError`` and prints a message
plus a hint. Anything else reaches the user as a traceback, and a traceback is
not a diagnosis no matter what message was discarded on the way past it.
"""

from __future__ import annotations

import json

import pytest
import requests
import responses as responses_lib

from protor.exceptions import (
    AuthError,
    ProtorError,
    RuntimeHTTPError,
    RuntimeUnavailableError,
)

OLLAMA_URL = "http://localhost:11434"
LLAMA_URL = "http://localhost:8080"


def _connection_error(message: str = "connection refused") -> requests.exceptions.ConnectionError:
    """A genuine ``requests`` exception, for ``responses`` to raise in place of a reply."""
    return requests.exceptions.ConnectionError(message)


def _raise_connection_error(*_args: object, **_kwargs: object) -> object:
    """Stand in for ``requests.post`` when the runtime is already gone."""
    raise _connection_error()


def _always_ok_get(*_args: object, **_kwargs: object) -> requests.Response:
    """A 200 for ``check_available``, so the code under test is actually reached."""
    resp = requests.Response()
    resp.status_code = 200
    resp._content = b'{"models":[]}'
    return resp


def _refuse_after(ok_calls: int = 1):
    """
    Build a ``requests.get`` stand-in that answers *ok_calls* times, then refuses.

    A function with a persistent attribute would leak that count between tests,
    so the state lives in the closure the factory creates.
    """
    seen = 0

    def _get(*_args: object, **_kwargs: object) -> requests.Response:
        nonlocal seen
        if seen >= ok_calls:
            raise _connection_error()
        seen += 1
        return _always_ok_get()

    return _get


# ── 1. requests exceptions escaping stream() ──────────────────────────────────


class TestStreamTransportFailuresAreTyped:
    """
    A runtime that dies between ``check_available()`` and the POST reached the
    user as a traceback.

    ``OllamaBackend.stream``, ``OpenAIBackend.stream`` and
    ``AnthropicBackend.stream`` called ``requests.post(...)`` bare. A refused
    connection raises ``requests.exceptions.ConnectionError``, which is an
    ``OSError`` — neither a ``ProtorError`` nor a ``ValueError`` — so
    ``cli.cli()``'s ``except`` chain missed it entirely. The message built for
    this exact case in ``OpenAICompatBackend.stream`` (same file, same call
    shape) never got a chance to print.

    The three are pinned together: two of them already did the right thing, so
    the fix is consistency rather than a new idea, and the test is consistency
    rather than three unrelated assertions.
    """

    @responses_lib.activate
    @pytest.mark.parametrize(
        ("backend", "kwargs", "expected_runtime"),
        [
            ("ollama", {}, "Ollama"),
            ("openai", {"api_key": "k"}, "OpenAI"),
            ("anthropic", {"api_key": "k"}, "Anthropic"),
        ],
    )
    def test_a_refused_connection_is_reported_not_raised_raw(
        self, backend, kwargs, expected_runtime
    ):
        from protor.llm_backends import create_backend as make

        # The POST is the first thing stream() does, so a single stub is enough.
        responses_lib.add(responses_lib.POST, _stream_url(backend), body=_connection_error())

        with pytest.raises(RuntimeUnavailableError) as exc:
            list(make(backend, "m", **kwargs).stream("hi"))

        assert exc.value.runtime == expected_runtime
        assert isinstance(exc.value, ProtorError)
        # The transport error is the cause, not swallowed: a caller can still
        # inspect it, and `raise ... from exc` is what keeps it in the traceback
        # if one is ever printed.
        assert isinstance(exc.value.__cause__, requests.exceptions.ConnectionError)

    @responses_lib.activate
    def test_an_unreachable_ollama_carries_its_url_and_start_command(self):
        """
        The typed error is only useful if it says where and how to fix it.

        ``RuntimeUnavailableError`` exists to carry those two fields; a bare
        "connection error" would be a typed exception that still tells the user
        nothing.
        """
        from protor.llm_backends import OllamaBackend

        responses_lib.add(
            responses_lib.POST, f"{OLLAMA_URL}/api/generate", body=_connection_error()
        )

        with pytest.raises(RuntimeUnavailableError) as exc:
            list(OllamaBackend("m", base_url=OLLAMA_URL).stream("hi"))
        assert exc.value.base_url == OLLAMA_URL
        assert exc.value.start_hint == "ollama serve"

    @responses_lib.activate
    def test_the_cli_renders_it_without_a_traceback(self, monkeypatch, tmp_path, capsys):
        """
        The end-to-end claim: ``protor analyze`` prints advice and exits 1.

        The other tests assert on an exception object. This one asserts on what
        the user actually sees, because the defect was never "the wrong type was
        raised" — it was that a traceback appeared where a sentence should have.
        """
        from protor.cli import cli

        index = tmp_path / "sites_index.json"
        index.write_text(
            json.dumps(
                [
                    {
                        "url": "https://example.com/",
                        "domain": "example.com",
                        "html_file": "example.com.html",
                        "js_count": 0,
                        "metadata": {"title": "Example"},
                        "text_content": "Real content worth analysing.",
                        "js_files": [],
                        "status": 200,
                    }
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "sys.argv",
            ["protor", "analyze", "--file", str(index), "--output", str(tmp_path / "out")],
        )
        # check_available() passes, then the runtime dies before the POST: the
        # window the wrapping exists for.
        monkeypatch.setattr("requests.get", _always_ok_get)
        monkeypatch.setattr("requests.post", _raise_connection_error)

        with pytest.raises(SystemExit) as exc:
            cli()

        err = capsys.readouterr().err
        assert exc.value.code == 1
        assert "Traceback" not in err, err
        assert "Cannot reach Ollama" in err, err


# ── 2. list_models wrapping connection errors on only one backend ─────────────


class TestListModelsTransportFailuresAreTyped:
    """
    The identical failure had three different behaviours on four backends.

    ``OpenAICompatBackend.list_models`` raised ``RuntimeUnavailableError``.
    ``OllamaBackend``, ``OpenAIBackend`` and ``AnthropicBackend`` called
    ``requests.get`` bare, so the same refused connection escaped as an
    ``OSError``. ``analyzer.list_runtime_models`` catches ``Exception``,
    prints "Could not list models" and **returns normally** — a failed model
    listing reported as a successful command with exit 0, which a script
    cannot distinguish from "the runtime has no models".

    All three are pinned against the one that was already right.
    """

    @responses_lib.activate
    @pytest.mark.parametrize(
        ("backend", "kwargs", "url", "expected_runtime"),
        [
            ("ollama", {}, f"{OLLAMA_URL}/api/tags", "Ollama"),
            ("openai", {"api_key": "k"}, "https://api.openai.com/v1/models", "OpenAI"),
            (
                "anthropic",
                {"api_key": "k"},
                "https://api.anthropic.com/v1/models",
                "Anthropic",
            ),
        ],
    )
    def test_a_refused_connection_is_typed(self, backend, kwargs, url, expected_runtime):
        from protor.llm_backends import create_backend as make

        responses_lib.add(responses_lib.GET, url, body=_connection_error())

        with pytest.raises(RuntimeUnavailableError) as exc:
            make(backend, "unused", **kwargs).list_models()

        assert exc.value.runtime == expected_runtime
        assert isinstance(exc.value, ProtorError)

    def test_an_unexpected_listing_failure_is_also_a_failure(self, monkeypatch):
        """
        The catch-all must not become the soft path again.

        The typed errors are what the backends now raise, so the ``except
        ProtorError: raise`` above handles every real case — which left the
        catch-all with no test of its own, and reverting it to
        ``print; return`` passed the whole suite. A backend that raises something
        else (a `requests` internal, a `KeyError` on an unexpected payload) is
        still a failed listing, and it has to exit non-zero: a green `protor
        models` against a dead runtime is indistinguishable from an empty one.
        """
        from protor.analyzer import list_runtime_models

        class _Boom:
            display_name = "Weird"
            base_url = "http://localhost:9999"
            start_hint = ""

            def check_available(self):
                return True

            def list_models(self):
                raise ValueError("something nobody planned for")

        monkeypatch.setattr("protor.analyzer.create_backend", lambda *a, **k: _Boom())

        with pytest.raises(ProtorError) as exc:
            list_runtime_models("ollama")

        assert "something nobody planned for" in str(exc.value)

    @responses_lib.activate
    def test_the_local_backend_carries_its_start_command(self):
        """
        Same remedy the analyzer offers when ``check_available`` fails, so both
        paths produce the same sentence and the same ``Start it with:`` hint.
        """
        from protor.llm_backends import OllamaBackend

        responses_lib.add(responses_lib.GET, f"{OLLAMA_URL}/api/tags", body=_connection_error())

        with pytest.raises(RuntimeUnavailableError) as exc:
            OllamaBackend("unused").list_models()
        assert exc.value.start_hint == "ollama serve"

    def test_the_models_command_says_where_and_how_instead_of_a_socket_error(
        self, monkeypatch, capsys
    ):
        """
        What ``protor models`` actually prints for a runtime that dies between
        the availability check and the listing.

        ``analyzer.list_runtime_models`` catches ``Exception`` around
        ``list_models()`` and prints ``Could not list models: {exc}``, so the text
        of the exception is the diagnosis. A bare ``OSError`` rendered as
        ``[Errno 111] Connection refused`` — no URL, no runtime name, no command
        to run. The typed error renders all three.

        It exits non-zero. The path used to be ``except Exception: print; return``, so
        ``protor models`` reported a dead runtime as a successful command — exit 0
        against a refused listing, indistinguishable from "no models", and a
        script piping this into ``&&`` could not tell them apart.
        """
        from protor.analyzer import list_runtime_models

        # check_available() answers, then the listing request is refused: the
        # only window in which list_models() is reached at all.
        monkeypatch.setattr("requests.get", _refuse_after())

        with pytest.raises(RuntimeUnavailableError) as exc:
            list_runtime_models("ollama")

        # The diagnosis is the raised error's own message — cli.cli() is what
        # renders it, and it renders it once, so asserting here rather than on
        # captured output is what pins that the text survives to the user.
        assert "Cannot reach Ollama at http://localhost:11434" in str(exc.value)
        assert "ollama serve" in str(exc.value)
        assert "Errno 111" not in str(exc.value)
        assert "Connection refused" not in str(exc.value)


# ── 3. trailing slashes on --base-url ─────────────────────────────────────────


class TestTrailingSlashOnBaseUrl:
    """
    ``--base-url http://localhost:11434/`` produced ``...//api/generate``.

    ``resolve_base_url`` strips trailing slashes precisely so path
    concatenation stays predictable, and ``_endpoint`` does it again for the
    OpenAI-compatible backends — but ``OllamaBackend`` and ``OpenAIBackend``
    stored the value verbatim and built their URLs by f-string. A trailing
    slash is a perfectly natural thing to type, and it broke every request the
    backend made while leaving ``check_available()`` False, which reads as
    "Ollama is stopped".
    """

    def test_ollama_strips_it(self):
        from protor.llm_backends import OllamaBackend

        assert OllamaBackend("m", base_url=f"{OLLAMA_URL}/").base_url == OLLAMA_URL

    def test_the_openai_hosted_backend_strips_it(self):
        """No ``base_url`` property here, so the stored value is read directly."""
        from protor.llm_backends import OpenAIBackend

        backend = OpenAIBackend("gpt-4o", api_key="k", base_url="https://api.openai.com/v1/")
        assert backend._base_url == "https://api.openai.com/v1"

    @responses_lib.activate
    def test_ollama_requests_have_no_double_slash(self):
        from protor.llm_backends import OllamaBackend

        responses_lib.add(
            responses_lib.GET, f"{OLLAMA_URL}/api/tags", json={"models": []}, status=200
        )
        responses_lib.add(
            responses_lib.POST,
            f"{OLLAMA_URL}/api/generate",
            body='{"response":"ok","done":true}',
            status=200,
        )

        backend = OllamaBackend("m", base_url=f"{OLLAMA_URL}/")
        assert backend.check_available() is True
        assert "".join(backend.stream("hi")) == "ok"
        for call in responses_lib.calls:
            assert "//api" not in str(call.request.url).split("11434", 1)[1], call.request.url

    @responses_lib.activate
    def test_the_openai_compat_backend_keeps_a_path_prefix_intact(self):
        """
        Stripping the slash must not eat a path: ``/v1`` stays ``/v1``.

        A base pasted out of KoboldCpp's docs ends in ``/v1``, and dropping it
        would send every request to a path the server does not serve.
        """
        from protor.llm_backends import OpenAICompatBackend

        backend = OpenAICompatBackend("m", base_url="http://localhost:5001/v1/")
        assert backend.base_url == "http://localhost:5001/v1"


# ── 4. empty / error-only streams reported as successful analyses ─────────────


class TestEmptyStreamIsAFailure:
    """
    A stream carrying no text produced a report file, a "saved" line and
    exit 0.

    ``_iter_sse_text`` skips any frame without ``choices``, which is exactly the
    shape of llama.cpp's mid-stream error frame (``data: {"error": {...}}``,
    sent with HTTP 200 when a model fails to load or the context overflows).
    Nothing checked that *any* text arrived, so ``raw = ""`` became an
    ``AnalysisResult`` with an empty analysis — a failure presented as a
    finding.
    """

    @responses_lib.activate
    def test_an_error_only_frame_is_a_failure(self):
        """
        llama.cpp sends this with HTTP 200 when the model will not load.
        """
        from protor.llm_backends import OpenAICompatBackend

        body = 'data: {"error":{"code":500,"message":"Failed to load model"}}\n\ndata: [DONE]\n'
        responses_lib.add(
            responses_lib.POST, f"{LLAMA_URL}/v1/chat/completions", body=body, status=200
        )

        with pytest.raises(RuntimeHTTPError) as exc:
            list(OpenAICompatBackend("m", base_url=LLAMA_URL).stream("hi"))
        assert "Failed to load model" in str(exc.value)

    @responses_lib.activate
    def test_a_completely_empty_stream_is_a_failure(self):
        """
        No error frame, no text: the runtime accepted the request and said
        nothing. Reporting that as an analysis is the failure being fixed.
        """
        from protor.llm_backends import OpenAICompatBackend

        responses_lib.add(
            responses_lib.POST, f"{LLAMA_URL}/v1/chat/completions", body="", status=200
        )

        with pytest.raises(RuntimeHTTPError) as exc:
            list(OpenAICompatBackend("m", base_url=LLAMA_URL).stream("hi"))
        assert isinstance(exc.value, ProtorError)

    @responses_lib.activate
    def test_a_stream_of_keepalives_only_is_a_failure(self):
        """
        Only ``:`` comments and ``data: null`` frames arrived — a real shape,
        produced by a runtime that is alive but generating nothing.
        """
        from protor.llm_backends import OpenAICompatBackend

        body = ": keepalive\n\ndata: null\n\ndata: [DONE]\n"
        responses_lib.add(
            responses_lib.POST, f"{LLAMA_URL}/v1/chat/completions", body=body, status=200
        )

        with pytest.raises(RuntimeHTTPError):
            list(OpenAICompatBackend("m", base_url=LLAMA_URL).stream("hi"))

    @responses_lib.activate
    def test_text_still_arrives_normally(self):
        """The guard is on emptiness, not on the presence of an error frame."""
        from protor.llm_backends import OpenAICompatBackend

        chunk = {"choices": [{"index": 0, "delta": {"content": "ok"}}]}
        body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n"
        responses_lib.add(
            responses_lib.POST, f"{LLAMA_URL}/v1/chat/completions", body=body, status=200
        )

        assert "".join(OpenAICompatBackend("m", base_url=LLAMA_URL).stream("hi")) == "ok"

    @responses_lib.activate
    def test_a_partial_reply_is_still_returned(self):
        """
        Truncation after real text is returned, not raised.

        Existing behaviour, pinned because the emptiness check must not swallow
        a generation that mostly succeeded: raising here would discard tokens
        the user already paid for.
        """
        from protor.llm_backends import OpenAICompatBackend

        chunk = {"choices": [{"index": 0, "delta": {"content": "partial"}}]}
        responses_lib.add(
            responses_lib.POST,
            f"{LLAMA_URL}/v1/chat/completions",
            body=f"data: {json.dumps(chunk)}\n",
            status=200,
        )

        assert "".join(OpenAICompatBackend("m", base_url=LLAMA_URL).stream("hi")) == "partial"

    @responses_lib.activate
    def test_an_ollama_error_frame_is_a_failure(self):
        """Ollama's native stream reports errors in-band, the same way."""
        from protor.llm_backends import OllamaBackend

        responses_lib.add(
            responses_lib.POST,
            f"{OLLAMA_URL}/api/generate",
            body='{"error":"model requires more system memory"}\n',
            status=200,
        )

        with pytest.raises(RuntimeHTTPError) as exc:
            list(OllamaBackend("m").stream("hi"))
        assert "more system memory" in str(exc.value)

    def test_an_anthropic_error_frame_is_a_failure(self, monkeypatch):
        """
        Anthropic names its events, so ``type: "error"`` was dropped by the same
        ``!= "content_block_delta"`` test that drops the bookkeeping frames.
        """
        from protor.llm_backends import AnthropicBackend

        class Resp:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def iter_lines(self):
                yield 'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'

        monkeypatch.setattr("requests.post", lambda *a, **k: Resp())

        with pytest.raises(RuntimeHTTPError) as exc:
            list(AnthropicBackend("claude", api_key="k").stream("hi"))
        assert "Overloaded" in str(exc.value)

    def test_an_anthropic_empty_stream_is_a_failure(self, monkeypatch):
        """
        Anthropic's own framing: ``message_start`` then ``message_stop`` and
        nothing in between. Every frame is a bookkeeping event, so the old type
        filter dropped all of them and the analysis was empty.
        """
        from protor.llm_backends import AnthropicBackend

        class Resp:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def iter_lines(self):
                yield 'data: {"type":"message_start"}'
                yield 'data: {"type":"message_stop"}'

        monkeypatch.setattr("requests.post", lambda *a, **k: Resp())

        with pytest.raises(RuntimeHTTPError):
            list(AnthropicBackend("claude", api_key="k").stream("hi"))

    @responses_lib.activate
    def test_no_report_is_written_and_the_exit_code_is_nonzero(self, tmp_path):
        """
        The outcome that mattered: no ``analysis.json``, no "✓ saved", exit 1.
        """
        from protor.analyzer import analyze

        responses_lib.add(
            responses_lib.GET, f"{OLLAMA_URL}/api/tags", json={"models": []}, status=200
        )
        responses_lib.add(responses_lib.POST, f"{OLLAMA_URL}/api/generate", body="", status=200)

        sites = [
            {
                "url": "https://example.com/",
                "domain": "example.com",
                "html_file": "example.com.html",
                "js_count": 0,
                "metadata": {"title": "Example"},
                "text_content": "Real content worth analysing.",
                "js_files": [],
                "status": 200,
            }
        ]

        with pytest.raises(ProtorError):
            analyze(sites, model="m", output_dir=tmp_path, backend="ollama")

        assert not (tmp_path / "analysis.json").exists(), "an empty report was written"


# ── 5. schemaless URLs from the environment ───────────────────────────────────


class TestSchemalessBaseUrl:
    """
    ``export OLLAMA_HOST=0.0.0.0:11434`` — the documented way to bind Ollama to
    every interface — reported the runtime as "stopped".

    ``resolve_base_url`` stripped trailing slashes but not a missing scheme, so
    ``_probe`` built ``0.0.0.0:11434/api/tags``, ``requests`` raised
    ``MissingSchema``, the probe counted that as "not running", and the user was
    told to start a runtime that was already serving. The value is valid; the
    resolution was wrong.

    Rule pinned here: a value that is nothing but a host and an optional port
    gets ``http://``. Local runtimes speak plain HTTP, and a bare host cannot
    mean anything else. Anything carrying a scheme, a path or credentials is
    left exactly as given — those imply the user knows what they typed, and
    rewriting them would be the bug rather than the fix.
    """

    def test_a_bare_host_and_port_gets_http(self, monkeypatch):
        from protor.runtimes import resolve_base_url

        monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
        assert resolve_base_url("ollama") == "http://0.0.0.0:11434"

    def test_a_bare_hostname_gets_http(self, monkeypatch):
        from protor.runtimes import resolve_base_url

        monkeypatch.setenv("OLLAMA_HOST", "ollama.internal")
        assert resolve_base_url("ollama") == "http://ollama.internal"

    def test_an_explicit_schemeless_override_gets_http_too(self):
        """
        ``--base-url localhost:8080`` is the same mistake typed at the prompt.

        Applying the rule to whatever ``resolve_base_url`` produced — override or
        environment — is one rule, not two, and it keeps the two paths from
        drifting again.
        """
        from protor.runtimes import resolve_base_url

        assert resolve_base_url("lmstudio", "localhost:1234") == "http://localhost:1234"

    @pytest.mark.parametrize(
        "value",
        [
            "http://host:1234",
            "https://host:1234",
            "http://user:pw@host:1234",
            "http://gw:8080/api/v1",
        ],
    )
    def test_a_value_that_already_means_something_is_left_alone(self, value):
        from protor.runtimes import resolve_base_url

        assert resolve_base_url("lmstudio", value) == value

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # Credentials with no scheme are left alone: "user:pw" is not a host.
            ("user:pw@host:1234", "user:pw@host:1234"),
            # A path implies the user means a prefix, not a bare endpoint.
            ("host:1234/api", "host:1234/api"),
        ],
    )
    def test_values_that_are_not_bare_host_and_port_are_left_alone(self, value, expected):
        from protor.runtimes import resolve_base_url

        assert resolve_base_url("lmstudio", value) == expected

    @responses_lib.activate
    def test_the_probe_reaches_a_schemeless_ollama_host(self, monkeypatch):
        """
        End to end: the value is used as a URL, not reported as "stopped".

        Before the fix ``_probe`` raised ``MissingSchema`` inside its own
        ``except Exception`` and returned False — indistinguishable from a
        runtime that is not running.
        """
        from protor.runtimes import _probe, get_runtime

        monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
        responses_lib.add(
            responses_lib.GET, "http://127.0.0.1:11434/api/tags", json={"models": []}, status=200
        )

        assert _probe(get_runtime("ollama"), 1.0) is True

    @responses_lib.activate
    def test_the_ollama_backend_uses_the_resolved_url(self, monkeypatch):
        from protor.llm_backends import OllamaBackend

        monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
        responses_lib.add(
            responses_lib.GET, "http://127.0.0.1:11434/api/tags", json={"models": []}, status=200
        )

        backend = OllamaBackend("m")
        assert backend.base_url == "http://127.0.0.1:11434"
        assert backend.check_available() is True


# ── 6. _probe's comment promising an auth error that never came ───────────────


class TestAuthChallengeIsReportedByTheListingPath:
    """
    ``_probe``'s comment claimed "the backend's own check will surface the auth
    problem". No backend check does: ``OpenAICompatBackend.check_available``
    returns True on 401/403 and reports nothing, so a wrong API key surfaced as
    a raw ``RuntimeHTTPError`` from ``_check_status`` on the listing path —
    while ``stream()`` three lines later produced ``AuthError`` ("Set an API
    token for it.").

    The 401-counts-as-running decision is correct and stays: "something is
    listening" is the question ``protor runtimes`` asks, and
    ``test_auth_challenge_still_counts_as_running`` pins it. What was wrong was
    the promise made on the comment's behalf, so the promise is made true —
    ``list_models`` now raises ``AuthError``, the same type ``stream()`` raises
    for the same condition, and the comment describes that.
    """

    @responses_lib.activate
    @pytest.mark.parametrize("status", [401, 403])
    def test_the_listing_path_raises_the_same_auth_error_as_stream(self, status):
        from protor.llm_backends import OpenAICompatBackend

        responses_lib.add(responses_lib.GET, f"{LLAMA_URL}/v1/models", json={}, status=status)

        with pytest.raises(AuthError) as exc:
            OpenAICompatBackend("m", base_url=LLAMA_URL).list_models()

        assert exc.value.status == status
        assert "API token" in str(exc.value)
        assert isinstance(exc.value, ProtorError)

    @responses_lib.activate
    @pytest.mark.parametrize(
        ("backend", "kwargs", "url"),
        [
            ("ollama", {}, f"{OLLAMA_URL}/api/tags"),
            ("openai", {"api_key": "k"}, "https://api.openai.com/v1/models"),
            ("anthropic", {"api_key": "k"}, "https://api.anthropic.com/v1/models"),
        ],
    )
    def test_every_listing_path_says_it_is_a_token_problem(self, backend, kwargs, url):
        """
        The other three had the same gap: a 401 on ``/models`` fell through to
        ``_check_status`` and became a generic HTTP error.
        """
        from protor.llm_backends import create_backend as make

        responses_lib.add(responses_lib.GET, url, json={}, status=401)

        with pytest.raises(AuthError):
            make(backend, "unused", **kwargs).list_models()

    @responses_lib.activate
    def test_probe_still_counts_an_auth_challenge_as_running(self, monkeypatch):
        """The deliberate half of the decision, left intact."""
        from protor.runtimes import _probe, get_runtime

        monkeypatch.setenv("OLLAMA_HOST", "http://x:1")
        responses_lib.add(responses_lib.GET, "http://x:1/api/tags", json={}, status=401)

        assert _probe(get_runtime("ollama"), 1.0) is True

    def test_the_probe_comment_describes_what_actually_happens(self):
        """
        Guard on the prose, because the prose was the bug.

        The comment asserted a behaviour that no code implemented. Anyone
        reading it to decide what to do about a 401 would be misled, which is how
        the mismatch survived an audit.
        """
        import inspect

        from protor import runtimes

        docstring = inspect.getdoc(runtimes._probe) or ""
        body = inspect.getsource(runtimes._probe)
        assert "backend's own check will surface the auth problem" not in body
        # What it promises now must be the thing the code does: the listing
        # raises AuthError, so the token problem surfaces there.
        assert "AuthError" in docstring or "token" in docstring.lower()


def _stream_url(backend: str) -> str:
    """The POST URL each backend streams to."""
    return {
        "ollama": f"{OLLAMA_URL}/api/generate",
        "openai": "https://api.openai.com/v1/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
    }[backend]
