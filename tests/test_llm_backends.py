"""Tests for protor.llm_backends — runtime-agnostic model backends."""

import json

import pytest
import responses as responses_lib

from protor.llm_backends import (
    BACKEND_CHOICES,
    ModelInfo,
    OllamaBackend,
    OpenAICompatBackend,
    _format_timestamp,
    _iter_sse_text,
    create_backend,
    list_models,
)
from protor.runtimes import resolve_base_url

OPENAI_URL = "http://localhost:8080"


class FakeStream:
    """Minimal stand-in for a streaming ``requests`` response."""

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")

    def iter_lines(self):
        yield from self._lines


def sse(*texts, done=True):
    """Build an OpenAI-compatible SSE body yielding *texts* as content deltas."""
    lines = []
    for text in texts:
        chunk = {
            "id": "c1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")
    if done:
        lines.append("data: [DONE]")
        lines.append("")
    return lines


# ── SSE parsing ───────────────────────────────────────────────────────────────


class TestSseParsing:
    def test_yields_content_deltas(self):
        stream = FakeStream(sse("Hello", " ", "world"))
        assert "".join(_iter_sse_text(stream)) == "Hello world"

    def test_stops_at_done_sentinel(self):
        lines = [*sse("a"), "data: {not json}", 'data: {"choices":[{"delta":{"content":"b"}}]}']
        assert "".join(_iter_sse_text(FakeStream(lines))) == "a"

    def test_handles_bytes_lines(self):
        stream = FakeStream(
            [b"data: " + json.dumps({"choices": [{"delta": {"content": "bytes ok"}}]}).encode()]
        )
        assert "".join(_iter_sse_text(stream)) == "bytes ok"

    def test_skips_comments_and_blank_lines(self):
        lines = [
            ": ping",
            "",
            "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
            "",
        ]
        assert "".join(_iter_sse_text(FakeStream(lines))) == "ok"

    def test_accepts_bare_json_without_data_prefix(self):
        lines = [json.dumps({"choices": [{"delta": {"content": "bare"}}]})]
        assert "".join(_iter_sse_text(FakeStream(lines))) == "bare"

    def test_ignores_reasoning_content(self):
        """Reasoning traces should not be shown as if they were the answer."""
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "thinking..."}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "answer"}}]}),
        ]
        assert "".join(_iter_sse_text(FakeStream(lines))) == "answer"

    def test_tolerates_usage_only_final_chunk(self):
        lines = [
            *sse("done", done=False),
            "data: " + json.dumps({"choices": [], "usage": {"total_tokens": 5}}),
            "data: [DONE]",
        ]
        assert "".join(_iter_sse_text(FakeStream(lines))) == "done"

    def test_skips_malformed_json(self):
        lines = [
            "data: {broken",
            "data: " + json.dumps({"choices": [{"delta": {"content": "survived"}}]}),
        ]
        assert "".join(_iter_sse_text(FakeStream(lines))) == "survived"


# ── timestamps ────────────────────────────────────────────────────────────────


class TestTimestampFormatting:
    def test_epoch_int_becomes_a_date(self):
        """OpenAI's `created` is an epoch int; slicing it gave '1750000000'."""
        assert _format_timestamp(1750000000) == "2025-06-15"

    def test_iso_string_is_trimmed_to_a_date(self):
        assert _format_timestamp("2026-01-02T03:04:05Z") == "2026-01-02"

    @pytest.mark.parametrize("value", [None, ""])
    def test_missing_is_empty(self, value):
        assert _format_timestamp(value) == ""


# ── OpenAI-compatible backend ────────────────────────────────────────────────


class TestOpenAICompatBackend:
    def test_resolves_runtime_url(self, monkeypatch):
        monkeypatch.delenv("LMSTUDIO_URL", raising=False)
        b = OpenAICompatBackend("m", runtime="lmstudio")
        assert b.base_url == "http://localhost:1234"
        assert b.display_name == "LM Studio"

    def test_base_url_override(self):
        b = OpenAICompatBackend("m", base_url="http://gpu:9000/")
        assert b.base_url == "http://gpu:9000"

    def test_explicit_base_url_for_unknown_runtime(self):
        b = OpenAICompatBackend("m", base_url="http://box:7000")
        assert b.base_url == "http://box:7000"
        assert b.display_name == "OpenAI-compatible"

    def test_missing_base_url_is_not_available(self):
        assert OpenAICompatBackend("m").check_available() is False

    @responses_lib.activate
    def test_check_available(self):
        responses_lib.add(
            responses_lib.GET, f"{OPENAI_URL}/v1/models", json={"data": []}, status=200
        )
        assert OpenAICompatBackend("m", base_url=OPENAI_URL).check_available() is True

    @responses_lib.activate
    def test_unreachable_is_not_available(self):
        responses_lib.add(
            responses_lib.GET,
            f"{OPENAI_URL}/v1/models",
            body=ConnectionError("refused"),
        )
        assert OpenAICompatBackend("m", base_url=OPENAI_URL).check_available() is False

    @responses_lib.activate
    def test_list_models_parses_size_and_date(self):
        responses_lib.add(
            responses_lib.GET,
            f"{OPENAI_URL}/v1/models",
            json={
                "data": [
                    {"id": "qwen3-8b-q4", "size": 4912898304, "created": 1750000000},
                    {"id": "no-meta"},
                ]
            },
            status=200,
        )
        models = OpenAICompatBackend("m", base_url=OPENAI_URL).list_models()
        assert [m.name for m in models] == ["qwen3-8b-q4", "no-meta"]
        assert models[0].size_bytes == 4912898304
        assert models[0].modified == "2025-06-15"
        assert models[1].modified == ""

    def test_stream_yields_deltas(self, monkeypatch):
        backend = OpenAICompatBackend("m", base_url=OPENAI_URL)

        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def iter_lines(self):
                yield from sse("one", "two")

        monkeypatch.setattr("requests.post", lambda *a, **k: Resp())
        assert "".join(backend.stream("hi")) == "onetwo"

    def test_model_not_found_names_the_runtime(self, monkeypatch):
        backend = OpenAICompatBackend("m", runtime="llamacpp")

        class Resp404:
            status_code = 404

            def raise_for_status(self):
                pass

            def iter_lines(self):
                return iter(())

        monkeypatch.setattr("requests.post", lambda *a, **k: Resp404())
        with pytest.raises(RuntimeError, match="not available"):
            list(backend.stream("hi"))

    def test_auth_error_is_explained(self, monkeypatch):
        backend = OpenAICompatBackend("m", base_url=OPENAI_URL)

        class Resp401:
            status_code = 401

            def raise_for_status(self):
                pass

            def iter_lines(self):
                return iter(())

        monkeypatch.setattr("requests.post", lambda *a, **k: Resp401())
        with pytest.raises(RuntimeError, match="API token"):
            list(backend.stream("hi"))


# ── factory ───────────────────────────────────────────────────────────────────


class TestCreateBackend:
    def test_ollama_gets_the_native_backend(self):
        assert isinstance(create_backend("ollama", "llama3"), OllamaBackend)

    @pytest.mark.parametrize("key", ["llamacpp", "lmstudio", "vllm", "localai", "jan"])
    def test_openai_compatible_runtimes_share_one_class(self, key):
        assert isinstance(create_backend(key, "m"), OpenAICompatBackend)

    @pytest.mark.parametrize("alias", ["llama.cpp", "llama-cpp", "LM-Studio", "vLLM"])
    def test_aliases_work_through_the_factory(self, alias):
        assert isinstance(create_backend(alias, "m"), OpenAICompatBackend)

    def test_generic_openai_compatible_needs_a_url(self):
        b = create_backend("openai-compatible", "m", base_url=OPENAI_URL)
        assert isinstance(b, OpenAICompatBackend)

    def test_hosted_backends(self):
        from protor.llm_backends import AnthropicBackend, OpenAIBackend

        assert isinstance(create_backend("openai", "gpt-4o", api_key="k"), OpenAIBackend)
        assert isinstance(create_backend("anthropic", "claude", api_key="k"), AnthropicBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown runtime"):
            create_backend("nope", "m")

    def test_all_choices_are_constructible_or_named(self):
        assert "openai-compatible" in BACKEND_CHOICES
        for key in ("ollama", "lmstudio", "vllm"):
            assert key in BACKEND_CHOICES

    def test_every_local_runtime_builds_without_credentials(self):
        """Local runtimes must never demand a cloud API key."""
        for key in ("ollama", "llamacpp", "lmstudio", "vllm", "localai", "jan"):
            backend = create_backend(key, "m")
            assert backend.model_name == "m"
            assert backend.start_hint()


class TestListModelsHelper:
    @responses_lib.activate
    def test_delegates_to_the_backend(self):
        responses_lib.add(
            responses_lib.GET,
            f"{OPENAI_URL}/v1/models",
            json={"data": [{"id": "a"}, {"id": "b"}]},
            status=200,
        )
        models = list_models("openai-compatible", base_url=OPENAI_URL)
        assert [m.name for m in models] == ["a", "b"]


def test_ollama_backend_url_matches_registry(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert OllamaBackend("m").base_url == resolve_base_url("ollama")


def test_model_info_defaults():
    m = ModelInfo(name="x")
    assert m.size_bytes is None
    assert m.modified == ""


# ── Ollama native API ─────────────────────────────────────────────────────────


class TestOllamaBackend:
    OLLAMA = "http://localhost:11434"

    @responses_lib.activate
    def test_check_available(self):
        responses_lib.add(
            responses_lib.GET, f"{self.OLLAMA}/api/tags", json={"models": []}, status=200
        )
        assert OllamaBackend("m").check_available() is True

    @responses_lib.activate
    def test_check_unavailable(self):
        responses_lib.add(
            responses_lib.GET, f"{self.OLLAMA}/api/tags", body=ConnectionError("refused")
        )
        assert OllamaBackend("m").check_available() is False

    @responses_lib.activate
    def test_list_models(self):
        responses_lib.add(
            responses_lib.GET,
            f"{self.OLLAMA}/api/tags",
            json={
                "models": [
                    {
                        "name": "llama3:latest",
                        "size": 4_000_000_000,
                        "modified_at": "2026-01-02T03:04:05Z",
                    }
                ]
            },
            status=200,
        )
        models = OllamaBackend("m").list_models()
        assert models[0].name == "llama3:latest"
        assert models[0].size_bytes == 4_000_000_000
        assert models[0].modified == "2026-01-02"

    @responses_lib.activate
    def test_stream_reads_newline_delimited_json(self):
        body = '{"response":"Hello ","done":false}\n{"response":"world","done":true}\n'
        responses_lib.add(
            responses_lib.POST,
            f"{self.OLLAMA}/api/generate",
            body=body,
            status=200,
        )
        assert "".join(OllamaBackend("m").stream("hi")) == "Hello world"

    @responses_lib.activate
    def test_missing_model_says_how_to_pull(self):
        responses_lib.add(responses_lib.POST, f"{self.OLLAMA}/api/generate", json={}, status=404)
        with pytest.raises(RuntimeError, match="ollama pull"):
            list(OllamaBackend("nope").stream("hi"))

    def test_start_hint_names_the_command(self):
        assert OllamaBackend("m").start_hint() == "ollama serve"


# ── hosted APIs ───────────────────────────────────────────────────────────────


class TestOpenAIBackend:
    API = "https://api.openai.com/v1"

    def _backend(self):
        from protor.llm_backends import OpenAIBackend

        return OpenAIBackend("gpt-4o", api_key="k")

    def test_requires_an_api_key(self, monkeypatch):
        from protor.llm_backends import OpenAIBackend

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIBackend("gpt-4o")

    @responses_lib.activate
    def test_check_available(self):
        responses_lib.add(responses_lib.GET, f"{self.API}/models", json={"data": []}, status=200)
        assert self._backend().check_available() is True

    @responses_lib.activate
    def test_list_models(self):
        responses_lib.add(
            responses_lib.GET, f"{self.API}/models", json={"data": [{"id": "gpt-4o"}]}, status=200
        )
        assert [m.name for m in self._backend().list_models()] == ["gpt-4o"]

    def test_stream_yields_deltas(self, monkeypatch):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def iter_lines(self):
                yield from sse("hi")

        monkeypatch.setattr("requests.post", lambda *a, **k: Resp())
        assert "".join(self._backend().stream("p")) == "hi"

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failure_is_explained(self, monkeypatch, status):
        class Resp:
            status_code = status

            def raise_for_status(self):
                pass

            def iter_lines(self):
                return iter(())

        monkeypatch.setattr("requests.post", lambda *a, **k: Resp())
        with pytest.raises(RuntimeError, match="Invalid OpenAI API key"):
            list(self._backend().stream("p"))

    def test_model_not_found_is_explained(self, monkeypatch):
        class Resp:
            status_code = 404

            def raise_for_status(self):
                pass

            def iter_lines(self):
                return iter(())

        monkeypatch.setattr("requests.post", lambda *a, **k: Resp())
        with pytest.raises(RuntimeError, match="not available"):
            list(self._backend().stream("p"))


class TestAnthropicBackend:
    API = "https://api.anthropic.com/v1"

    def _backend(self):
        from protor.llm_backends import AnthropicBackend

        return AnthropicBackend("claude-sonnet", api_key="k")

    def test_requires_an_api_key(self, monkeypatch):
        from protor.llm_backends import AnthropicBackend

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicBackend("claude")

    @responses_lib.activate
    def test_check_available_accepts_valid_key_rejecting_probe(self):
        """A 400 still proves the key authenticated."""
        responses_lib.add(responses_lib.POST, f"{self.API}/messages", json={}, status=400)
        assert self._backend().check_available() is True

    @responses_lib.activate
    def test_check_unavailable(self):
        responses_lib.add(responses_lib.POST, f"{self.API}/messages", json={}, status=401)
        assert self._backend().check_available() is False

    @responses_lib.activate
    def test_list_models(self):
        responses_lib.add(
            responses_lib.GET,
            f"{self.API}/models",
            json={"data": [{"id": "claude-sonnet", "created_at": "2026-03-04T00:00:00Z"}]},
            status=200,
        )
        models = self._backend().list_models()
        assert models[0].name == "claude-sonnet"
        assert models[0].modified == "2026-03-04"

    def test_stream_reads_content_block_deltas(self, monkeypatch):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def iter_lines(self):
                yield "event: message_start"
                yield "data: " + json.dumps({"type": "message_start"})
                yield "data: " + json.dumps(
                    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}
                )
                yield "data: " + json.dumps({"type": "message_stop"})

        monkeypatch.setattr("requests.post", lambda *a, **k: Resp())
        assert "".join(self._backend().stream("p")) == "Hi"

    @pytest.mark.parametrize(
        ("status", "expected"),
        [(401, "Invalid Anthropic API key"), (404, "not available")],
    )
    def test_errors_are_explained(self, monkeypatch, status, expected):
        class Resp:
            status_code = status

            def raise_for_status(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def iter_lines(self):
                return iter(())

        monkeypatch.setattr("requests.post", lambda *a, **k: Resp())
        with pytest.raises(RuntimeError, match=expected):
            list(self._backend().stream("p"))


# ── display metadata ──────────────────────────────────────────────────────────


def test_backends_report_friendly_display_names():
    assert create_backend("ollama", "m").display_name == "Ollama"
    assert create_backend("llamacpp", "m").display_name == "llama.cpp"
    assert create_backend("lmstudio", "m").display_name == "LM Studio"
    assert create_backend("vllm", "m").display_name == "vLLM"
    assert create_backend("openai", "m", api_key="k").display_name == "OpenAI"
    assert create_backend("anthropic", "m", api_key="k").display_name == "Anthropic"
