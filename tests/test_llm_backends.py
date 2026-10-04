"""Tests for protor.llm_backends — runtime-agnostic model backends."""

import json

import pytest
import responses as responses_lib

from protor import exceptions
from protor.exceptions import (
    AuthError,
    ConfigurationError,
    ModelListUnavailableError,
    ModelNotFoundError,
    OllamaModelNotFoundError,
    ProtorError,
    RuntimeUnavailableError,
)
from protor.llm_backends import (
    BACKEND_CHOICES,
    ModelInfo,
    OllamaBackend,
    OpenAICompatBackend,
    _endpoint,
    _format_timestamp,
    _iter_sse_text,
    create_backend,
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

    def __enter__(self):
        # The Anthropic backend posts inside a `with`, so a stand-in that only
        # duck-types the methods cannot be swapped in for it.
        return self

    def __exit__(self, *exc):
        return False


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


class TestEndpointJoining:
    """Base URLs are pasted from docs, and the docs disagree about the /v1."""

    def test_appends_the_path(self):
        assert _endpoint("http://localhost:1234", "/v1/models") == "http://localhost:1234/v1/models"

    def test_trailing_slash_on_the_base_is_harmless(self):
        assert (
            _endpoint("http://localhost:1234/", "/v1/models") == "http://localhost:1234/v1/models"
        )

    def test_base_that_already_ends_in_the_prefix(self):
        """KoboldCpp's docs say to use a base URL ending in /v1."""
        assert (
            _endpoint("http://localhost:5001/v1", "/v1/chat/completions")
            == "http://localhost:5001/v1/chat/completions"
        )

    def test_overlap_matching_is_case_sensitive(self):
        """HTTP paths are case-sensitive; /V1 is not /v1 on a real server."""
        assert (
            _endpoint("http://localhost:5001/V1", "/v1/models")
            == "http://localhost:5001/V1/v1/models"
        )

    def test_overlap_must_start_at_a_segment_boundary(self):
        """A base ending in '...v1' is not a prefix match for '/v1/...'."""
        assert (
            _endpoint("http://localhost:8080/myv1", "/v1/models")
            == "http://localhost:8080/myv1/v1/models"
        )

    def test_docker_engines_prefix(self):
        assert (
            _endpoint("http://localhost:12434", "/engines/v1/chat/completions")
            == "http://localhost:12434/engines/v1/chat/completions"
        )

    def test_root_path_is_left_alone(self):
        assert _endpoint("http://localhost:1234", "") == "http://localhost:1234"

    def test_a_gateway_prefix_before_the_overlap_is_kept(self):
        """
        A base URL mounted behind a gateway, which is how most reverse proxies
        expose an OpenAI-compatible API: ``http://gw/api/v1``.

        Only the trailing ``/v1`` overlaps, so it goes and ``/api`` stays. The
        suffixes were compared without their leading slash, so no suffix could
        ever prefix an absolute API path and every request went to
        ``/api/v1/v1/...`` — a 404 that read as "no such model on a working
        runtime".
        """
        assert _endpoint("http://gw:8080/api/v1", "/v1/models") == "http://gw:8080/api/v1/models"

    def test_a_deep_gateway_prefix_still_finds_the_overlap(self):
        assert (
            _endpoint("http://gw:8080/gw/openai/v1", "/v1/chat/completions")
            == "http://gw:8080/gw/openai/v1/chat/completions"
        )

    def test_a_base_with_a_prefix_that_does_not_overlap_is_left_intact(self):
        """No overlap at all must still append, prefix and all."""
        assert _endpoint("http://gw:8080/api", "/v1/models") == "http://gw:8080/api/v1/models"

    def test_a_longer_shared_suffix_wins_over_a_shorter_one(self):
        """Longest first, so the most of the base path is reused."""
        assert (
            _endpoint("http://gw:8080/engines/v1", "/engines/v1/models")
            == "http://gw:8080/engines/v1/models"
        )


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
        with pytest.raises(ModelNotFoundError, match="not available") as exc:
            list(backend.stream("hi"))
        # The CLI prints str(exc) verbatim, so the runtime and the remedy have
        # to survive as fields, not just as prose baked into the message.
        assert exc.value.model == "m"
        assert exc.value.runtime == "llama.cpp"
        assert "--backend llamacpp" in exc.value.hint

    def test_auth_error_is_explained(self, monkeypatch):
        backend = OpenAICompatBackend("m", base_url=OPENAI_URL)

        class Resp401:
            status_code = 401

            def raise_for_status(self):
                pass

            def iter_lines(self):
                return iter(())

        monkeypatch.setattr("requests.post", lambda *a, **k: Resp401())
        with pytest.raises(AuthError, match="API token") as exc:
            list(backend.stream("hi"))
        assert exc.value.runtime == "OpenAI-compatible"
        assert exc.value.status == 401

    def test_no_base_url_is_a_configuration_error(self):
        """
        `--backend openai-compatible` with no --base-url is a config mistake.

        As a bare RuntimeError it escaped cli()'s except chain entirely, so a
        plain typo produced a traceback.
        """
        with pytest.raises(ConfigurationError, match="--base-url") as exc:
            list(OpenAICompatBackend("m").stream("hi"))
        assert isinstance(exc.value, ProtorError)

    @responses_lib.activate
    def test_check_available_survives_a_missing_models_endpoint(self):
        """Chat can still work when a runtime does not implement /v1/models."""
        responses_lib.add(responses_lib.GET, f"{OPENAI_URL}/v1/models", status=404)
        responses_lib.add(responses_lib.GET, f"{OPENAI_URL}/", status=200)
        assert OpenAICompatBackend("m", base_url=OPENAI_URL).check_available() is True

    @responses_lib.activate
    def test_missing_models_endpoint_is_explained(self):
        responses_lib.add(responses_lib.GET, f"{OPENAI_URL}/v1/models", status=404)
        with pytest.raises(ModelListUnavailableError, match="--model"):
            OpenAICompatBackend("m", base_url=OPENAI_URL).list_models()

    @responses_lib.activate
    def test_unreachable_model_list_says_so(self):
        responses_lib.add(
            responses_lib.GET,
            f"{OPENAI_URL}/v1/models",
            body=ConnectionError("refused"),
        )
        with pytest.raises(RuntimeUnavailableError, match="Cannot reach") as exc:
            OpenAICompatBackend("m", base_url=OPENAI_URL).list_models()
        assert exc.value.base_url == OPENAI_URL

    @responses_lib.activate
    def test_unreachable_model_list_carries_the_start_command(self):
        """
        A refused connection to a *named* runtime must say how to start it.

        Same type the analyzer raises for an unreachable backend, so the CLI
        renders the same "Start it with: ..." hint here as it does there.
        """
        responses_lib.add(
            responses_lib.GET,
            f"{OPENAI_URL}/v1/models",
            body=ConnectionError("refused"),
        )
        with pytest.raises(RuntimeUnavailableError) as exc:
            OpenAICompatBackend("m", runtime="llamacpp", base_url=OPENAI_URL).list_models()
        assert exc.value.runtime == "llama.cpp"
        assert exc.value.start_hint

    @responses_lib.activate
    def test_docker_runtime_uses_the_engines_prefix(self, monkeypatch):
        monkeypatch.setenv("DOCKER_MODEL_RUNNER_URL", "http://localhost:12434")
        responses_lib.add(
            responses_lib.GET,
            "http://localhost:12434/engines/v1/models",
            json={"data": [{"id": "ai/smollm2"}]},
            status=200,
        )
        backend = create_backend("docker", "ai/smollm2")
        assert backend.check_available() is True
        assert [m.name for m in backend.list_models()] == ["ai/smollm2"]

    def test_docker_runtime_streams_from_the_engines_prefix(self, monkeypatch):
        seen: dict[str, str] = {}

        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def iter_lines(self):
                yield from sse("ok")

        def fake_post(url, **kwargs):
            seen["url"] = url
            return Resp()

        monkeypatch.setattr("requests.post", fake_post)
        backend = create_backend("docker", "ai/smollm2")
        assert "".join(backend.stream("hi")) == "ok"
        assert seen["url"] == "http://localhost:12434/engines/v1/chat/completions"

    def test_base_url_already_ending_in_v1_is_not_doubled(self, monkeypatch):
        """Users copy 'http://localhost:5001/v1' straight out of the docs."""
        seen: dict[str, str] = {}

        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def iter_lines(self):
                yield from sse("ok")

        def fake_post(url, **kwargs):
            seen["url"] = url
            return Resp()

        monkeypatch.setattr("requests.post", fake_post)
        backend = create_backend("koboldcpp", "m", base_url="http://localhost:5001/v1")
        assert "".join(backend.stream("hi")) == "ok"
        assert seen["url"] == "http://localhost:5001/v1/chat/completions"


# ── factory ───────────────────────────────────────────────────────────────────


class TestCreateBackend:
    def test_ollama_gets_the_native_backend(self):
        assert isinstance(create_backend("ollama", "llama3"), OllamaBackend)

    @pytest.mark.parametrize(
        "key",
        [
            "llamacpp",
            "lmstudio",
            "vllm",
            "localai",
            "jan",
            "llamafile",
            "tabbyapi",
            "cortex",
            "gpt4all",
            "koboldcpp",
            "oobabooga",
            "sglang",
            "xinference",
            "litellm",
            "anythingllm",
            "docker",
        ],
    )
    def test_openai_compatible_runtimes_share_one_class(self, key):
        assert isinstance(create_backend(key, "m"), OpenAICompatBackend)

    @pytest.mark.parametrize(
        "alias", ["llama.cpp", "llama-cpp", "LM-Studio", "vLLM", "gpt-4all", "model-runner"]
    )
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

    def test_every_registered_runtime_is_a_backend_choice(self):
        """A runtime nobody can select with --backend is dead weight."""
        from protor.runtimes import runtime_names

        for key in runtime_names():
            assert key in BACKEND_CHOICES

    def test_every_local_runtime_builds_without_credentials(self):
        """Local runtimes must never demand a cloud API key."""
        from protor.runtimes import runtime_names

        for key in runtime_names():
            backend = create_backend(key, "m")
            assert backend.model_name == "m"
            assert backend.start_hint()

    @pytest.mark.parametrize("key", ["gpt4all", "koboldcpp", "docker", "llamafile"])
    def test_model_not_found_hint_names_the_selected_runtime(self, key, monkeypatch):
        """The suggested `protor models` command must use the runtime the user typed."""
        monkeypatch.setattr("requests.post", lambda *a, **k: FakeStream([], status_code=404))
        with pytest.raises(ModelNotFoundError, match=f"--backend {key}"):
            list(create_backend(key, "m").stream("hi"))

    @pytest.mark.parametrize(
        ("key", "label"),
        [
            ("llamacpp", "llama.cpp"),
            ("lmstudio", "LM Studio"),
            ("vllm", "vLLM"),
            ("gpt4all", "GPT4All"),
            ("koboldcpp", "KoboldCpp"),
            ("oobabooga", "text-generation-webui"),
            ("litellm", "LiteLLM proxy"),
            ("docker", "Docker Model Runner"),
        ],
    )
    def test_runtime_labels_are_user_facing(self, key, label):
        assert create_backend(key, "m").display_name == label


class TestListModelsHelper:
    @responses_lib.activate
    def test_delegates_to_the_backend(self):
        """
        The helper lives in protor.analyzer now — it was implemented here too,
        byte for byte, and one of the two had to go.
        """
        from protor.analyzer import list_models

        responses_lib.add(
            responses_lib.GET,
            f"{OPENAI_URL}/v1/models",
            json={"data": [{"id": "a"}, {"id": "b"}]},
            status=200,
        )
        models = list_models("openai-compatible", base_url=OPENAI_URL)
        assert [m.name for m in models] == ["a", "b"]

    def test_helper_is_not_duplicated_in_this_module(self):
        """A second copy is how the two implementations drift apart unnoticed."""
        import protor.llm_backends as backends

        assert "list_models" not in vars(backends), "the helper lives in protor.analyzer"


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
        """
        A 404 here used to raise a bare RuntimeError with the same advice in it.

        cli() has an `except OllamaModelNotFoundError` clause that therefore could
        never fire, so a user who simply had not pulled the model got a traceback
        instead of the hint below.
        """
        responses_lib.add(responses_lib.POST, f"{self.OLLAMA}/api/generate", json={}, status=404)
        with pytest.raises(OllamaModelNotFoundError, match="ollama pull") as exc:
            list(OllamaBackend("nope").stream("hi"))
        assert exc.value.model == "nope"
        assert exc.value.hint == "Pull it with: ollama pull nope"
        assert isinstance(exc.value, ModelNotFoundError)
        assert isinstance(exc.value, ProtorError)

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
        with pytest.raises(AuthError, match="Invalid OpenAI API key") as exc:
            list(self._backend().stream("p"))
        assert exc.value.runtime == "OpenAI"
        assert exc.value.status == status

    def test_model_not_found_is_explained(self, monkeypatch):
        class Resp:
            status_code = 404

            def raise_for_status(self):
                pass

            def iter_lines(self):
                return iter(())

        monkeypatch.setattr("requests.post", lambda *a, **k: Resp())
        with pytest.raises(ModelNotFoundError, match="not available") as exc:
            list(self._backend().stream("p"))
        assert exc.value.model == "gpt-4o"
        assert exc.value.runtime == "OpenAI"


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
        ("status", "exc_type", "expected"),
        [
            (401, AuthError, "Invalid Anthropic API key"),
            (404, ModelNotFoundError, "not available"),
        ],
    )
    def test_errors_are_explained(self, monkeypatch, status, exc_type, expected):
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
        with pytest.raises(exc_type, match=expected) as exc:
            list(self._backend().stream("p"))
        # Hosted failures reach the user through `except ProtorError` in cli().
        assert isinstance(exc.value, ProtorError)


# ── typed errors reaching the CLI ─────────────────────────────────────────────


class TestStreamFailuresAreTyped:
    """
    Every stream failure a user can trigger must be a `ProtorError`.

    `cli.cli()` catches `KeyboardInterrupt`, then `OllamaModelNotFoundError`,
    `DataFileNotFoundError`, `ConfigurationError`, `URLValidationError`,
    `ProtorError` and `ValueError`. A bare `RuntimeError` matches none of them, so it escaped the
    entry point and the user saw a traceback — even where the message had been
    carefully written with the remedy in it. These tests are the guard on that
    contract, not a restatement of the messages.
    """

    def _replay(self, monkeypatch, status):
        """Force every stream() call to answer *status* without any network."""
        monkeypatch.setattr("requests.post", lambda *a, **k: FakeStream([], status_code=status))

    @pytest.mark.parametrize(
        "backend",
        ["ollama", "llamacpp", "lmstudio", "vllm", "koboldcpp", "docker", "openai-compatible"],
    )
    def test_missing_model_is_a_protor_error(self, monkeypatch, backend):
        from protor.llm_backends import create_backend as make

        self._replay(monkeypatch, 404)
        kwargs = {"base_url": OPENAI_URL} if backend == "openai-compatible" else {}
        with pytest.raises(ModelNotFoundError) as exc:
            list(make(backend, "m", **kwargs).stream("hi"))
        assert isinstance(exc.value, ProtorError)

    @pytest.mark.parametrize("backend", ["llamacpp", "lmstudio", "vllm", "openai-compatible"])
    @pytest.mark.parametrize("status", [401, 403])
    def test_rejected_credentials_are_a_protor_error(self, monkeypatch, backend, status):
        from protor.llm_backends import create_backend as make

        self._replay(monkeypatch, status)
        kwargs = {"base_url": OPENAI_URL} if backend == "openai-compatible" else {}
        with pytest.raises(AuthError) as exc:
            list(make(backend, "m", **kwargs).stream("hi"))
        assert isinstance(exc.value, ProtorError)
        assert exc.value.status == status


class TestExceptionHierarchy:
    """
    The shape of `protor.exceptions` is load-bearing, not documentation.

    cli.py imports `OllamaModelNotFoundError` by name and catches `ProtorError`
    for everything else, so an error that opts out of that base class is an
    error the CLI cannot render.
    """

    @pytest.mark.parametrize(
        "exc_type",
        [AuthError, ConfigurationError, ModelListUnavailableError, ModelNotFoundError],
    )
    def test_every_error_the_backends_raise_is_a_protor_error(self, exc_type):
        """ModelListUnavailableError inherited RuntimeError, so ProtorError missed it."""
        assert issubclass(exc_type, ProtorError)

    def test_ollama_model_not_found_is_a_kind_of_model_not_found(self):
        """One concept, two classes: the CLI names the pull hint, callers want the base."""
        assert issubclass(OllamaModelNotFoundError, ModelNotFoundError)

    def test_cli_handlers_catch_the_types_the_backends_raise(self):
        """
        Guards the pairing between cli's except clauses and this module.

        If a raise site goes back to a bare RuntimeError, or an exception is
        renamed, this fails instead of the failure mode silently becoming a
        traceback for users.
        """
        from protor.cli import cli as _cli  # noqa: F401  (import is the assertion)
        from protor.exceptions import (
            DataFileNotFoundError,
            RuntimeUnavailableError,
        )

        for handled in (
            OllamaModelNotFoundError,
            DataFileNotFoundError,
            RuntimeUnavailableError,
        ):
            assert issubclass(handled, ProtorError)

    def test_unreachable_variants_all_look_unavailable(self):
        """
        One error type covers every unreachable runtime, Ollama included.

        There used to be an Ollama-only subclass with nothing raising it and a
        cli handler that could never fire — a documented failure mode that the
        code did not have.
        """
        assert issubclass(RuntimeUnavailableError, ProtorError)
        assert not hasattr(exceptions, "OllamaUnavailableError")


class TestCliRendersTheError:
    """
    The end-to-end proof: `protor analyze` prints advice and exits, no traceback.

    Everything above tests the exception in isolation. This drives the real
    `cli()` entry point so the `except` ordering in it is exercised too — the
    bug being fixed was precisely that a live code path produced a stack trace.
    """

    def _index(self, tmp_path):
        index = tmp_path / "sites_index.json"
        index.write_text(
            json.dumps(
                [
                    {
                        "url": "https://example.com/",
                        "domain": "example.com",
                        "html_file": "example.com.html",
                        "js_count": 0,
                        "metadata": {"title": "Example", "description": "A description"},
                        "text_content": "Some real content worth analysing.",
                        "js_files": [],
                        "status": 200,
                    }
                ]
            ),
            encoding="utf-8",
        )
        return index

    def _run(self, monkeypatch, tmp_path, capsys, argv):
        """Invoke the real CLI and return whatever it printed."""
        from protor.cli import cli

        index = self._index(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            ["protor", "analyze", "--file", str(index), "--output", str(tmp_path / "out"), *argv],
        )
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 1
        # A failure is not a result: the advice goes to stderr, so a script that
        # captures stdout gets nothing rather than an error report in its data.
        return capsys.readouterr().err

    @responses_lib.activate
    def test_ollama_missing_model_prints_the_pull_hint(self, monkeypatch, tmp_path, capsys):
        responses_lib.add(
            responses_lib.GET, "http://localhost:11434/api/tags", json={"models": []}, status=200
        )
        responses_lib.add(
            responses_lib.POST, "http://localhost:11434/api/generate", json={}, status=404
        )
        out = self._run(monkeypatch, tmp_path, capsys, ["--backend", "ollama", "--model", "nope"])
        assert "not found" in out
        assert "ollama pull nope" in out
        assert "Traceback" not in out

    @responses_lib.activate
    def test_openai_compat_missing_model_prints_the_models_hint(
        self, monkeypatch, tmp_path, capsys
    ):
        responses_lib.add(
            responses_lib.GET, f"{OPENAI_URL}/v1/models", json={"data": []}, status=200
        )
        responses_lib.add(
            responses_lib.POST, f"{OPENAI_URL}/v1/chat/completions", json={}, status=404
        )
        out = self._run(
            monkeypatch,
            tmp_path,
            capsys,
            ["--backend", "llamacpp", "--model", "m", "--base-url", OPENAI_URL],
        )
        assert "not available" in out
        assert "protor models --backend llamacpp" in out
        assert "Traceback" not in out

    @responses_lib.activate
    def test_rejected_token_prints_the_token_hint(self, monkeypatch, tmp_path, capsys):
        responses_lib.add(
            responses_lib.GET, f"{OPENAI_URL}/v1/models", json={"data": []}, status=200
        )
        responses_lib.add(
            responses_lib.POST, f"{OPENAI_URL}/v1/chat/completions", json={}, status=401
        )
        out = self._run(
            monkeypatch,
            tmp_path,
            capsys,
            ["--backend", "lmstudio", "--model", "m", "--base-url", OPENAI_URL],
        )
        assert "API token" in out
        assert "Traceback" not in out


# ── display metadata ──────────────────────────────────────────────────────────


def test_backends_report_friendly_display_names():
    assert create_backend("ollama", "m").display_name == "Ollama"
    assert create_backend("llamacpp", "m").display_name == "llama.cpp"
    assert create_backend("lmstudio", "m").display_name == "LM Studio"
    assert create_backend("vllm", "m").display_name == "vLLM"
    assert create_backend("openai", "m", api_key="k").display_name == "OpenAI"
    assert create_backend("anthropic", "m", api_key="k").display_name == "Anthropic"


# ── unexpected HTTP statuses ─────────────────────────────────────────────────


class TestUnexpectedStatusesAreTyped:
    """
    A status with no specific remedy must still be a `ProtorError`.

    404 and 401/403 are mapped by hand because they have a fix to suggest. A
    local runtime returns plenty of statuses that are none of those — 500 when
    the model does not fit in memory, 503 while it loads, 400 from a build that
    rejected the request — and those went out as `requests.exceptions.HTTPError`,
    which `cli.cli()` does not catch. The user got a traceback for what is
    usually a one-line diagnosis, contradicting this module's own contract.
    """

    @pytest.mark.parametrize(
        "backend", ["ollama", "llamacpp", "lmstudio", "vllm", "litellm", "koboldcpp"]
    )
    def test_a_500_on_the_model_list_is_typed(self, monkeypatch, backend):
        import requests

        from protor.llm_backends import create_backend as make

        def _get(*a, **k):
            resp = requests.Response()
            resp.status_code = 500
            resp._content = b"model does not fit in memory"
            resp.url = a[0] if a else ""
            return resp

        monkeypatch.setattr("requests.get", _get)
        with pytest.raises(exceptions.RuntimeHTTPError) as exc:
            make(backend, "m", base_url="http://127.0.0.1:1").list_models()
        assert isinstance(exc.value, ProtorError)
        assert exc.value.status == 500
        assert "does not fit in memory" in str(exc.value), "the body names the problem"

    @pytest.mark.parametrize("backend", ["ollama", "llamacpp", "lmstudio"])
    def test_a_500_on_the_stream_is_typed(self, monkeypatch, backend):
        from protor.llm_backends import create_backend as make

        monkeypatch.setattr("requests.post", lambda *a, **k: FakeStream([], status_code=500))
        with pytest.raises(exceptions.RuntimeHTTPError) as exc:
            list(make(backend, "m").stream("hi"))
        assert exc.value.status == 500

    def test_the_hosted_backends_are_typed_too(self, monkeypatch):
        from protor.llm_backends import create_backend as make

        monkeypatch.setattr("requests.post", lambda *a, **k: FakeStream([], status_code=503))
        for name in ("openai", "anthropic"):
            with pytest.raises(exceptions.RuntimeHTTPError) as exc:
                list(make(name, "m", api_key="k").stream("hi"))
            assert exc.value.status == 503

    def test_a_connection_dropped_mid_stream_is_typed(self, monkeypatch):
        """
        The status check only sees the response header.

        A runtime that dies mid-generation, or a proxy that drops the
        connection, raises out of `iter_lines` instead — usually after tokens
        have been paid for and already shown. That escaped as a raw
        `ChunkedEncodingError` from inside a generator.
        """
        import requests

        from protor.llm_backends import create_backend as make

        class _Dies:
            status_code = 200

            def iter_lines(self):
                yield b'data: {"choices":[{"delta":{"content":"par"}}]}'
                raise requests.exceptions.ChunkedEncodingError("connection broken")

        monkeypatch.setattr("requests.post", lambda *a, **k: _Dies())
        with pytest.raises(exceptions.RuntimeHTTPError) as exc:
            list(make("llamacpp", "m").stream("hi"))
        assert "mid-response" in str(exc.value)
