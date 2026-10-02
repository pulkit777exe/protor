"""Tests for protor.runtimes — the local model runtime registry."""

import pytest
import responses as responses_lib

from protor.runtimes import (
    RUNTIMES,
    _probe,
    detect_runtimes,
    get_runtime,
    resolve_api_key,
    resolve_base_url,
    runtime_names,
)


class TestRegistry:
    def test_known_runtimes_are_registered(self):
        for key in ("ollama", "llamacpp", "lmstudio", "vllm", "localai", "jan"):
            assert key in runtime_names()

    def test_documented_default_urls(self):
        # Values taken from each project's own documentation.
        assert get_runtime("ollama").default_url == "http://localhost:11434"
        assert get_runtime("llamacpp").default_url == "http://localhost:8080"
        assert get_runtime("lmstudio").default_url == "http://localhost:1234"
        assert get_runtime("vllm").default_url == "http://localhost:8000"

    def test_every_runtime_has_a_start_hint_and_docs(self):
        for runtime in RUNTIMES.values():
            assert runtime.start_hint
            assert runtime.docs.startswith("http")

    def test_only_ollama_uses_its_native_api(self):
        """Everything else is OpenAI-compatible, so they share one backend."""
        native = {r.key for r in RUNTIMES.values() if r.api == "ollama"}
        compat = {r.key for r in RUNTIMES.values() if r.api == "openai"}
        assert native == {"ollama"}
        assert "llamacpp" in compat and "lmstudio" in compat and "vllm" in compat

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("llama.cpp", "llamacpp"),
            ("llama-cpp", "llamacpp"),
            ("llama", "llamacpp"),
            ("LM-Studio", "lmstudio"),
            ("lm_studio", "lmstudio"),
            ("vLLM", "vllm"),
            ("local-ai", "localai"),
            ("  Ollama  ", "ollama"),
        ],
    )
    def test_aliases_resolve(self, alias, expected):
        assert get_runtime(alias).key == expected

    def test_unknown_runtime_lists_valid_options(self):
        with pytest.raises(ValueError) as exc:
            get_runtime("definitely-not-a-runtime")
        message = str(exc.value)
        assert "definitely-not-a-runtime" in message
        assert "ollama" in message and "lmstudio" in message


class TestResolution:
    def test_default_used_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("LMSTUDIO_URL", raising=False)
        assert resolve_base_url("lmstudio") == "http://localhost:1234"

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_URL", "http://gpu-box:9999")
        assert resolve_base_url("lmstudio") == "http://gpu-box:9999"

    def test_explicit_override_beats_env(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_URL", "http://from-env:1")
        assert resolve_base_url("lmstudio", "http://explicit:2") == "http://explicit:2"

    def test_trailing_slash_is_stripped(self):
        """Path concatenation must not produce 'http://host:1234//v1/models'."""
        assert resolve_base_url("ollama", "http://host:11434/") == "http://host:11434"

    def test_no_api_key_means_no_header(self, monkeypatch):
        monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)
        assert resolve_api_key("lmstudio") is None

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_API_KEY", "tok-from-env")
        assert resolve_api_key("lmstudio") == "tok-from-env"

    def test_api_key_override_wins(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_API_KEY", "tok-from-env")
        assert resolve_api_key("lmstudio", "explicit") == "explicit"


class TestDetection:
    def test_detection_reports_running_only(self, monkeypatch):
        monkeypatch.setattr(
            "protor.runtimes._probe", lambda runtime, timeout: runtime.key == "vllm"
        )
        detected = {r.key for r in detect_runtimes()}
        assert detected == {"vllm"}

    def test_no_runtimes_running(self, monkeypatch):
        monkeypatch.setattr("protor.runtimes._probe", lambda runtime, timeout: False)
        assert detect_runtimes() == []

    def test_duplicate_urls_are_probed_once(self, monkeypatch):
        """llama.cpp and LocalAI can share a URL; probe it once."""
        monkeypatch.setenv("LLAMA_CPP_URL", "http://localhost:9999")
        monkeypatch.setenv("LOCALAI_URL", "http://localhost:9999")
        calls: list[str] = []
        monkeypatch.setattr(
            "protor.runtimes._probe",
            lambda runtime, timeout: calls.append(runtime.url) or False,
        )
        detect_runtimes()
        assert len(calls) == len(set(calls))


class TestProbe:
    """The probe only reports 'something is listening', never auth details."""

    @pytest.fixture(autouse=True)
    def _point_ollama_at_stub(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://x:1")

    @responses_lib.activate
    def test_healthy_endpoint_counts_as_running(self):
        responses_lib.add(responses_lib.GET, "http://x:1/api/tags", json={"models": []}, status=200)
        assert _probe(get_runtime("ollama"), 1.0) is True

    @responses_lib.activate
    def test_auth_challenge_still_counts_as_running(self):
        """A 401 proves the server is up; only the backend should judge auth."""
        responses_lib.add(responses_lib.GET, "http://x:1/api/tags", json={}, status=401)
        assert _probe(get_runtime("ollama"), 1.0) is True

    @responses_lib.activate
    def test_server_error_does_not_count(self):
        responses_lib.add(responses_lib.GET, "http://x:1/api/tags", json={}, status=503)
        assert _probe(get_runtime("ollama"), 1.0) is False

    @responses_lib.activate
    def test_connection_refused_is_not_running(self):
        responses_lib.add(responses_lib.GET, "http://x:1/api/tags", body=ConnectionError("refused"))
        assert _probe(get_runtime("ollama"), 1.0) is False

    @responses_lib.activate
    def test_probe_hits_the_runtime_health_path(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_URL", "http://x:2")
        responses_lib.add(responses_lib.GET, "http://x:2/v1/models", json={"data": []}, status=200)
        assert _probe(get_runtime("lmstudio"), 1.0) is True
        assert str(responses_lib.calls[0].request.url) == "http://x:2/v1/models"


class TestUnavailableError:
    def test_carries_url_and_start_hint(self):
        from protor.exceptions import RuntimeUnavailableError

        err = RuntimeUnavailableError("vLLM", "http://localhost:8000", "vllm serve m")
        assert err.runtime == "vLLM"
        assert err.base_url == "http://localhost:8000"
        assert "http://localhost:8000" in str(err)
        assert "vllm serve m" in str(err)

    def test_ollama_error_is_a_runtime_error(self):
        from protor.exceptions import OllamaUnavailableError, RuntimeUnavailableError

        err = OllamaUnavailableError()
        assert isinstance(err, RuntimeUnavailableError)
        assert "ollama serve" in str(err)
