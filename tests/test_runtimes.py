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
    shared_url_runtimes,
)

#: Every major local runtime protor claims to support, with the URL each one
#: listens on by default. Ports are taken from each project's own documentation.
EXPECTED_RUNTIMES = {
    "ollama": "http://localhost:11434",
    "lmstudio": "http://localhost:1234",
    "llamacpp": "http://localhost:8080",
    "vllm": "http://localhost:8000",
    "localai": "http://localhost:8081",
    "jan": "http://localhost:1337",
    "llamafile": "http://localhost:8080",
    "tabbyapi": "http://localhost:8080",
    "cortex": "http://localhost:8080",
    "gpt4all": "http://localhost:4891",
    "koboldcpp": "http://localhost:5001",
    "oobabooga": "http://localhost:5000",
    "sglang": "http://localhost:30000",
    "xinference": "http://localhost:9997",
    "litellm": "http://localhost:4000",
    "anythingllm": "http://localhost:3001",
    "docker": "http://localhost:12434",
}


class TestRegistry:
    def test_known_runtimes_are_registered(self):
        for key in EXPECTED_RUNTIMES:
            assert key in runtime_names()

    @pytest.mark.parametrize(("key", "url"), sorted(EXPECTED_RUNTIMES.items()))
    def test_documented_default_urls(self, key, url):
        # Values taken from each project's own documentation.
        assert get_runtime(key).default_url == url

    def test_every_runtime_has_a_start_hint_and_docs(self):
        for runtime in RUNTIMES.values():
            assert runtime.start_hint
            assert runtime.docs.startswith("http")

    def test_every_runtime_declares_its_endpoints(self):
        """Endpoints are data, not assumptions — each runtime must supply them."""
        for runtime in RUNTIMES.values():
            assert runtime.health_path.startswith("/")
            assert runtime.models_path.startswith("/")
            assert runtime.chat_path.startswith("/")

    def test_openai_compatible_runtimes_use_the_v1_api(self):
        """Only Docker deviates, so anything else is a typo in the table."""
        for runtime in RUNTIMES.values():
            if runtime.api != "openai" or runtime.key == "docker":
                continue
            assert runtime.models_path == "/v1/models", runtime.key
            assert runtime.chat_path == "/v1/chat/completions", runtime.key

    def test_docker_model_runner_uses_the_engines_prefix(self):
        """It is OpenAI-compatible, but not at /v1."""
        runtime = get_runtime("docker")
        assert runtime.models_path == "/engines/v1/models"
        assert runtime.chat_path == "/engines/v1/chat/completions"
        assert runtime.default_url == "http://localhost:12434"

    def test_only_ollama_uses_its_native_api(self):
        """Everything else is OpenAI-compatible, so they share one backend."""
        native = {r.key for r in RUNTIMES.values() if r.api == "ollama"}
        compat = {r.key for r in RUNTIMES.values() if r.api == "openai"}
        assert native == {"ollama"}
        assert compat == set(EXPECTED_RUNTIMES) - {"ollama"}

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
            ("gpt-4all", "gpt4all"),
            ("GPT4All", "gpt4all"),
            ("nomic", "gpt4all"),
            ("kobold", "koboldcpp"),
            ("kobold-cpp", "koboldcpp"),
            ("ooba", "oobabooga"),
            ("text-generation-webui", "oobabooga"),
            ("webui", "oobabooga"),
            ("tabby", "tabbyapi"),
            ("llamafile", "llamafile"),
            ("cortex.cpp", "cortex"),
            ("sglang", "sglang"),
            ("xinference", "xinference"),
            ("lite-llm", "litellm"),
            ("anything-llm", "anythingllm"),
            ("model-runner", "docker"),
            ("Docker-Model-Runner", "docker"),
            ("dmr", "docker"),
        ],
    )
    def test_aliases_resolve(self, alias, expected):
        assert get_runtime(alias).key == expected

    def test_every_runtime_key_has_an_alias(self):
        """`--backend <key>` must work even if no alias entry exists for it."""
        for key in runtime_names():
            assert get_runtime(key).key == key

    def test_no_alias_points_at_a_missing_runtime(self):
        """A typo'd alias would only surface when a user typed it."""
        from protor.runtimes import _ALIASES

        for alias, target in _ALIASES.items():
            assert target in RUNTIMES, f"{alias} -> {target}"

    def test_env_var_names_are_unique(self):
        """Two runtimes sharing a variable would silently fight over the URL."""
        envs = [r.env_url for r in RUNTIMES.values() if r.env_url]
        assert len(envs) == len(set(envs))

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

    def test_runtimes_sharing_a_url_are_all_reported(self, monkeypatch):
        """
        The llama.cpp family shares port 8080; one server answers for all of it.

        Probing was deduplicated by URL, so only the first was ever reported —
        the other three read as "stopped" while a working server sat on the port.
        """
        monkeypatch.setattr(
            "protor.runtimes._probe",
            lambda runtime, timeout: runtime.url == "http://localhost:8080",
        )
        detected = {r.key for r in detect_runtimes()}
        assert {"llamacpp", "llamafile", "tabbyapi", "cortex"} <= detected

    def test_shared_url_group_is_reported_once(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(
            "protor.runtimes._probe",
            lambda runtime, timeout: calls.append(f"{runtime.url}{runtime.health_path}") or False,
        )
        detect_runtimes()
        assert len(calls) == len(set(calls))

    def test_env_overrides_split_a_shared_url(self, monkeypatch):
        """Pointing TabbyAPI elsewhere must stop it inheriting llama.cpp's probe."""
        monkeypatch.setenv("TABBY_API_URL", "http://localhost:7777")
        monkeypatch.setattr(
            "protor.runtimes._probe",
            lambda runtime, timeout: runtime.url == "http://localhost:7777",
        )
        detected = {r.key for r in detect_runtimes()}
        assert detected == {"tabbyapi"}


class TestSharedUrlRuntimes:
    def test_real_registry_exposes_the_llama_cpp_family_group(self):
        """The 8080 group is real, not hypothetical — guard the footnote input."""
        groups = {url: {r.key for r in rs} for url, rs in shared_url_runtimes()}
        assert groups["http://localhost:8080"] == {"llamacpp", "llamafile", "tabbyapi", "cortex"}

    def test_groups_runtimes_that_resolve_to_one_url(self, monkeypatch):
        """Two runtimes pointed at the same server must be grouped, not deduped away."""
        monkeypatch.setenv("VLLM_URL", "http://localhost:1234")
        groups = {url: {r.key for r in rs} for url, rs in shared_url_runtimes()}
        assert groups["http://localhost:1234"] == {"lmstudio", "vllm"}

    def test_separate_ports_are_not_grouped(self, monkeypatch):
        monkeypatch.setenv("VLLM_URL", "http://localhost:9999")
        grouped = {url for url, _ in shared_url_runtimes()}
        assert "http://localhost:9999" not in grouped


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

    @responses_lib.activate
    def test_probe_falls_back_to_the_base_url(self, monkeypatch):
        """A runtime serving its API elsewhere is still 'running'."""
        monkeypatch.setenv("LMSTUDIO_URL", "http://x:3")
        responses_lib.add(responses_lib.GET, "http://x:3/v1/models", status=404)
        responses_lib.add(responses_lib.GET, "http://x:3/", status=200)
        assert _probe(get_runtime("lmstudio"), 1.0) is True

    @responses_lib.activate
    def test_probe_uses_the_runtime_specific_prefix(self, monkeypatch):
        """Docker Model Runner does not serve /v1/models."""
        monkeypatch.setenv("DOCKER_MODEL_RUNNER_URL", "http://x:4")
        responses_lib.add(
            responses_lib.GET, "http://x:4/engines/v1/models", json={"data": []}, status=200
        )
        assert _probe(get_runtime("docker"), 1.0) is True
        assert str(responses_lib.calls[0].request.url) == "http://x:4/engines/v1/models"


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
