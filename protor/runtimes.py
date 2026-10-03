"""
protor.runtimes
~~~~~~~~~~~~~~~
Registry of local model runtimes that protor can analyse scraped pages with.

Almost every modern local runtime — llama.cpp's ``llama-server``, LM Studio,
vLLM, LocalAI, Jan, GPT4All, KoboldCpp, llamafile, TabbyAPI, SGLang, Xinference,
LiteLLM and the rest — exposes the *same* OpenAI-compatible
``/v1/chat/completions`` + ``/v1/models`` API. So there is exactly one
OpenAI-compatible backend in :mod:`protor.llm_backends`, and everything here is
configuration: where it listens, which paths it serves, how to probe it, and how
to start it.

Ollama is the exception: it has its own native, newline-delimited API and keeps
a dedicated backend. Docker Model Runner is the other odd one out — it is
OpenAI-compatible, but hangs its API off ``/engines/v1`` rather than ``/v1``,
which is why the paths are declared per runtime instead of assumed.

Public API
----------
    RUNTIMES, Runtime, get_runtime, runtime_names
    resolve_base_url, resolve_api_key
    detect_runtimes,
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    "RUNTIMES",
    "Runtime",
    "detect_runtimes",
    "get_runtime",
    "resolve_api_key",
    "resolve_base_url",
    "runtime_names",
    "shared_url_runtimes",
]


@dataclass(frozen=True)
class Runtime:
    """Static description of a local model runtime."""

    key: str
    label: str
    #: ``"ollama"`` for the native API, ``"openai"`` for OpenAI-compatible.
    api: str
    default_url: str
    #: Cheap GET used to decide whether this runtime is up.
    health_path: str
    #: GET returning a model list.
    models_path: str
    #: POST that streams a chat completion.
    chat_path: str
    #: Environment variable that overrides ``default_url``.
    env_url: str | None
    #: Environment variable holding an API token, if the runtime uses one.
    env_key: str | None
    #: How to start it, shown when it is not running.
    start_hint: str
    docs: str

    @property
    def url(self) -> str:
        """Base URL after applying any environment override."""
        return resolve_base_url(self.key)


#: Paths shared by the many OpenAI-compatible servers.
_V1_MODELS = "/v1/models"
_V1_CHAT = "/v1/chat/completions"

#: Runtimes in probe order. Ollama leads because it is the historical default;
#: the llama.cpp family follows it, and several of those deliberately share
#: port 8080 — any OpenAI-compatible server there answers for all of them.
RUNTIMES: dict[str, Runtime] = {
    "ollama": Runtime(
        key="ollama",
        label="Ollama",
        api="ollama",
        default_url="http://localhost:11434",
        health_path="/api/tags",
        models_path="/api/tags",
        chat_path="/api/generate",
        env_url="OLLAMA_HOST",
        env_key=None,
        start_hint="ollama serve",
        docs="https://ollama.ai",
    ),
    "lmstudio": Runtime(
        key="lmstudio",
        label="LM Studio",
        api="openai",
        default_url="http://localhost:1234",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="LMSTUDIO_URL",
        env_key="LMSTUDIO_API_KEY",
        start_hint="lms server start",
        docs="https://lmstudio.ai/docs/developer/core/server",
    ),
    "llamacpp": Runtime(
        key="llamacpp",
        label="llama.cpp",
        api="openai",
        default_url="http://localhost:8080",
        # /v1/models is present even while a model is loading, unlike /health.
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="LLAMA_CPP_URL",
        env_key="LLAMA_CPP_API_KEY",
        start_hint="llama-server -m model.gguf",
        docs="https://github.com/ggml-org/llama.cpp/tree/master/tools/server",
    ),
    "vllm": Runtime(
        key="vllm",
        label="vLLM",
        api="openai",
        default_url="http://localhost:8000",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="VLLM_URL",
        env_key="VLLM_API_KEY",
        start_hint="vllm serve <model>",
        docs="https://docs.vllm.ai/en/latest/serving/online_serving/",
    ),
    "localai": Runtime(
        key="localai",
        label="LocalAI",
        api="openai",
        default_url="http://localhost:8081",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="LOCALAI_URL",
        env_key="LOCALAI_API_KEY",
        start_hint="localai run",
        docs="https://localai.io/features/",
    ),
    "jan": Runtime(
        key="jan",
        label="Jan",
        api="openai",
        default_url="http://localhost:1337",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="JAN_URL",
        env_key="JAN_API_KEY",
        start_hint="start the Jan app and enable its local server",
        docs="https://jan.ai",
    ),
    "llamafile": Runtime(
        key="llamafile",
        label="llamafile",
        api="openai",
        default_url="http://localhost:8080",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="LLAMAFILE_URL",
        env_key="LLAMAFILE_API_KEY",
        start_hint="llamafile -m model.gguf --server",
        docs="https://github.com/Mozilla-Ocho/llamafile",
    ),
    "tabbyapi": Runtime(
        key="tabbyapi",
        label="TabbyAPI",
        api="openai",
        default_url="http://localhost:8080",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="TABBY_API_URL",
        env_key="TABBY_API_KEY",
        start_hint="tabby serve --model <model-repo>",
        docs="https://github.com/theroyallab/tabbyAPI",
    ),
    "cortex": Runtime(
        key="cortex",
        label="Cortex.cpp",
        api="openai",
        default_url="http://localhost:8080",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="CORTEX_URL",
        env_key="CORTEX_API_KEY",
        start_hint="cortex server --config config.yaml",
        docs="https://github.com/cortexcpp/cortex",
    ),
    "gpt4all": Runtime(
        key="gpt4all",
        label="GPT4All",
        api="openai",
        default_url="http://localhost:4891",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="GPT4ALL_URL",
        env_key="GPT4ALL_API_KEY",
        start_hint="GPT4All → Settings → Application → Enable Local API Server",
        docs="https://docs.gpt4all.io/gpt4all-api-server/home.html",
    ),
    "koboldcpp": Runtime(
        key="koboldcpp",
        label="KoboldCpp",
        api="openai",
        default_url="http://localhost:5001",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="KOBOLDCPP_URL",
        env_key="KOBOLDCPP_API_KEY",
        start_hint="koboldcpp --model model.gguf --port 5001",
        docs="https://github.com/LostRuins/koboldcpp",
    ),
    "oobabooga": Runtime(
        key="oobabooga",
        label="text-generation-webui",
        api="openai",
        default_url="http://localhost:5000",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="OOBABOOGA_URL",
        env_key="OOBABOOGA_API_KEY",
        start_hint="python server.py --api --api-port 5000",
        docs="https://github.com/oobabooga/text-generation-webui",
    ),
    "sglang": Runtime(
        key="sglang",
        label="SGLang",
        api="openai",
        default_url="http://localhost:30000",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="SGLANG_URL",
        env_key="SGLANG_API_KEY",
        start_hint="python -m sglang.launch_server --model-path <model>",
        docs="https://docs.sglang.ai",
    ),
    "xinference": Runtime(
        key="xinference",
        label="Xinference",
        api="openai",
        default_url="http://localhost:9997",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="XINFERENCE_URL",
        env_key="XINFERENCE_API_KEY",
        start_hint="xinference-local",
        docs="https://inference.readthedocs.io",
    ),
    "litellm": Runtime(
        key="litellm",
        label="LiteLLM proxy",
        api="openai",
        default_url="http://localhost:4000",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="LITELLM_URL",
        env_key="LITELLM_API_KEY",
        start_hint="litellm --config config.yaml",
        docs="https://docs.litellm.ai/docs/proxy/quick_start",
    ),
    "anythingllm": Runtime(
        key="anythingllm",
        label="AnythingLLM",
        api="openai",
        default_url="http://localhost:3001",
        health_path=_V1_MODELS,
        models_path=_V1_MODELS,
        chat_path=_V1_CHAT,
        env_url="ANYTHINGLLM_URL",
        env_key="ANYTHINGLLM_API_KEY",
        start_hint="docker run -p 3001:3001 anythingllm",
        docs="https://docs.anythingllm.com/developer/open-api-compatibility",
    ),
    "docker": Runtime(
        key="docker",
        label="Docker Model Runner",
        # OpenAI-compatible, but the API hangs off /engines/v1, not /v1.
        api="openai",
        default_url="http://localhost:12434",
        health_path="/engines/v1/models",
        models_path="/engines/v1/models",
        chat_path="/engines/v1/chat/completions",
        env_url="DOCKER_MODEL_RUNNER_URL",
        env_key=None,
        start_hint="docker desktop enable model-runner --tcp=12434",
        docs="https://docs.docker.com/ai/model-runner/api-reference",
    ),
}

#: Friendly spellings accepted for ``--backend``.
_ALIASES = {
    # llama.cpp and friends
    "llama.cpp": "llamacpp",
    "llama-cpp": "llamacpp",
    "llamacpp": "llamacpp",
    "llama": "llamacpp",
    "llamafile": "llamafile",
    "llama-file": "llamafile",
    "tabbyapi": "tabbyapi",
    "tabby": "tabbyapi",
    "tabby-api": "tabbyapi",
    "cortex": "cortex",
    "cortex.cpp": "cortex",
    "cortex-cpp": "cortex",
    # desktop apps
    "lm-studio": "lmstudio",
    "lmstudio": "lmstudio",
    "lm_studio": "lmstudio",
    "jan": "jan",
    "gpt4all": "gpt4all",
    "gpt-4all": "gpt4all",
    "gpt_4all": "gpt4all",
    "nomic": "gpt4all",
    "anythingllm": "anythingllm",
    "anything-llm": "anythingllm",
    "anything_llm": "anythingllm",
    # servers and frameworks
    "vllm": "vllm",
    "v-llm": "vllm",
    "local-ai": "localai",
    "localai": "localai",
    "koboldcpp": "koboldcpp",
    "kobold-cpp": "koboldcpp",
    "kobold_cpp": "koboldcpp",
    "kobold": "koboldcpp",
    "oobabooga": "oobabooga",
    "ooba": "oobabooga",
    "oobabooga-webui": "oobabooga",
    "textgen": "oobabooga",
    "text-generation-webui": "oobabooga",
    "text-generation": "oobabooga",
    "webui": "oobabooga",
    "sglang": "sglang",
    "s-glang": "sglang",
    "xinference": "xinference",
    "x-inference": "xinference",
    "litellm": "litellm",
    "lite-llm": "litellm",
    "litellm-proxy": "litellm",
    "ollama": "ollama",
    "docker": "docker",
    "docker-model-runner": "docker",
    "docker-modelrunner": "docker",
    "model-runner": "docker",
    "modelrunner": "docker",
    "dmr": "docker",
}


def runtime_names() -> tuple[str, ...]:
    """Canonical runtime keys, in probe order."""
    return tuple(RUNTIMES)


def shared_url_runtimes() -> list[tuple[str, list[Runtime]]]:
    """
    Group runtimes that resolve to the same URL.

    The llama.cpp family all default to port 8080, so one server answers for
    several entries. Surfacing the grouping keeps ``protor runtimes`` from
    looking wrong — or, worse, like protor cannot tell running from stopped.
    """
    by_url: dict[str, list[Runtime]] = {}
    for runtime in RUNTIMES.values():
        by_url.setdefault(runtime.url, []).append(runtime)
    return [(url, rs) for url, rs in by_url.items() if len(rs) > 1]


def get_runtime(key: str) -> Runtime:
    """
    Look up a runtime by key or common alias.

    Raises ValueError with the accepted values if unknown.
    """
    normalized = _ALIASES.get(key.strip().lower())
    if normalized is None:
        raise ValueError(f"Unknown runtime: {key!r}. Choose from: {', '.join(runtime_names())}")
    return RUNTIMES[normalized]


def resolve_base_url(key: str, override: str | None = None) -> str:
    """
    Resolve a runtime's base URL.

    Precedence: explicit *override*, then the runtime's environment variable,
    then its documented default. Trailing slashes are stripped so path
    concatenation stays predictable.
    """
    runtime = get_runtime(key)
    url = override or (os.environ.get(runtime.env_url) if runtime.env_url else None)
    return (url or runtime.default_url).rstrip("/")


def resolve_api_key(key: str, override: str | None = None) -> str | None:
    """
    Resolve a runtime's API token.

    Local servers usually run without auth, so this returns None by default and
    only sends a header when a token is actually configured.
    """
    if override:
        return override
    runtime = get_runtime(key)
    if runtime.env_key:
        return os.environ.get(runtime.env_key) or None
    return None


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _probe(runtime: Runtime, timeout: float) -> bool:
    """
    Return True if *runtime* answers its health path at its resolved URL.

    Falls back to the bare base URL when the documented path is missing (404) or
    unreachable: a runtime may serve its API under a prefix other than the one
    documented here, and "something is listening" is still the question being
    asked.
    """
    import requests

    try:
        resp = requests.get(
            f"{runtime.url}{runtime.health_path}", headers=_headers(None), timeout=timeout
        )
        status = resp.status_code
    except Exception:
        status = None

    # Any 2xx/3xx/4xx means *something* is listening. Only a connection error
    # means "not running" — a 401 still proves the server is there, and the
    # backend's own check will surface the auth problem.
    if status is not None and status < 500:
        return True
    # A 5xx is a real answer from a broken endpoint, not a working runtime.
    if status is not None:
        return False

    try:
        return requests.get(runtime.url, headers=_headers(None), timeout=timeout).status_code < 500
    except Exception:
        return False


def detect_runtimes(timeout: float = 1.0) -> list[Runtime]:
    """
    Probe every registered runtime and return those that are running.

    Runtimes can share a URL — llama.cpp, llamafile, TabbyAPI and Cortex.cpp all
    default to port 8080, and an OpenAI-compatible server there answers for all
    of them — so each distinct (URL, path) pair is only probed once, then every
    runtime on a URL that answers is reported as up.
    """
    results: dict[tuple[str, str], bool] = {}
    found: list[Runtime] = []
    for runtime in RUNTIMES.values():
        target = (runtime.url, runtime.health_path)
        if target not in results:
            results[target] = _probe(runtime, timeout)
        if results[target]:
            found.append(runtime)
    return found

