"""
protor.runtimes
~~~~~~~~~~~~~~~
Registry of local model runtimes that protor can analyse scraped pages with.

Almost every modern local runtime — llama.cpp's ``llama-server``, LM Studio,
vLLM, LocalAI, Jan — exposes the *same* OpenAI-compatible
``/v1/chat/completions`` + ``/v1/models`` API. So there is exactly one
OpenAI-compatible backend in :mod:`protor.llm_backends`, and everything here is
configuration: where it listens, how to probe it, and how to start it.

Ollama is the exception: it has its own native, newline-delimited API and keeps
a dedicated backend.

Public API
----------
    RUNTIMES, Runtime, get_runtime, runtime_names
    resolve_base_url, resolve_api_key
    detect_runtimes, detect_runtime
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    "RUNTIMES",
    "Runtime",
    "detect_runtime",
    "detect_runtimes",
    "get_runtime",
    "resolve_api_key",
    "resolve_base_url",
    "runtime_names",
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


RUNTIMES: dict[str, Runtime] = {
    "ollama": Runtime(
        key="ollama",
        label="Ollama",
        api="ollama",
        default_url="http://localhost:11434",
        health_path="/api/tags",
        models_path="/api/tags",
        env_url="OLLAMA_HOST",
        env_key=None,
        start_hint="ollama serve",
        docs="https://ollama.ai",
    ),
    "llamacpp": Runtime(
        key="llamacpp",
        label="llama.cpp",
        api="openai",
        default_url="http://localhost:8080",
        # /v1/models is present even while a model is loading, unlike /health.
        health_path="/v1/models",
        models_path="/v1/models",
        env_url="LLAMA_CPP_URL",
        env_key="LLAMA_CPP_API_KEY",
        start_hint="llama-server -m model.gguf",
        docs="https://github.com/ggml-org/llama.cpp/tree/master/tools/server",
    ),
    "lmstudio": Runtime(
        key="lmstudio",
        label="LM Studio",
        api="openai",
        default_url="http://localhost:1234",
        health_path="/v1/models",
        models_path="/v1/models",
        env_url="LMSTUDIO_URL",
        env_key="LMSTUDIO_API_KEY",
        start_hint="lms server start",
        docs="https://lmstudio.ai/docs/developer/core/server",
    ),
    "vllm": Runtime(
        key="vllm",
        label="vLLM",
        api="openai",
        default_url="http://localhost:8000",
        health_path="/v1/models",
        models_path="/v1/models",
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
        health_path="/v1/models",
        models_path="/v1/models",
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
        health_path="/v1/models",
        models_path="/v1/models",
        env_url="JAN_URL",
        env_key="JAN_API_KEY",
        start_hint="start the Jan app and enable its local server",
        docs="https://jan.ai",
    ),
}

#: Friendly spellings accepted for ``--backend``.
_ALIASES = {
    "llama.cpp": "llamacpp",
    "llama-cpp": "llamacpp",
    "llamacpp": "llamacpp",
    "llama": "llamacpp",
    "lm-studio": "lmstudio",
    "lmstudio": "lmstudio",
    "lm_studio": "lmstudio",
    "vllm": "vllm",
    "local-ai": "localai",
    "localai": "localai",
    "ollama": "ollama",
    "jan": "jan",
}


def runtime_names() -> tuple[str, ...]:
    """Canonical runtime keys, in probe order."""
    return tuple(RUNTIMES)


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
    """Return True if *runtime* answers its health path at its resolved URL."""
    import requests

    url = f"{runtime.url}{runtime.health_path}"
    try:
        resp = requests.get(url, headers=_headers(None), timeout=timeout)
    except Exception:
        return False
    # Any 2xx/3xx/4xx means *something* is listening. Only a connection error
    # means "not running" — a 401 still proves the server is there, and the
    # backend's own check will surface the auth problem.
    return resp.status_code < 500


def detect_runtimes(timeout: float = 1.0) -> list[Runtime]:
    """
    Probe every registered runtime and return those that are running.

    Two runtimes may share a URL (llama.cpp and LocalAI both default to
    localhost), so each distinct URL is only probed once.
    """
    found: list[Runtime] = []
    seen_urls: set[str] = set()
    for runtime in RUNTIMES.values():
        url = runtime.url
        if url in seen_urls:
            continue
        seen_urls.add(url)
        if _probe(runtime, timeout):
            found.append(runtime)
    return found


def detect_runtime(timeout: float = 1.0) -> Runtime | None:
    """
    Return the first running runtime, or None.

    Used to pick a sensible default when the user did not name one.
    """
    detected = detect_runtimes(timeout)
    return detected[0] if detected else None
