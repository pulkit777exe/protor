"""
protor.llm_backends
~~~~~~~~~~~~~~~~~~~
Backend abstraction for talking to a model.

Two local-runtime backends and two hosted ones:

* :class:`OllamaBackend` — Ollama's native newline-delimited API.
* :class:`OpenAICompatBackend` — the ``/v1/chat/completions`` API that
  llama.cpp, LM Studio, vLLM, LocalAI and Jan all implement. One
  implementation covers all of them; :mod:`protor.runtimes` supplies the
  per-runtime URL and start hints.
* :class:`OpenAIBackend` / :class:`AnthropicBackend` — hosted APIs.

Everything here uses ``requests`` rather than vendor SDKs, so pointing protor
at a local runtime never requires installing an optional cloud dependency.

Public API
----------
    BACKEND_CHOICES, create_backend, list_models
    LLMBackend, ModelInfo
    OllamaBackend, OpenAICompatBackend, OpenAIBackend, AnthropicBackend
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import ANALYSIS_TIMEOUT, OLLAMA_CHECK_TIMEOUT
from .runtimes import get_runtime, resolve_api_key, resolve_base_url, runtime_names

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "BACKEND_CHOICES",
    "AnthropicBackend",
    "LLMBackend",
    "ModelInfo",
    "OllamaBackend",
    "OpenAIBackend",
    "OpenAICompatBackend",
    "create_backend",
    "list_models",
]

#: Backends selectable with ``--backend``.
BACKEND_CHOICES = (*runtime_names(), "openai", "anthropic", "openai-compatible")


@dataclass(frozen=True)
class ModelInfo:
    """One model advertised by a runtime."""

    name: str
    size_bytes: int | None = None
    modified: str = ""


def _format_timestamp(value: object) -> str:
    """
    Render a model timestamp as a date.

    Runtimes are inconsistent: Ollama returns an ISO string, while the
    OpenAI-compatible shape returns ``created`` as a Unix epoch integer.
    Slicing that to 10 characters produced "1750000000", so convert it.
    """
    if value in (None, ""):
        return ""
    if isinstance(value, int | float):
        from datetime import UTC, datetime

        return datetime.fromtimestamp(value, tz=UTC).strftime("%Y-%m-%d")
    return str(value)[:10]


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """Yield response chunks for the given prompt as they arrive."""
        ...

    @abstractmethod
    def check_available(self) -> bool:
        """Check if the backend is available and running."""
        ...

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Return the models this backend can serve."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable backend label for CLI output."""
        return type(self).__name__.removesuffix("Backend")

    def start_hint(self) -> str:
        """How to start this backend, for error messages."""
        return ""


# ── local runtimes ────────────────────────────────────────────────────────────


def _auth_headers(api_key: str | None) -> dict[str, str]:
    """Build request headers, adding auth only when a token is configured."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class OllamaBackend(LLMBackend):
    """Ollama, using its native newline-delimited JSON API."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: int = ANALYSIS_TIMEOUT,
    ) -> None:
        self._model = model
        self._base_url = base_url or resolve_base_url("ollama")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def display_name(self) -> str:
        return "Ollama"

    def start_hint(self) -> str:
        return get_runtime("ollama").start_hint

    def check_available(self) -> bool:
        import requests

        try:
            resp = requests.get(
                f"{self._base_url}/api/tags",
                headers=_auth_headers(self._api_key),
                timeout=OLLAMA_CHECK_TIMEOUT,
            )
            status: int = resp.status_code
            return status == 200
        except Exception:
            return False

    def list_models(self) -> list[ModelInfo]:
        import requests

        resp = requests.get(
            f"{self._base_url}/api/tags",
            headers=_auth_headers(self._api_key),
            timeout=OLLAMA_CHECK_TIMEOUT,
        )
        resp.raise_for_status()
        return [
            ModelInfo(
                name=str(m.get("name", "?")),
                size_bytes=m.get("size"),
                modified=_format_timestamp(m.get("modified_at")),
            )
            for m in resp.json().get("models", [])
        ]

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield Ollama response chunks; raise RuntimeError if the model is missing."""
        import requests

        resp = requests.post(
            f"{self._base_url}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": True},
            headers=_auth_headers(self._api_key),
            stream=True,
            timeout=self._timeout,
        )

        if resp.status_code == 404:
            raise RuntimeError(
                f"Model '{self._model}' not found. Pull with: ollama pull {self._model}"
            )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = chunk.get("response", "")
            if text:
                yield text
            if chunk.get("done"):
                break


class OpenAICompatBackend(LLMBackend):
    """
    Any runtime speaking the OpenAI chat-completions API.

    Serves llama.cpp, LM Studio, vLLM, LocalAI and Jan — they all implement the
    same ``POST /v1/chat/completions`` SSE contract, so they share this class and
    differ only in URL. Pass *runtime* to pick one by key; otherwise supply
    *base_url* directly.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        runtime: str | None = None,
        timeout: int = ANALYSIS_TIMEOUT,
        label: str | None = None,
    ) -> None:
        self._model = model
        self._runtime_key = runtime
        if runtime is not None:
            resolved_key = get_runtime(runtime).key
            self._base_url = resolve_base_url(resolved_key, base_url)
            self._api_key = resolve_api_key(resolved_key, api_key)
            self._label = label or get_runtime(resolved_key).label
        else:
            self._base_url = (base_url or "").rstrip("/")
            self._api_key = api_key
            self._label = label or "OpenAI-compatible"
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def display_name(self) -> str:
        return self._label

    def start_hint(self) -> str:
        if self._runtime_key is None:
            return ""
        return get_runtime(self._runtime_key).start_hint

    def check_available(self) -> bool:
        import requests

        if not self._base_url:
            return False
        try:
            resp = requests.get(
                f"{self._base_url}/v1/models",
                headers=_auth_headers(self._api_key),
                timeout=OLLAMA_CHECK_TIMEOUT,
            )
            status: int = resp.status_code
            return status < 500
        except Exception:
            return False

    def list_models(self) -> list[ModelInfo]:
        import requests

        resp = requests.get(
            f"{self._base_url}/v1/models",
            headers=_auth_headers(self._api_key),
            timeout=OLLAMA_CHECK_TIMEOUT,
        )
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        models: list[ModelInfo] = []
        for entry in payload.get("data", []):
            models.append(
                ModelInfo(
                    name=str(entry.get("id", "?")),
                    size_bytes=entry.get("size"),
                    modified=_format_timestamp(entry.get("created")),
                )
            )
        return models

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield chat-completion deltas from an OpenAI-compatible SSE stream."""
        import requests

        if not self._base_url:
            raise RuntimeError(f"No base URL configured for {self._label}")

        resp = requests.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            headers=_auth_headers(self._api_key),
            stream=True,
            timeout=self._timeout,
        )

        if resp.status_code == 404:
            raise RuntimeError(
                f"Model '{self._model}' not available on {self._label}. "
                f"List what is loaded with: protor models --backend {self._runtime_key or 'openai-compatible'}"
            )
        if resp.status_code in (401, 403):
            raise RuntimeError(
                f"{self._label} rejected the request (HTTP {resp.status_code}). "
                f"Set an API token for it."
            )
        resp.raise_for_status()

        yield from _iter_sse_text(resp)


def _iter_sse_text(resp: Any) -> Iterator[str]:
    """
    Yield assistant text from an OpenAI-compatible SSE stream.

    Handles the two shapes seen in the wild: ``data: {...}`` SSE framing with a
    ``data: [DONE]`` sentinel, and bare JSON lines. Reasoning models may also
    emit ``reasoning_content``, which is skipped so only the answer is shown.
    """
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line or line.startswith(":"):  # blank line or SSE comment
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        text = delta.get("content") or ""
        if text:
            yield text


# ── hosted APIs ───────────────────────────────────────────────────────────────


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        timeout: int = ANALYSIS_TIMEOUT,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self._base_url = base_url or "https://api.openai.com/v1"
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def check_available(self) -> bool:
        import requests

        try:
            resp = requests.get(
                f"{self._base_url}/models",
                headers=_auth_headers(self._api_key),
                timeout=OLLAMA_CHECK_TIMEOUT,
            )
            status: int = resp.status_code
            return status == 200
        except Exception:
            return False

    def list_models(self) -> list[ModelInfo]:
        import requests

        resp = requests.get(
            f"{self._base_url}/models",
            headers=_auth_headers(self._api_key),
            timeout=OLLAMA_CHECK_TIMEOUT,
        )
        resp.raise_for_status()
        return [ModelInfo(name=str(m.get("id", "?"))) for m in resp.json().get("data", [])]

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield OpenAI API chunks; wrap auth/model errors as RuntimeError."""
        import requests

        resp = requests.post(
            f"{self._base_url}/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            headers=_auth_headers(self._api_key),
            stream=True,
            timeout=self._timeout,
        )
        if resp.status_code in (401, 403):
            raise RuntimeError("Invalid OpenAI API key")
        if resp.status_code == 404:
            raise RuntimeError(f"Model '{self._model}' not available")
        resp.raise_for_status()
        yield from _iter_sse_text(resp)


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        timeout: int = ANALYSIS_TIMEOUT,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def display_name(self) -> str:
        return "Anthropic"

    def check_available(self) -> bool:
        import requests

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                timeout=OLLAMA_CHECK_TIMEOUT * 2,
            )
            status: int = resp.status_code
            # 200 works; 400 means the key authenticated but the probe payload
            # was rejected, which still proves the key is valid.
            return status in (200, 400)
        except Exception:
            return False

    def list_models(self) -> list[ModelInfo]:
        import requests

        resp = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": self._api_key, "anthropic-version": "2023-06-01"},
            timeout=OLLAMA_CHECK_TIMEOUT * 2,
        )
        resp.raise_for_status()
        return [
            ModelInfo(name=str(m.get("id", "?")), modified=_format_timestamp(m.get("created_at")))
            for m in resp.json().get("data", [])
        ]

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield Anthropic response chunks; wrap auth/model errors as RuntimeError."""
        import requests

        with requests.post(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": self._model,
                "max_tokens": 4096,
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            stream=True,
            timeout=self._timeout,
        ) as resp:
            if resp.status_code in (401, 403):
                raise RuntimeError("Invalid Anthropic API key")
            if resp.status_code == 404:
                raise RuntimeError(f"Model '{self._model}' not available")
            resp.raise_for_status()

            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                line = line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Anthropic streams named events; only content deltas matter.
                if event.get("type") != "content_block_delta":
                    continue
                text = (event.get("delta") or {}).get("text", "")
                if text:
                    yield text


# ── factory ───────────────────────────────────────────────────────────────────

_HOSTED = {"openai": OpenAIBackend, "anthropic": AnthropicBackend}


def create_backend(backend: str, model: str, **kwargs: Any) -> LLMBackend:
    """
    Create a backend by name.

    Accepts a runtime key (``ollama``, ``llamacpp``, ``lmstudio``, ``vllm``,
    ``localai``, ``jan`` and aliases like ``llama.cpp``), the generic
    ``openai-compatible``, or a hosted ``openai`` / ``anthropic``.
    """
    name = backend.strip().lower()
    cls = _HOSTED.get(name)
    if cls is not None:
        backend_obj: LLMBackend = cls(model, **kwargs)  # type: ignore[call-arg]
        return backend_obj
    if name in ("openai-compatible", "local", "compat"):
        compat: LLMBackend = OpenAICompatBackend(model, **kwargs)  # type: ignore[arg-type]
        return compat
    # Raises ValueError listing valid runtimes.
    runtime = get_runtime(name)
    if runtime.api == "ollama":
        native: LLMBackend = OllamaBackend(model, **kwargs)  # type: ignore[arg-type]
        return native
    local: LLMBackend = OpenAICompatBackend(  # type: ignore[arg-type]
        model, runtime=runtime.key, **kwargs
    )
    return local


def list_models(
    backend: str = "ollama",
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[ModelInfo]:
    """Return the models a backend has available."""
    return create_backend(backend, "unused", base_url=base_url, api_key=api_key).list_models()
