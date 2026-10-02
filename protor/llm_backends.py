"""LLM backend abstraction for protor analyzer."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .config import ANALYSIS_TIMEOUT, OLLAMA_CHECK_TIMEOUT

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "BACKEND_CHOICES",
    "AnthropicBackend",
    "LLMBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "create_backend",
]

BACKEND_CHOICES = ("ollama", "openai", "anthropic")


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

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...


class OllamaBackend(LLMBackend):
    """Ollama backend using local API."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self._model = model
        self._base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    @property
    def model_name(self) -> str:
        return self._model

    def check_available(self) -> bool:
        import requests as _requests

        try:
            resp = _requests.get(f"{self._base_url}/api/tags", timeout=OLLAMA_CHECK_TIMEOUT)
            status: int = resp.status_code
            return status == 200
        except Exception:
            return False

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield Ollama response chunks; raise RuntimeError if the model is missing."""
        import json

        import requests as _requests

        resp = _requests.post(
            f"{self._base_url}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=ANALYSIS_TIMEOUT,
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


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

    @property
    def model_name(self) -> str:
        return self._model

    def check_available(self) -> bool:
        import openai

        try:
            client = openai.OpenAI(api_key=self._api_key)
            client.models.list()
            return True
        except Exception:
            return False

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield OpenAI response chunks; wrap auth/model errors as RuntimeError."""
        import openai

        client = openai.OpenAI(api_key=self._api_key)

        try:
            stream = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except openai.AuthenticationError as exc:
            raise RuntimeError("Invalid OpenAI API key") from exc
        except openai.NotFoundError as exc:
            raise RuntimeError(f"Model '{self._model}' not available") from exc


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    @property
    def model_name(self) -> str:
        return self._model

    def check_available(self) -> bool:
        import requests

        try:
            resp = requests.get(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=10,
            )
            return resp.status_code in (200, 400)
        except Exception:
            return False

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield Anthropic response chunks; wrap auth/model errors as RuntimeError."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        try:
            with client.messages.stream(
                model=self._model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                yield from stream.text_stream
        except anthropic.AuthenticationError as exc:
            raise RuntimeError("Invalid Anthropic API key") from exc
        except anthropic.NotFoundError as exc:
            raise RuntimeError(f"Model '{self._model}' not available") from exc


def create_backend(backend: str, model: str, **kwargs: object) -> LLMBackend:
    """Factory function to create an LLM backend."""
    backends: dict[str, type[LLMBackend]] = {
        "ollama": OllamaBackend,
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
    }
    cls = backends.get(backend.lower())
    if cls is None:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from: {', '.join(backends)}")
    return cls(model, **kwargs)  # type: ignore[call-arg]
