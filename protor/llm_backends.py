"""LLM backend abstraction for protor analyzer."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

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
    def stream(self, prompt: str) -> str:
        """Stream a response for the given prompt."""
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
            resp = _requests.get(f"{self._base_url}/api/tags", timeout=5)
            status: int = resp.status_code
            return status == 200
        except Exception:
            return False

    def stream(self, prompt: str) -> str:
        import json

        import requests as _requests
        from rich.console import Console

        console = Console()
        full: list[str] = []

        resp = _requests.post(
            f"{self._base_url}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=300,
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
                console.print(text, end="", style="grey85")
                full.append(text)
            if chunk.get("done"):
                break

        console.print()
        console.print()
        return "".join(full)


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

    def stream(self, prompt: str) -> str:
        import openai
        from rich.console import Console

        console = Console()
        client = openai.OpenAI(api_key=self._api_key)
        full: list[str] = []

        try:
            stream = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    console.print(delta.content, end="", style="grey85")
                    full.append(delta.content)
        except openai.AuthenticationError as exc:
            raise RuntimeError("Invalid OpenAI API key") from exc
        except openai.NotFoundError as exc:
            raise RuntimeError(f"Model '{self._model}' not available") from exc

        console.print()
        console.print()
        return "".join(full)


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

    def stream(self, prompt: str) -> str:
        import anthropic
        from rich.console import Console

        console = Console()
        client = anthropic.Anthropic(api_key=self._api_key)
        full: list[str] = []

        try:
            with client.messages.stream(
                model=self._model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    console.print(text, end="", style="grey85")
                    full.append(text)
        except anthropic.AuthenticationError as exc:
            raise RuntimeError("Invalid Anthropic API key") from exc
        except anthropic.NotFoundError as exc:
            raise RuntimeError(f"Model '{self._model}' not available") from exc

        console.print()
        console.print()
        return "".join(full)


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
