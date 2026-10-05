"""
protor.llm_backends
~~~~~~~~~~~~~~~~~~~
Backend abstraction for talking to a model.

Two local-runtime backends and two hosted ones:

* :class:`OllamaBackend` — Ollama's native newline-delimited API.
* :class:`OpenAICompatBackend` — the ``/v1/chat/completions`` API that
  llama.cpp, LM Studio, vLLM, LocalAI, Jan, GPT4All, KoboldCpp, llamafile,
  TabbyAPI, SGLang, LiteLLM and every other modern local runtime implement. One
  implementation covers all of them; :mod:`protor.runtimes` supplies the
  per-runtime URL and endpoint paths.
* :class:`OpenAIBackend` / :class:`AnthropicBackend` — hosted APIs.

Everything here uses ``requests`` rather than vendor SDKs, so pointing protor
at a local runtime never requires installing an optional cloud dependency.

Every user-facing failure — an unreachable runtime, a missing model, a rejected
token — is raised as a :class:`~protor.exceptions.ProtorError` subclass. That is
the contract ``cli.cli()`` relies on: it catches ``ProtorError`` to print a
message plus a hint, so a bare ``RuntimeError`` escaping this module reaches the
user as a traceback no matter how actionable the message that was discarded.

Public API
----------
    BACKEND_CHOICES, create_backend
    LLMBackend, ModelInfo
    OllamaBackend, OpenAICompatBackend, OpenAIBackend, AnthropicBackend
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from .config import ANALYSIS_TIMEOUT, OLLAMA_CHECK_TIMEOUT
from .exceptions import (
    AuthError,
    ConfigurationError,
    ModelListUnavailableError,
    ModelNotFoundError,
    OllamaModelNotFoundError,
    RuntimeHTTPError,
    RuntimeUnavailableError,
)
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
]

#: Backends selectable with ``--backend``.
#: ``local`` and ``compat`` are synonyms the factory has always honoured for
#: ``openai-compatible``. They were not offered by the CLI, which meant
#: ``--backend local`` died with "invalid choice" for a name the same package
#: accepts — the two lists had drifted apart and only one of them was reachable.
BACKEND_CHOICES = (
    *runtime_names(),
    "openai",
    "anthropic",
    "openai-compatible",
    "local",
    "compat",
)


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


def _check_status(resp: Any, runtime: str, url: str = "") -> None:
    """
    Turn any non-2xx response into a typed error.

    A drop-in for ``resp.raise_for_status()``, which raises a
    ``requests.exceptions.HTTPError`` — not a ``ProtorError`` — so ``cli.cli()``
    does not catch it and the user gets a traceback instead of a message. Local
    runtimes return 5xx routinely (out of memory, still loading, request shape
    rejected by the build), so this is a normal path, not a corner case.

    The statuses with a specific remedy are mapped by the callers before this
    runs, so anything reaching here has no better diagnosis than the status
    itself. The first part of the body is included because a runtime's error
    page usually names the actual problem.
    """
    if resp.status_code < 400:
        return
    detail = ""
    # getattr rather than attribute access: this is also handed the duck-typed
    # stand-ins the tests use, and a missing .text must not mask the status.
    body = getattr(resp, "text", "") or ""
    if body.strip():
        detail = " ".join(body.split())[:200]
    raise RuntimeHTTPError(
        runtime, resp.status_code, url or str(getattr(resp, "url", "") or ""), detail
    )


def _error_detail(error: Any) -> str:
    """
    Render a runtime's in-band error object as one readable line.

    Runtimes report a mid-stream failure inside the body rather than as a
    status: llama.cpp sends ``{"error": {"code": 500, "message": "..."}}`` with
    HTTP 200 when a model will not load, Ollama sends ``{"error": "..."}``, and
    Anthropic sends ``{"type": "error", "error": {"type": ..., "message": ...}}``.
    The message is the only part that names a remedy, so it is preferred over the
    type name; anything unrecognised falls back to the JSON rather than ``{}``.
    """
    if isinstance(error, dict):
        for key in ("message", "detail", "error", "type"):
            value = error.get(key)
            if isinstance(value, str) and value:
                return value
        return json.dumps(error)[:200]
    return str(error)


def _stream_error(runtime: str, error: Any, resp: Any) -> RuntimeHTTPError:
    """
    Build the typed error for a failure the runtime reported *in* a 200 body.

    The status is taken from the frame when it carries one — llama.cpp sends the
    real status in ``error.code`` — and otherwise from the response, which is
    200: the point of this error is that a successful-looking reply was not one.
    """
    status = getattr(resp, "status_code", 0)
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, int) and 100 <= code <= 599:
        status = code
    return RuntimeHTTPError(
        runtime, status, str(getattr(resp, "url", "") or ""), _error_detail(error)
    )


#: Said when the stream ended having produced nothing at all. Deliberately
#: mentions both causes rather than guessing: an empty 200 body is either a
#: model that never loaded or a prompt that would not fit, and the user can tell
#: which from the model they asked for.
_NO_TEXT = (
    "the runtime answered successfully but sent no text. The model may have "
    "failed to load, or the request may have exceeded its context window."
)


def _no_text_error(runtime: str, resp: Any) -> RuntimeHTTPError:
    """
    Build the typed error for a stream that produced no assistant text.

    Without this an empty or error-only stream became ``raw = ""``: a report file
    was written, "saved" was printed and the exit code was 0, so a runtime that
    refused the model was reported as a successful analysis of nothing.
    """
    return RuntimeHTTPError(
        runtime,
        getattr(resp, "status_code", 0),
        str(getattr(resp, "url", "") or ""),
        _NO_TEXT,
    )


def _endpoint(base_url: str, path: str) -> str:
    """
    Join a base URL with an API path without duplicating the overlapping part.

    Runtimes document their base URL inconsistently: some say ``http://host:port``
    and some (KoboldCpp, LM Studio) say ``http://host:port/v1``. Naively appending
    ``/v1/chat/completions`` to the latter yields a ``/v1/v1/...`` 404, so any path
    segment already present at the end of the base is dropped from the API path.

    The overlap is looked for as a *suffix of the base path* that opens the API
    path, longest first, so ``http://gw/api/v1`` keeps its ``/api`` prefix and
    still drops the ``/v1``. Comparing against ``"api/v1"`` instead of
    ``"/api/v1"`` silently found no overlap for anything but the whole base path,
    which sent every request to ``/api/v1/v1/...`` and reported a working runtime
    as having no such model.
    """
    base = base_url.rstrip("/")
    if not path:
        return base

    base_path = urlsplit(base).path.rstrip("/")
    if not base_path:
        return f"{base}{path}"

    # Longest suffix of the base path that also opens the API path wins. Empty
    # segments are dropped first: an absolute path splits with a leading "", and
    # keeping it would make the suffix "/api/v1" compare as "//api/v1".
    segments = [segment for segment in base_path.split("/") if segment]
    for i in range(len(segments)):
        overlap = "/" + "/".join(segments[i:])
        if path == overlap or path.startswith(f"{overlap}/"):
            return f"{base}{path[len(overlap) :]}"
    return f"{base}{path}"


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
        # resolve_base_url rather than the value verbatim: it strips the trailing
        # slash every other backend removes, so `--base-url http://host:11434/`
        # did not turn into `http://host:11434//api/tags`. It also fills in the
        # env var and the default when no override was given.
        self._base_url = resolve_base_url("ollama", base_url)
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

        url = f"{self._base_url}/api/tags"
        try:
            resp = requests.get(
                url, headers=_auth_headers(self._api_key), timeout=OLLAMA_CHECK_TIMEOUT
            )
        except Exception as exc:
            # The same diagnosis check_available() failing produces, so both paths
            # raise the same type: only the typed error carries the URL and the
            # command that starts the runtime. Bare, this escaped as an OSError —
            # which analyzer.list_runtime_models catches, prints, and then returns
            # from, reporting a failed listing as a successful command.
            raise RuntimeUnavailableError(
                self.display_name, self._base_url, self.start_hint()
            ) from exc
        if resp.status_code in (401, 403):
            raise AuthError(self.display_name, resp.status_code)
        _check_status(resp, "Ollama", url)
        return [
            ModelInfo(
                name=str(m.get("name", "?")),
                size_bytes=m.get("size"),
                modified=_format_timestamp(m.get("modified_at")),
            )
            for m in resp.json().get("models", [])
        ]

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Yield Ollama response chunks.

        Raises
        ------
        OllamaModelNotFoundError
            If the model has not been pulled. Typed because ``cli.cli()`` has a
            handler for it that prints the ``ollama pull`` hint; as a bare
            ``RuntimeError`` the user got a traceback instead.
        RuntimeUnavailableError
            If the runtime cannot be reached. ``check_available()`` can pass and
            the runtime still die before the POST arrives; the bare
            ``requests.post`` let that ``requests.exceptions.ConnectionError``
            — an ``OSError``, caught by nothing in ``cli.cli()`` — reach the user
            as a traceback instead of "Cannot reach Ollama at ...".
        RuntimeHTTPError
            If Ollama reports a failure in the stream body, or sends no text at
            all. Ollama answers ``200`` and then emits ``{"error": "..."}``, which
            skipping left as an empty analysis.
        """
        import requests

        url = f"{self._base_url}/api/generate"
        try:
            resp = requests.post(
                url,
                json={"model": self._model, "prompt": prompt, "stream": True},
                headers=_auth_headers(self._api_key),
                stream=True,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            # Same wrapping as OpenAICompatBackend.stream: the check that passed a
            # moment ago says nothing about the connection now.
            raise RuntimeUnavailableError(
                self.display_name, self._base_url, self.start_hint()
            ) from exc

        if resp.status_code == 404:
            raise OllamaModelNotFoundError(self._model)
        if resp.status_code in (401, 403):
            raise AuthError(self.display_name, resp.status_code)
        _check_status(resp, "Ollama", url)

        produced = False
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            error = chunk.get("error")
            if error:
                raise _stream_error(self.display_name, error, resp)
            text = chunk.get("response", "")
            if isinstance(text, str) and text:
                produced = True
                yield text
            if chunk.get("done"):
                break
        if not produced:
            raise _no_text_error(self.display_name, resp)


class OpenAICompatBackend(LLMBackend):
    """
    Any runtime speaking the OpenAI chat-completions API.

    Serves llama.cpp, LM Studio, vLLM, LocalAI, Jan, GPT4All, KoboldCpp,
    llamafile, TabbyAPI, SGLang, LiteLLM and the rest — they all implement the
    same ``POST /v1/chat/completions`` SSE contract, so they share this class.
    Pass *runtime* to pick one by key; otherwise supply *base_url* directly.

    The endpoints come from the runtime record rather than being hardcoded, which
    is what lets Docker Model Runner work: it is OpenAI-compatible but serves
    ``/engines/v1/...`` instead of ``/v1/...``.
    """

    #: Used when no runtime is named and the caller only gave a bare URL.
    _DEFAULT_MODELS_PATH = "/v1/models"
    _DEFAULT_CHAT_PATH = "/v1/chat/completions"

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
            record = get_runtime(runtime)
            self._base_url = resolve_base_url(record.key, base_url)
            self._api_key = resolve_api_key(record.key, api_key)
            self._label = label or record.label
            self._models_path = record.models_path
            self._chat_path = record.chat_path
        else:
            self._base_url = (base_url or "").rstrip("/")
            self._api_key = api_key
            self._label = label or "OpenAI-compatible"
            self._models_path = self._DEFAULT_MODELS_PATH
            self._chat_path = self._DEFAULT_CHAT_PATH
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
        """
        Report whether the runtime is reachable.

        A missing ``/models`` endpoint is not treated as "down": the chat
        endpoint is the one that matters, and a 404 on the listing would
        otherwise block analysis for runtimes that simply do not implement one.
        """
        import requests

        if not self._base_url:
            return False
        for path in (self._models_path, ""):
            try:
                resp = requests.get(
                    _endpoint(self._base_url, path),
                    headers=_auth_headers(self._api_key),
                    timeout=OLLAMA_CHECK_TIMEOUT,
                )
                status: int = resp.status_code
            except Exception:
                continue
            if status < 500:
                return True
        return False

    def list_models(self) -> list[ModelInfo]:
        import requests

        try:
            resp = requests.get(
                _endpoint(self._base_url, self._models_path),
                headers=_auth_headers(self._api_key),
                timeout=OLLAMA_CHECK_TIMEOUT,
            )
        except Exception as exc:
            # Same diagnosis the analyzer makes when check_available() fails, so
            # it gets the same type: only the typed error carries the URL and the
            # command that starts the runtime.
            raise RuntimeUnavailableError(self._label, self._base_url, self.start_hint()) from exc
        if resp.status_code == 404:
            raise ModelListUnavailableError(
                self._label, _endpoint(self._base_url, self._models_path)
            )
        if resp.status_code in (401, 403):
            # Before _check_status: a rejected token is a token problem, and the
            # same AuthError — carrying the remedy — already comes out of
            # stream(). Letting it fall through produced a bare HTTP error here
            # and a "Set an API token for it." there, for one mistake.
            raise AuthError(self._label, resp.status_code)
        _check_status(resp, self._label, _endpoint(self._base_url, self._models_path))
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
        """
        Yield chat-completion deltas from an OpenAI-compatible SSE stream.

        Raises
        ------
        ConfigurationError
            If no base URL was configured for this backend.
        ModelNotFoundError
            If the runtime has no such model loaded. Carries the
            ``protor models --backend <runtime>`` hint, which used to be built
            here and then thrown away into an untyped ``RuntimeError`` that the
            CLI could not render.
        AuthError
            If the runtime rejected the token.
        """
        import requests

        if not self._base_url:
            raise ConfigurationError(
                f"No base URL configured for {self._label}. Pass --base-url <url>."
            )

        try:
            resp = requests.post(
                _endpoint(self._base_url, self._chat_path),
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                },
                headers=_auth_headers(self._api_key),
                stream=True,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            # check_available() already reports an unreachable runtime in these
            # terms, and cli.cli() knows how to print it. Letting requests'
            # own ConnectionError escape bypassed both, so a runtime that died
            # between the check and the call surfaced as a traceback instead of
            # "Cannot reach llama.cpp at http://localhost:8080".
            raise RuntimeUnavailableError(self._label, self._base_url, self.start_hint()) from exc

        if resp.status_code == 404:
            raise ModelNotFoundError(
                self._model,
                self._label,
                f"List what is loaded with: protor models --backend {self._runtime_key or 'openai-compatible'}",
            )
        if resp.status_code in (401, 403):
            raise AuthError(self._label, resp.status_code)
        _check_status(resp, self._label, _endpoint(self._base_url, self._chat_path))

        yield from _yield_text(resp, self._label)


def _iter_sse_text(resp: Any, runtime: str = "the runtime") -> Iterator[str]:
    """
    Yield assistant text from an OpenAI-compatible SSE stream.

    Handles the two shapes seen in the wild: ``data: {...}`` SSE framing with a
    ``data: [DONE]`` sentinel, and bare JSON lines. Reasoning models may also
    emit ``reasoning_content``, which is skipped so only the answer is shown.

    A frame carrying ``error`` is raised rather than skipped. It has no
    ``choices``, so the frame-shaped filter below dropped it — and that frame is
    exactly what llama.cpp sends with HTTP 200 when a model fails to load or the
    prompt overflows the context window. The stream then ended with no text and
    the analysis was written as empty, so the diagnosis was lost twice.
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
        if not isinstance(chunk, dict):
            # Valid JSON of a non-object shape. Runtimes emit `data: null` as a
            # keepalive during long generations, and assuming a mapping here
            # raised AttributeError and killed the whole analysis mid-stream.
            continue

        error = chunk.get("error")
        if error:
            raise _stream_error(runtime, error, resp)

        choices = chunk.get("choices") or []
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if not isinstance(first, dict):
            continue
        delta = first.get("delta")
        if not isinstance(delta, dict):
            # `delta: null` appears on the first and last frames of a stream.
            continue
        text = delta.get("content")
        # Content is normally a string, but some proxies emit it as a list of
        # fragments; joining anything else would raise rather than skip.
        if isinstance(text, str) and text:
            yield text
        elif isinstance(text, list):
            # Some gateways send content as an array of fragments, either bare
            # strings or {"type":"text","text":...} objects.
            for part in text:
                if isinstance(part, str) and part:
                    yield part
                elif isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
                    yield part["text"]


def _yield_text(resp: Any, runtime: str) -> Iterator[str]:
    """
    Stream assistant text, turning a broken or empty stream into a typed error.

    The status check covers the response *header*, but the body is read lazily as
    it arrives: a runtime that dies mid-generation, or a proxy that drops the
    connection, raises out of ``iter_lines`` as a ``requests`` exception long
    after the request was accepted — and often after tokens have been paid for
    and shown. Untyped, that reached the user as a traceback from inside a
    generator. The partial text is not salvaged: silently ending the stream
    would report a truncated answer as a complete one.

    A stream that produced *nothing* is also a failure. An empty body, a stream
    of keepalives, and one carrying only an error frame all ended with ``raw =
    ""``: a report file was written, "saved" was printed and the exit code was
    0, so a runtime that refused the model was reported as a successful analysis
    of nothing. Truncation after real text is unaffected — that still returns
    what arrived.
    """
    import requests

    produced = False
    try:
        for text in _iter_sse_text(resp, runtime):
            produced = True
            yield text
    except requests.RequestException as exc:
        request = getattr(exc, "request", None)
        raise RuntimeHTTPError(
            runtime,
            getattr(getattr(exc, "response", None), "status_code", 0),
            getattr(request, "url", "") or "",
            f"the connection failed mid-response ({exc.__class__.__name__})",
        ) from exc
    if not produced:
        raise _no_text_error(runtime, resp)


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
        # rstrip for the reason every other backend strips: the paths below are
        # appended by f-string, so a trailing slash became `/v1//models`.
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
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

        url = f"{self._base_url}/models"
        try:
            resp = requests.get(
                url, headers=_auth_headers(self._api_key), timeout=OLLAMA_CHECK_TIMEOUT
            )
        except Exception as exc:
            # As in OpenAICompatBackend.list_models: a refused connection is an
            # unreachable runtime, not a stray OSError. No start hint — this
            # backend is not in the runtime registry.
            raise RuntimeUnavailableError(self.display_name, self._base_url) from exc
        if resp.status_code in (401, 403):
            raise AuthError(self.display_name, resp.status_code, "Invalid OpenAI API key")
        _check_status(resp, "OpenAI", url)
        return [ModelInfo(name=str(m.get("id", "?"))) for m in resp.json().get("data", [])]

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Yield OpenAI API chunks.

        Raises
        ------
        AuthError
            If the API key was rejected.
        ModelNotFoundError
            If the model is not available to this key.
        RuntimeUnavailableError
            If the endpoint cannot be reached at all.
        """
        import requests

        try:
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
        except requests.RequestException as exc:
            # check_available() answers "reachable" for a base URL that then
            # refuses, so the POST has to be wrapped too — see the same comment on
            # OpenAICompatBackend.stream.
            raise RuntimeUnavailableError(self.display_name, self._base_url) from exc
        if resp.status_code in (401, 403):
            raise AuthError("OpenAI", resp.status_code, "Invalid OpenAI API key")
        if resp.status_code == 404:
            raise ModelNotFoundError(self._model, "OpenAI")
        _check_status(resp, "OpenAI")
        yield from _yield_text(resp, "OpenAI")


#: Anthropic's endpoints have no base URL to configure, so they are named once
#: here instead of being spelled out at each of the four use sites.
_ANTHROPIC_API_URL = "https://api.anthropic.com/v1"
_ANTHROPIC_MESSAGES_URL = f"{_ANTHROPIC_API_URL}/messages"
_ANTHROPIC_MODELS_URL = f"{_ANTHROPIC_API_URL}/models"


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
                _ANTHROPIC_MESSAGES_URL,
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

        try:
            resp = requests.get(
                _ANTHROPIC_MODELS_URL,
                headers={"x-api-key": self._api_key, "anthropic-version": "2023-06-01"},
                timeout=OLLAMA_CHECK_TIMEOUT * 2,
            )
        except Exception as exc:
            # As in OpenAICompatBackend.list_models: a refused connection is an
            # unreachable API, not a stray OSError.
            raise RuntimeUnavailableError(self.display_name, _ANTHROPIC_API_URL) from exc
        if resp.status_code in (401, 403):
            raise AuthError(self.display_name, resp.status_code, "Invalid Anthropic API key")
        _check_status(resp, "Anthropic", _ANTHROPIC_MODELS_URL)
        return [
            ModelInfo(name=str(m.get("id", "?")), modified=_format_timestamp(m.get("created_at")))
            for m in resp.json().get("data", [])
        ]

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Yield Anthropic response chunks.

        Raises
        ------
        AuthError
            If the API key was rejected.
        ModelNotFoundError
            If the model is not available to this key.
        RuntimeUnavailableError
            If the API cannot be reached at all.
        RuntimeHTTPError
            If the API reports a failure in the event stream, or sends no text
            at all.
        """
        import requests

        try:
            context = requests.post(
                _ANTHROPIC_MESSAGES_URL,
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
            )
        except requests.RequestException as exc:
            # check_available() answers "reachable" and then the POST can still be
            # refused, so it needs the same wrapping as the local backends.
            raise RuntimeUnavailableError(self.display_name, _ANTHROPIC_API_URL) from exc

        with context as resp:
            if resp.status_code in (401, 403):
                raise AuthError(self.display_name, resp.status_code, "Invalid Anthropic API key")
            if resp.status_code == 404:
                raise ModelNotFoundError(self._model, self.display_name)
            _check_status(resp, "Anthropic")

            produced = False
            try:
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
                    if not isinstance(event, dict):
                        continue
                    # `type: "error"` arrives mid-stream with HTTP 200, so the
                    # type filter below dropped the one frame that named the
                    # failure and the analysis came out empty.
                    if event.get("type") == "error":
                        raise _stream_error(self.display_name, event.get("error") or event, resp)
                    # Anthropic streams named events; only content deltas matter.
                    if event.get("type") != "content_block_delta":
                        continue
                    delta = event.get("delta")
                    text = delta.get("text", "") if isinstance(delta, dict) else ""
                    if text:
                        produced = True
                        yield text
            except requests.RequestException as exc:
                # The status check above only saw the response header; a
                # connection dropped mid-generation raises out of iter_lines,
                # usually after tokens have been paid for and shown.
                raise RuntimeHTTPError(
                    "Anthropic",
                    getattr(getattr(exc, "response", None), "status_code", 0),
                    _ANTHROPIC_MESSAGES_URL,
                    f"the connection failed mid-response ({exc.__class__.__name__})",
                ) from exc
            if not produced:
                raise _no_text_error(self.display_name, resp)


# ── factory ───────────────────────────────────────────────────────────────────

_HOSTED = {"openai": OpenAIBackend, "anthropic": AnthropicBackend}


def create_backend(backend: str, model: str, **kwargs: Any) -> LLMBackend:
    """
    Create a backend by name.

    Accepts any runtime key or alias (``ollama``, ``llamacpp``, ``lmstudio``,
    ``vllm``, ``localai``, ``jan``, ``gpt4all``, ``koboldcpp``, ``docker`` …),
    the generic ``openai-compatible``, or a hosted ``openai`` / ``anthropic``.
    """
    name = backend.strip().lower()
    cls = _HOSTED.get(name)
    if cls is not None:
        backend_obj: LLMBackend = cls(model, **kwargs)
        return backend_obj
    if name in ("openai-compatible", "local", "compat"):
        compat: LLMBackend = OpenAICompatBackend(model, **kwargs)
        return compat
    # Raises ValueError listing valid runtimes.
    runtime = get_runtime(name)
    if runtime.api == "ollama":
        native: LLMBackend = OllamaBackend(model, **kwargs)
        return native
    local: LLMBackend = OpenAICompatBackend(model, runtime=runtime.key, **kwargs)
    return local
