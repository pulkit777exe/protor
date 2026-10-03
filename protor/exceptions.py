"""Typed exception hierarchy for protor."""


class ProtorError(RuntimeError):
    """
    Base class for all protor errors.

    Subclasses ``RuntimeError`` so that code written against the older bare
    ``RuntimeError`` raises — including every ``except RuntimeError`` and every
    ``pytest.raises(RuntimeError)`` — keeps working now that the backends raise
    typed subclasses. Nothing is caught less than before, and callers that want
    to distinguish an auth failure from an unreachable runtime can now do so.
    """


class FetchError(ProtorError):
    """Raised when an HTTP fetch fails."""

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        self.reason = reason
        super().__init__(f"Fetch failed for {url!r}: {reason}")


class ConfigurationError(ProtorError):
    """
    Raised when protor is asked to run with settings that cannot possibly work.

    The message is passed through untouched: only the raiser knows which setting
    is wrong, and this text is also what the CLI prints. Distinct from
    :class:`ValueError`, which is raised when a *caller* passes a bad argument to
    an API (an unknown runtime key, a missing API key) — by the time a
    configuration error is raised the arguments were valid, the resolved
    settings were not.
    """


class URLValidationError(ProtorError):
    """
    Raised when a URL the user supplied cannot be scraped.

    Distinct from ConfigurationError because the CLI gives each a different hint.
    One handler used to serve every ValueError from every layer, so a missing API
    key was reported with "URLs must include a scheme", sending the user to look
    at entirely the wrong thing.
    """

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        self.reason = reason
        super().__init__(reason)


class RuntimeUnavailableError(ProtorError):
    """
    Raised when a selected local model runtime cannot be reached.

    Carries the URL and a start hint so the CLI can tell the user exactly what
    to run, instead of a generic "connection failed".
    """

    def __init__(
        self,
        runtime: str,
        base_url: str = "",
        start_hint: str = "",
    ) -> None:
        self.runtime = runtime
        self.base_url = base_url
        self.start_hint = start_hint
        where = f" at {base_url}" if base_url else ""
        hint = f" Start it with: {start_hint}" if start_hint else ""
        super().__init__(f"Cannot reach {runtime}{where}.{hint}")


class OllamaUnavailableError(RuntimeUnavailableError):
    """
    Ollama-flavoured :class:`RuntimeUnavailableError`.

    Nothing raises this today: the analyzer builds the general
    ``RuntimeUnavailableError`` for every local runtime, including Ollama, and
    the runtime registry already supplies ``ollama serve`` as the start hint.
    It is kept because ``cli.cli()`` still imports it for a dedicated handler —
    delete the class and that handler together, or make something raise it.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        super().__init__("Ollama", base_url, "ollama serve")


class ModelNotFoundError(ProtorError):
    """
    Raised when a backend does not have the requested model loaded.

    Carries the model, the runtime that refused it and an actionable *hint*,
    because the remedy differs per backend: pull it into Ollama, load it in a
    local server, or rename it for a hosted API. The hint is supplied by the
    raiser, which is the only place that knows the command to suggest.

    Was a bare ``RuntimeError`` at every raise site, which meant ``cli.cli()``'s
    ``except ProtorError`` never saw these and the user got a traceback instead
    of the message already written for them.
    """

    #: How the refusal is worded. Ollama 404s on an unknown tag, so it says
    #: "not found"; a server holding many models reports one as unavailable.
    _wording = "not available"

    def __init__(self, model: str, runtime: str, hint: str = "") -> None:
        self.model = model
        self.runtime = runtime
        self.hint = hint
        where = f" on {runtime}" if runtime else ""
        tail = f" {hint}" if hint else ""
        super().__init__(f"Model {model!r} {self._wording}{where}.{tail}")


class OllamaModelNotFoundError(ModelNotFoundError):
    """
    Raised when the requested model has not been pulled into the local Ollama.

    A subclass rather than a sibling so a caller that only cares that the model
    is missing can catch :class:`ModelNotFoundError`; ``cli.cli()`` still
    special-cases this type to print the ``ollama pull`` hint.
    """

    _wording = "not found"

    def __init__(self, model: str) -> None:
        super().__init__(model, "Ollama", f"Pull it with: ollama pull {model}")


class AuthError(ProtorError):
    """
    Raised when a backend rejects the request's credentials (HTTP 401/403).

    Split out from generic HTTP failures because the fix is always the same one:
    supply a valid token for that runtime. *message* is passed through verbatim
    by the hosted backends, whose remedy names their own environment variable
    ("Invalid OpenAI API key") rather than a generic token.
    """

    def __init__(self, runtime: str, status: int | None = None, message: str = "") -> None:
        self.runtime = runtime
        self.status = status
        where = f" (HTTP {status})" if status else ""
        super().__init__(
            message or f"{runtime} rejected the request{where}. Set an API token for it."
        )


class DataFileNotFoundError(ProtorError):
    """Raised when a scraped-data index file cannot be located."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Data file not found: {path!r}. Run: protor scrape <urls>")


class ModelListUnavailableError(ProtorError):
    """
    Raised when a runtime serves chat completions but exposes no model listing.

    Distinct from a generic failure because the remedy is different: the model
    name must be passed by hand rather than discovered.

    Now a :class:`ProtorError` like every other typed error here. It used to
    inherit ``RuntimeError``, which meant the one typed error raised from
    ``llm_backends`` was invisible to ``cli.cli()``'s ``except ProtorError``.
    """

    def __init__(self, runtime: str, url: str) -> None:
        self.runtime = runtime
        self.url = url
        super().__init__(
            f"{runtime} does not expose a model list at {url}. Pass the model explicitly: --model <name>"
        )
