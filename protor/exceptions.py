"""Typed exception hierarchy for protor."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


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
    """
    Raised when a scraped-data index file cannot be used as input.

    Covers more than absence: pointing ``--index`` at a directory, or at a file
    that is not readable, produced a bare ``IsADirectoryError`` traceback rather
    than a sentence saying which path was wrong.
    """

    def __init__(self, path: str, reason: str = "") -> None:
        self.path = path
        detail = f" ({reason})" if reason else ""
        super().__init__(
            f"Cannot read data file {path!r}{detail}. "
            f"Run: protor scrape <urls>   # it writes sites_index.json"
        )


class OutputPathError(ProtorError):
    """
    Raised when ``--output`` names something that cannot be a directory.

    ``Path.mkdir(exist_ok=True)`` still fails when the path exists as a *file*,
    so ``protor scrape https://x -o notes.txt`` died with
    ``FileExistsError`` from inside the run — after the user had already waited
    for the network.
    """

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot use {path!r} as an output directory: {reason}")


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


class InvalidSelectorError(ProtorError):
    """
    Raised when a schema's CSS selector cannot be parsed.

    A malformed selector used to be swallowed per field, so ``scrape --schema``
    reported pages scraped while every extracted field was ``None`` — a failure
    presented as success. It is now raised when the schema is loaded, before any
    page is fetched, and says which field (or which base_selector) was at fault.
    """

    def __init__(self, selector: str, *, where: str = "", reason: str = "") -> None:
        self.selector = selector
        self.where = where
        self.reason = reason
        super().__init__(
            f"Invalid CSS selector {selector!r}"
            + (f" in {where}" if where else "")
            + (f": {reason}" if reason else "")
        )


@dataclass
class InvalidManifestError(ProtorError, ValueError):
    """
    Raised when a record cannot be read as a site manifest.

    Carries what was missing and, where known, the file it came from: a bare
    ``TypeError: missing 8 required positional arguments`` named neither, which
    is no help when the record arrived from JSON written minutes ago.

    Two tiers, because a partial record is not always a wrong one. A record
    short a *measurement* -- the shape a run killed mid-write leaves -- still
    names a real page, so it loads with that measurement empty. A record with no
    ``url``/``domain`` names no page at all, and only that case raises.

    Also a ``ValueError``, since it reports bad data rather than a failed
    operation, so callers already catching ``ValueError`` around parsing keep
    working.
    """

    missing_fields: Sequence[str] = ()
    source: str | Path | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        parts = []
        if self.detail:
            parts.append(self.detail)
        elif self.missing_fields:
            parts.append("not a site manifest: missing " + ", ".join(self.missing_fields))
        message = "; ".join(parts)
        where = f"{self.source}: " if self.source is not None else ""
        super().__init__(where + message)
