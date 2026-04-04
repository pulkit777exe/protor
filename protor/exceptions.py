"""Typed exception hierarchy for protor."""


class ProtorError(Exception):
    """Base class for all protor errors."""


class FetchError(ProtorError):
    """Raised when an HTTP fetch fails."""

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        self.reason = reason
        super().__init__(f"Fetch failed for {url!r}: {reason}")


class OllamaUnavailableError(ProtorError):
    """Raised when the Ollama service cannot be reached."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url
        super().__init__(
            f"Cannot reach Ollama at {base_url}. "
            "Start it with: ollama serve"
        )


class OllamaModelNotFoundError(ProtorError):
    """Raised when the requested model is not available locally."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"Model {model!r} not found. Pull it with: ollama pull {model}"
        )


class DataFileNotFoundError(ProtorError):
    """Raised when a scraped-data index file cannot be located."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(
            f"Data file not found: {path!r}. "
            "Run: protor scrape <urls>"
        )
