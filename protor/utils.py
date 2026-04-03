"""Shared utilities: I/O helpers, filename sanitisation, paths."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def safe_filename(name: str) -> str:
    """Return a filesystem-safe version of *name*."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name).strip("_") or "unnamed"


def save_json(data: Any, path: str | Path) -> None:
    """Serialise *data* to JSON, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Load JSON from *path*, raising FileNotFoundError with a clear message."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_default_output_dir() -> Path:
    """Return a sensible default output directory, cross-platform."""
    return Path.home() / "Downloads" / "protor"


def human_bytes(n: int) -> str:
    """Human-readable byte count."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def validate_url(url: str) -> str:
    """Validate and normalise *url*.

    Raises ValueError if the URL is malformed or missing a scheme.
    Returns the normalised URL string.
    """
    if not url or not isinstance(url, str):
        raise ValueError(f"URL must be a non-empty string, got: {url!r}")

    parsed = urlparse(url)
    if not parsed.scheme:
        raise ValueError(f"URL must include a scheme (http/https): {url!r}")
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL scheme must be http or https, got: {parsed.scheme!r}")
    if not parsed.netloc:
        raise ValueError(f"URL must include a hostname: {url!r}")

    return url
