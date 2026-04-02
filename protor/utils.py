"""Shared utilities: I/O helpers, filename sanitisation, paths."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


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
    """Return ~/Downloads/protor, cross-platform."""
    base = Path(os.environ.get("USERPROFILE", "~")).expanduser()
    return base / "Downloads" / "protor"


def human_bytes(n: int) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n //= 1024
    return f"{n:.0f} TB"