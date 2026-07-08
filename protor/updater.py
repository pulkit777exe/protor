"""
protor.updater
~~~~~~~~~~~~~~
Check for updates and upgrade protor via PyPI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from packaging.version import InvalidVersion, Version

from . import __version__

PYPI_URL = "https://pypi.org/pypi/protor/json"
UPDATE_TIMEOUT = 10


def get_current_version() -> str:
    """Return the currently installed protor version."""
    return __version__


def get_latest_version() -> str | None:
    """Fetch the latest protor version from PyPI."""
    try:
        with urlopen(PYPI_URL, timeout=UPDATE_TIMEOUT) as response:
            data = json.loads(response.read().decode())
            version = data.get("info", {}).get("version")
            return str(version) if version is not None else None
    except (URLError, OSError, json.JSONDecodeError, KeyError):
        return None


def check_for_update() -> dict | None:
    """Compare current and latest versions.

    Returns:
        dict with keys: current, latest, update_available
        None if the check failed (network error, etc.)
    """
    current = get_current_version()
    latest = get_latest_version()

    if latest is None:
        return None

    try:
        current_ver = Version(current)
        latest_ver = Version(latest)
        update_available = latest_ver > current_ver
    except InvalidVersion:
        update_available = current != latest

    return {
        "current": current,
        "latest": latest,
        "update_available": update_available,
    }


def _is_editable_install() -> bool:
    """Detect if protor is installed in editable/dev mode."""
    try:
        import protor

        source_path = Path(protor.__file__).resolve()
        project_root = Path(__file__).resolve().parent.parent
        return project_root in source_path.parents or source_path == project_root / "protor"
    except (ImportError, AttributeError):
        return False


def perform_update() -> bool:
    """Run pip install --upgrade protor.

    Returns:
        True if upgrade succeeded, False otherwise.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "protor"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
