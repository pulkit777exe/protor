"""
protor.updater
~~~~~~~~~~~~~~
Check for updates and upgrade protor via PyPI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
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


class UpdateInfo(TypedDict):
    """
    What a successful version comparison found.

    A :class:`~typing.TypedDict` rather than a dataclass because callers index
    the result by key (``result["update_available"]``), and rather than a bare
    ``dict[str, str | bool]`` because the three values have three different
    types — a union would make every read a narrowing site.
    """

    current: str
    latest: str
    update_available: bool


def check_for_update() -> UpdateInfo | None:
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
    """
    True when protor is running from a source checkout rather than site-packages.

    The previous version compared paths, which cannot work. ``updater.py`` and
    ``__init__.py`` always share a parent directory, so "the package's grandparent is
    an ancestor of the package" is true for a plain ``pip install protor`` exactly as
    it is for ``pip install -e .`` — every wheel install reported itself as a dev
    tree, so ``protor update`` refused to install anything and told the user to run
    ``git pull`` in a directory that has no git.

    What actually distinguishes the two is the checkout itself: a source tree has a
    ``pyproject.toml`` (and usually a ``.git``) directly above the package, and an
    installed wheel has neither. The residual risk is protor vendored inside some
    larger project that has its own ``pyproject.toml``, which would read as a
    checkout and cost a refused update with a correct explanation.
    """
    try:
        import protor

        package_dir = Path(protor.__file__).resolve().parent
        return _looks_like_a_checkout(package_dir)
    except (ImportError, AttributeError):
        return False


def _looks_like_a_checkout(package_dir: Path) -> bool:
    """
    Whether *package_dir* is a package inside a source checkout.

    Split out from :func:`_is_editable_install` so it can be tested against a
    constructed `site-packages` layout. In-process, ``updater.py`` always sits in
    this repository, so the old path predicate's "project root" is the checkout no
    matter what ``protor.__file__`` is faked to be — which is exactly why the mocked
    test for it passed against the bug, and why the wheel case needs a real path to
    be reachable at all.
    """
    project_root = package_dir.parent
    return (project_root / "pyproject.toml").exists() or (project_root / ".git").exists()


#: How long to let pip run before giving up on it. Separate from
#: ``UPDATE_TIMEOUT``, which is the PyPI *metadata* lookup and is deliberately
#: short — a version check should never be the thing that hangs.
INSTALL_TIMEOUT = 300


@dataclass(frozen=True)
class UpdateOutcome:
    """
    What happened, and what to tell the user about it.

    A bare bool was not enough. pip's own output was captured and then discarded,
    so a user watched "Updating protor to v2.10.0..." for up to two minutes with no
    sign of life, and on failure was told only "Update failed. Try: pip install
    --upgrade protor" — never *why*, which is the only part that differs between a
    permissions failure, a yanked release and a proxy that cannot reach PyPI.
    """

    ok: bool
    #: pip's stderr, already trimmed, for the failure case.
    reason: str = ""


def perform_update() -> UpdateOutcome:
    """Run ``pip install --upgrade protor``, letting pip write its own progress.

    Returns:
        An :class:`UpdateOutcome` saying whether it worked and, if not, why.
    """
    try:
        # No capture_output: pip's download and install progress is what makes the
        # wait legible, and hiding it behind a two-minute silence is its own bug.
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "protor"],
            text=True,
            timeout=INSTALL_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return UpdateOutcome(False, f"pip did not finish within {INSTALL_TIMEOUT}s")
    except (FileNotFoundError, OSError) as exc:
        return UpdateOutcome(False, str(exc))
    if result.returncode == 0:
        return UpdateOutcome(True)
    return UpdateOutcome(False, f"pip exited {result.returncode}")
