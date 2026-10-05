"""Tests for protor.updater module."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

from protor import __version__
from protor.updater import (
    PYPI_URL,
    UpdateOutcome,
    _is_editable_install,
    check_for_update,
    get_current_version,
    get_latest_version,
    perform_update,
)


class TestGetCurrentVersion:
    def test_returns_installed_version(self):
        assert get_current_version() == __version__


class TestIsEditableInstall:
    def test_current_env_is_editable(self):
        result = _is_editable_install()
        assert result is True

    def test_a_constructed_site_packages_is_not_a_checkout(self, tmp_path):
        """
        The wheel case, on real paths.

        `_is_editable_install` cannot reach it in-process — this repository *is* the
        project root, so any path predicate anchored on `updater.py`'s own location
        answers "checkout" for every faked `__file__`. That is precisely how the
        original mocked test passed against the bug, so the decision is factored into
        `_looks_like_a_checkout` and tested where the layout can actually be built.
        """
        from protor.updater import _looks_like_a_checkout

        site_packages = tmp_path / "lib" / "python3.13" / "site-packages" / "protor"
        site_packages.mkdir(parents=True)
        assert _looks_like_a_checkout(site_packages) is False

    def test_a_wheel_install_is_not_an_editable_checkout(self, tmp_path, monkeypatch):
        """
        The case that made `protor update` useless for most people.

        `updater.py` and `__init__.py` always share a parent directory, so the old
        predicate — "the package's grandparent is an ancestor of the package" — is
        true for a plain `pip install protor` exactly as it is for `pip install -e .`.
        Every wheel install reported itself as a dev tree, and `update` refused to
        install anything, telling the user to run `git pull` in a directory with no
        git in it.

        Built on the real filesystem rather than a mock: the previous version of this
        test patched `Path` and set `parents = []`, which forced the answer and so
        passed against the bug it was written for.
        """
        import protor

        site_packages = tmp_path / "lib" / "python3.13" / "site-packages"
        (site_packages / "protor").mkdir(parents=True)
        assert not (site_packages / "pyproject.toml").exists()
        monkeypatch.setattr(protor, "__file__", str(site_packages / "protor" / "__init__.py"))

        assert _is_editable_install() is False, "a wheel install read as a dev tree"

    def test_a_source_checkout_is_detected_by_its_project_file(self, tmp_path, monkeypatch):
        import protor

        checkout = tmp_path / "protor-src"
        (checkout / "protor").mkdir(parents=True)
        (checkout / "pyproject.toml").write_text("[project]\nname = 'protor'\n")
        monkeypatch.setattr(protor, "__file__", str(checkout / "protor" / "__init__.py"))

        assert _is_editable_install() is True

    def test_a_vendored_package_without_a_project_file_is_not_a_checkout(
        self, tmp_path, monkeypatch
    ):
        """A git repository alone is enough, and nothing else is."""
        import protor

        vendored = tmp_path / "vendor" / "protor"
        vendored.mkdir(parents=True)
        monkeypatch.setattr(protor, "__file__", str(vendored / "__init__.py"))
        assert _is_editable_install() is False

        (tmp_path / "vendor" / ".git").mkdir()
        assert _is_editable_install() is True


class TestCmdUpdate:
    def test_check_flag_shows_version_info(self, capsys):
        from argparse import Namespace

        from protor.cli import _cmd_update

        args = Namespace(check=True, yes=False)

        with (
            patch("protor.updater._is_editable_install", return_value=False),
            patch("protor.cli.check_for_update") as mock_check,
        ):
            mock_check.return_value = {
                "current": "2.0.0",
                "latest": "2.1.0",
                "update_available": True,
            }
            _cmd_update(args)
            captured = capsys.readouterr()
            assert "2.0.0" in captured.out
            assert "2.1.0" in captured.out

    def test_yes_flag_skips_confirmation(self, capsys):
        from argparse import Namespace

        from protor.cli import _cmd_update

        args = Namespace(check=False, yes=True)

        with (
            patch("protor.updater._is_editable_install", return_value=False),
            patch("protor.cli.check_for_update") as mock_check,
            patch("protor.cli.perform_update", return_value=UpdateOutcome(True)),
        ):
            mock_check.return_value = {
                "current": "2.0.0",
                "latest": "2.1.0",
                "update_available": True,
            }
            _cmd_update(args)
            captured = capsys.readouterr()
            assert "updated to v2.1.0" in captured.out

    def test_no_update_available(self, capsys):
        from argparse import Namespace

        from protor.cli import _cmd_update

        args = Namespace(check=False, yes=False)

        with (
            patch("protor.updater._is_editable_install", return_value=False),
            patch("protor.cli.check_for_update") as mock_check,
        ):
            mock_check.return_value = {
                "current": "2.0.0",
                "latest": "2.0.0",
                "update_available": False,
            }
            _cmd_update(args)
            captured = capsys.readouterr()
            assert "already up to date" in captured.out.lower()


class TestGetLatestVersion:
    @patch("protor.updater.urlopen")
    def test_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"info": {"version": "3.0.0"}}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = get_latest_version()
        assert result == "3.0.0"
        mock_urlopen.assert_called_once_with(PYPI_URL, timeout=10)

    @patch("protor.updater.urlopen")
    def test_network_error_returns_none(self, mock_urlopen):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("network error")

        result = get_latest_version()
        assert result is None

    @patch("protor.updater.urlopen")
    def test_invalid_json_returns_none(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b"not json"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = get_latest_version()
        assert result is None

    @patch("protor.updater.urlopen")
    def test_missing_version_key_returns_none(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"info": {}}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = get_latest_version()
        assert result is None


class TestCheckForUpdate:
    @patch("protor.updater.get_latest_version")
    def test_update_available(self, mock_latest):
        mock_latest.return_value = "99.0.0"

        result = check_for_update()
        assert result is not None
        assert result["current"] == __version__
        assert result["latest"] == "99.0.0"
        assert result["update_available"] is True

    @patch("protor.updater.get_latest_version")
    def test_up_to_date(self, mock_latest):
        mock_latest.return_value = __version__

        result = check_for_update()
        assert result is not None
        assert result["update_available"] is False

    @patch("protor.updater.get_latest_version")
    def test_network_failure_returns_none(self, mock_latest):
        mock_latest.return_value = None

        result = check_for_update()
        assert result is None


class TestPerformUpdate:
    """
    An update says what happened, not only whether it happened.

    `capture_output=True` hid pip's download and install progress behind a
    two-minute silence, and the captured stderr — the only thing that distinguishes
    a permissions failure from a yanked release from a proxy that cannot reach
    PyPI — was thrown away, leaving "Update failed. Try: pip install --upgrade
    protor" as the whole report.
    """

    @patch("protor.updater.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        outcome = perform_update()
        assert outcome.ok is True
        assert outcome.reason == ""
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pip" in args
        assert "install" in args
        assert "--upgrade" in args
        assert "protor" in args

    @patch("protor.updater.subprocess.run")
    def test_pip_output_is_not_captured(self, mock_run):
        """The progress is what makes the wait legible; hiding it is its own bug."""
        mock_run.return_value = MagicMock(returncode=0)

        perform_update()
        assert not mock_run.call_args.kwargs.get("capture_output"), (
            "pip's progress must reach the terminal"
        )

    @patch("protor.updater.subprocess.run")
    def test_failure_non_zero_exit_says_which(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)

        outcome = perform_update()
        assert outcome.ok is False
        assert "1" in outcome.reason, outcome.reason

    @patch("protor.updater.subprocess.run")
    def test_timeout_says_how_long_it_waited(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pip", timeout=300)

        outcome = perform_update()
        assert outcome.ok is False
        assert "300" in outcome.reason, outcome.reason

    @patch("protor.updater.subprocess.run")
    def test_file_not_found_reports_the_cause(self, mock_run):
        mock_run.side_effect = FileNotFoundError("no pip here")

        outcome = perform_update()
        assert outcome.ok is False
        assert "no pip here" in outcome.reason, outcome.reason
