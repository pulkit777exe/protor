"""Tests for protor.updater module."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

from protor import __version__
from protor.updater import (
    PYPI_URL,
    check_for_update,
    get_current_version,
    get_latest_version,
    perform_update,
)


class TestGetCurrentVersion:
    def test_returns_installed_version(self):
        assert get_current_version() == __version__


class TestGetLatestVersion:
    @patch("protor.updater.urlopen")
    def test_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "info": {"version": "3.0.0"}
        }).encode()
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
    @patch("protor.updater.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        result = perform_update()
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pip" in args
        assert "install" in args
        assert "--upgrade" in args
        assert "protor" in args

    @patch("protor.updater.subprocess.run")
    def test_failure_non_zero_exit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)

        result = perform_update()
        assert result is False

    @patch("protor.updater.subprocess.run")
    def test_timeout_returns_false(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pip", timeout=120)

        result = perform_update()
        assert result is False

    @patch("protor.updater.subprocess.run")
    def test_file_not_found_returns_false(self, mock_run):
        mock_run.side_effect = FileNotFoundError()

        result = perform_update()
        assert result is False
