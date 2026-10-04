"""Tests for protor.cli command handlers and error paths."""

import json
from unittest.mock import MagicMock, patch

import pytest

from protor.cli import (
    _cmd_analyze,
    _cmd_crawl,
    _cmd_update,
    _cmd_version,
    _load_index,
    cli,
)
from protor.exceptions import DataFileNotFoundError


class TestLoadIndex:
    def test_load_existing_file(self, tmp_path):
        data = [{"url": "https://example.com"}]
        f = tmp_path / "index.json"
        f.write_text(json.dumps(data))
        result = _load_index(str(f))
        assert result == data

    def test_load_missing_file_raises(self):
        with pytest.raises(DataFileNotFoundError):
            _load_index("/nonexistent/path.json")


class TestCmdAnalyze:
    @patch("protor.cli.analyze_with_runtime")
    def test_analyze_with_default_output(self, mock_analyze, tmp_path):
        index_file = tmp_path / "sites_index.json"
        index_file.write_text(json.dumps([]))

        args = MagicMock()
        args.file = str(index_file)
        args.output = "analysis"
        args.backend = "ollama"
        args.base_url = None
        args.api_key = None
        args.model = "llama3"
        args.focus = "general"
        args.prompt = None
        args.prompt_file = None
        args.format = "markdown"

        _cmd_analyze(args)
        mock_analyze.assert_called_once()

    @patch("protor.cli.analyze_with_runtime")
    def test_analyze_passes_the_selected_runtime(self, mock_analyze, tmp_path):
        index_file = tmp_path / "sites_index.json"
        index_file.write_text(json.dumps([]))

        args = MagicMock()
        args.file = str(index_file)
        args.output = "analysis"
        args.backend = "lmstudio"
        args.base_url = "http://gpu:1234"
        args.api_key = "tok"
        args.model = "granite"
        args.focus = "general"
        args.prompt = None
        args.prompt_file = None
        args.format = "markdown"

        _cmd_analyze(args)

        assert mock_analyze.call_args.args[1] == "lmstudio"
        assert mock_analyze.call_args.kwargs["base_url"] == "http://gpu:1234"
        assert mock_analyze.call_args.kwargs["api_key"] == "tok"

    @patch("protor.cli.analyze_with_runtime")
    def test_analyze_with_custom_output(self, mock_analyze, tmp_path):
        index_file = tmp_path / "sites_index.json"
        index_file.write_text(json.dumps([]))

        out_dir = tmp_path / "custom_output"
        args = MagicMock()
        args.file = str(index_file)
        args.output = str(out_dir)
        args.backend = "ollama"
        args.base_url = None
        args.api_key = None
        args.model = "mistral"
        args.focus = "technical"
        args.prompt = "Custom prompt"
        args.prompt_file = None
        args.format = "html"

        _cmd_analyze(args)
        mock_analyze.assert_called_once()

    @patch("protor.cli.analyze_with_runtime")
    def test_analyze_with_prompt_file(self, mock_analyze, tmp_path):
        index_file = tmp_path / "sites_index.json"
        index_file.write_text(json.dumps([]))

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Analyze this site")

        args = MagicMock()
        args.file = str(index_file)
        args.output = "analysis"
        args.backend = "ollama"
        args.base_url = None
        args.api_key = None
        args.model = "llama3"
        args.focus = "general"
        args.prompt = None
        args.prompt_file = str(prompt_file)
        args.format = "markdown"

        _cmd_analyze(args)
        mock_analyze.assert_called_once()


class TestCmdCrawl:
    @patch("protor.cli.Crawler")
    def test_crawl_command(self, mock_crawler_cls):
        mock_crawler = MagicMock()
        mock_crawler_cls.return_value = mock_crawler

        args = MagicMock()
        args.url = "https://example.com"
        args.max_pages = 5
        args.output = "/tmp/crawl_test"

        _cmd_crawl(args)
        mock_crawler_cls.assert_called_once()
        mock_crawler.crawl.assert_called_once()


class TestCmdVersion:
    @patch("protor.cli.console")
    def test_version_command(self, mock_console):
        args = MagicMock()
        _cmd_version(args)
        assert mock_console.print.called


class TestCmdUpdate:
    # These patch the *stderr* console: what they assert on is a diagnostic, and
    # stdout carries the report of what the command did.
    @patch("protor.cli.err_console")
    @patch("protor.cli.console")
    @patch("protor.updater._is_editable_install")
    @patch("protor.cli.check_for_update")
    @patch("protor.cli.perform_update")
    def test_editable_install_is_not_installed_over(
        self, mock_perform, mock_check, mock_editable, mock_console, mock_err_console
    ):
        """The checkout wins: pip must not replace a dev tree with a release."""
        mock_editable.return_value = True
        mock_check.return_value = {
            "current": "2.4.0",
            "latest": "2.5.0",
            "update_available": True,
        }
        args = MagicMock()
        args.check = False
        args.yes = False

        _cmd_update(args)
        assert not mock_perform.called, "an editable checkout was overwritten"
        printed = " ".join(str(c) for c in mock_err_console.print.call_args_list)
        assert "Editable install" in printed

    @patch("protor.cli.console")
    @patch("protor.updater._is_editable_install")
    @patch("protor.cli.check_for_update")
    def test_check_still_reports_on_an_editable_install(
        self, mock_check, mock_editable, mock_console
    ):
        """
        `--check` is documented as "only check for updates, don't install".

        It returned before the check, so in a dev tree — exactly where somebody
        runs it to see whether they are behind — it printed a refusal to install
        and said nothing about whether an update existed.
        """
        mock_editable.return_value = True
        mock_check.return_value = {
            "current": "2.4.0",
            "latest": "2.5.0",
            "update_available": True,
        }
        args = MagicMock()
        args.check = True
        args.yes = False

        _cmd_update(args)
        assert mock_check.called, "--check never asked PyPI"
        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "2.5.0" in printed, f"the latest version was not reported: {printed}"

    @patch("protor.cli.console")
    @patch("protor.updater._is_editable_install")
    @patch("protor.cli.check_for_update")
    def test_no_update_available(self, mock_check, mock_editable, mock_console):
        mock_editable.return_value = False
        mock_check.return_value = {
            "current": "2.4.0",
            "latest": "2.4.0",
            "update_available": False,
        }
        args = MagicMock()
        args.check = True
        args.yes = False

        _cmd_update(args)
        assert mock_console.print.called

    @patch("protor.cli.console")
    @patch("protor.updater._is_editable_install")
    @patch("protor.cli.check_for_update")
    def test_update_available_check_flag(self, mock_check, mock_editable, mock_console):
        mock_editable.return_value = False
        mock_check.return_value = {
            "current": "2.3.0",
            "latest": "2.4.0",
            "update_available": True,
        }
        args = MagicMock()
        args.check = True
        args.yes = False

        _cmd_update(args)
        assert mock_console.print.called

    @patch("protor.cli.err_console")
    @patch("protor.updater._is_editable_install")
    @patch("protor.cli.check_for_update")
    def test_check_network_failure(self, mock_check, mock_editable, mock_err_console):
        """A failed check exits non-zero: `--check` in a script must not read as up to date."""
        mock_editable.return_value = False
        mock_check.return_value = None
        args = MagicMock()
        args.check = True
        args.yes = False

        with pytest.raises(SystemExit) as excinfo:
            _cmd_update(args)
        assert excinfo.value.code == 1
        assert mock_err_console.print.called


class TestCLIErrorHandling:
    # A failure is not a result: everything here lands on stderr, so a script that
    # captures stdout gets nothing rather than a traceback's worth of noise.
    @patch("protor.cli.err_console")
    def test_keyboard_interrupt(self, mock_err_console):
        with patch("protor.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.func.side_effect = KeyboardInterrupt
            mock_parser.return_value.parse_args.return_value = mock_args

            with pytest.raises(SystemExit) as exc_info:
                cli()
            assert exc_info.value.code == 130
            assert mock_err_console.print.called

    @patch("protor.cli.err_console")
    def test_value_error_shows_hint(self, mock_err_console):
        with patch("protor.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.func.side_effect = ValueError("Invalid URL")
            mock_parser.return_value.parse_args.return_value = mock_args

            with pytest.raises(SystemExit):
                cli()
            assert mock_err_console.print.called
