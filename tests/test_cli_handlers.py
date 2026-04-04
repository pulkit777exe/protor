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
    @patch("protor.cli.analyze_with_ollama")
    def test_analyze_with_default_output(self, mock_analyze, tmp_path):
        index_file = tmp_path / "sites_index.json"
        index_file.write_text(json.dumps([]))

        args = MagicMock()
        args.file = str(index_file)
        args.output = "analysis"
        args.model = "llama3"
        args.focus = "general"
        args.prompt = None
        args.prompt_file = None
        args.format = "markdown"

        _cmd_analyze(args)
        mock_analyze.assert_called_once()

    @patch("protor.cli.analyze_with_ollama")
    def test_analyze_with_custom_output(self, mock_analyze, tmp_path):
        index_file = tmp_path / "sites_index.json"
        index_file.write_text(json.dumps([]))

        out_dir = tmp_path / "custom_output"
        args = MagicMock()
        args.file = str(index_file)
        args.output = str(out_dir)
        args.model = "mistral"
        args.focus = "technical"
        args.prompt = "Custom prompt"
        args.prompt_file = None
        args.format = "html"

        _cmd_analyze(args)
        mock_analyze.assert_called_once()

    @patch("protor.cli.analyze_with_ollama")
    def test_analyze_with_prompt_file(self, mock_analyze, tmp_path):
        index_file = tmp_path / "sites_index.json"
        index_file.write_text(json.dumps([]))

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Analyze this site")

        args = MagicMock()
        args.file = str(index_file)
        args.output = "analysis"
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
    @patch("protor.cli.console")
    @patch("protor.updater._is_editable_install")
    def test_editable_install_warning(self, mock_editable, mock_console):
        mock_editable.return_value = True
        args = MagicMock()
        args.check = False
        args.yes = False

        _cmd_update(args)
        assert mock_console.print.called

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

    @patch("protor.cli.console")
    @patch("protor.updater._is_editable_install")
    @patch("protor.cli.check_for_update")
    def test_check_network_failure(self, mock_check, mock_editable, mock_console):
        mock_editable.return_value = False
        mock_check.return_value = None
        args = MagicMock()
        args.check = True
        args.yes = False

        _cmd_update(args)
        assert mock_console.print.called


class TestCLIErrorHandling:
    @patch("protor.cli.console")
    def test_keyboard_interrupt(self, mock_console):
        with patch("protor.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.func.side_effect = KeyboardInterrupt
            mock_parser.return_value.parse_args.return_value = mock_args

            with pytest.raises(SystemExit) as exc_info:
                cli()
            assert exc_info.value.code == 130

    @patch("protor.cli.console")
    def test_value_error_shows_hint(self, mock_console):
        with patch("protor.cli._build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.func.side_effect = ValueError("Invalid URL")
            mock_parser.return_value.parse_args.return_value = mock_args

            with pytest.raises(SystemExit):
                cli()
            assert mock_console.print.called
