"""Unit tests for protor.cli module"""

import json
from unittest.mock import patch

import pytest

from protor.cli import _abort, _build_parser, _load_index
from protor.exceptions import DataFileNotFoundError
from protor.utils import get_default_output_dir


class TestBuildParser:
    def test_parser_exists(self):
        parser = _build_parser()
        assert parser is not None

    def test_scrape_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["scrape", "https://example.com"])
        assert args.command == "scrape"
        assert args.urls == ["https://example.com"]
        assert args.no_js is False
        assert args.timeout == 30
        assert args.concurrency == 6

    def test_scrape_with_options(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "scrape",
                "https://example.com",
                "--no-js",
                "--timeout",
                "60",
                "--concurrency",
                "3",
                "--output",
                "/tmp/test",
            ]
        )
        assert args.no_js is True
        assert args.timeout == 60
        assert args.concurrency == 3
        assert args.output == "/tmp/test"

    def test_analyze_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["analyze"])
        assert args.command == "analyze"
        assert args.model == "llama3"
        assert args.focus == "general"
        assert args.file == str(get_default_output_dir() / "sites_index.json")

    def test_analyze_with_options(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "analyze",
                "--model",
                "mistral",
                "--focus",
                "technical",
                "--file",
                "/tmp/data.json",
                "--output",
                "/tmp/analysis",
            ]
        )
        assert args.model == "mistral"
        assert args.focus == "technical"
        assert args.file == "/tmp/data.json"
        assert args.output == "/tmp/analysis"

    def test_run_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "https://example.com"])
        assert args.command == "run"
        assert args.urls == ["https://example.com"]
        assert args.model == "llama3"
        assert args.focus == "general"

    def test_run_with_options(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "https://example.com",
                "https://other.com",
                "--model",
                "codellama",
                "--focus",
                "seo",
                "--no-js",
                "--concurrency",
                "2",
            ]
        )
        assert args.urls == ["https://example.com", "https://other.com"]
        assert args.model == "codellama"
        assert args.focus == "seo"
        assert args.no_js is True
        assert args.concurrency == 2

    def test_crawl_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["crawl", "https://example.com"])
        assert args.command == "crawl"
        assert args.url == "https://example.com"
        assert args.max_pages == 10

    def test_crawl_with_options(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["crawl", "https://example.com", "--max-pages", "50", "--output", "/tmp/crawl"]
        )
        assert args.max_pages == 50
        assert args.output == "/tmp/crawl"

    def test_models_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["models"])
        assert args.command == "models"

    def test_version_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["version"])
        assert args.command == "version"

    def test_no_command_shows_help(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestLoadIndex:
    def test_load_existing_file(self, tmp_path):
        index_file = tmp_path / "sites_index.json"
        data = [{"domain": "example.com", "url": "https://example.com"}]
        index_file.write_text(json.dumps(data))
        result = _load_index(str(index_file))
        assert result == data

    def test_load_missing_file(self):
        with pytest.raises(DataFileNotFoundError):
            _load_index("/nonexistent/path.json")


class TestRuntimeFlags:
    """The runtime selection flags on `analyze`, `run`, and `models`."""

    def test_backend_defaults_to_ollama(self):
        for cmd in ("analyze", "models"):
            args = _build_parser().parse_args([cmd])
            assert args.backend == "ollama"

    @pytest.mark.parametrize("key", ["ollama", "llamacpp", "lmstudio", "vllm", "localai", "jan"])
    def test_every_runtime_is_selectable(self, key):
        args = _build_parser().parse_args(["analyze", "--backend", key])
        assert args.backend == key

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("llama.cpp", "llamacpp"),
            ("llama-cpp", "llamacpp"),
            ("LM-Studio", "lmstudio"),
            ("vLLM", "vllm"),
            ("local-ai", "localai"),
        ],
    )
    def test_friendly_aliases_are_normalised(self, alias, canonical):
        """The factory accepts these, so argparse must not reject them."""
        args = _build_parser().parse_args(["analyze", "--backend", alias])
        assert args.backend == canonical

    def test_hosted_backends_are_selectable(self):
        for key in ("openai", "anthropic", "openai-compatible"):
            assert _build_parser().parse_args(["analyze", "-b", key]).backend == key

    def test_base_url_and_api_key_pass_through(self):
        args = _build_parser().parse_args(
            ["analyze", "--base-url", "http://gpu:8080", "--api-key", "tok"]
        )
        assert args.base_url == "http://gpu:8080"
        assert args.api_key == "tok"

    def test_run_accepts_runtime_flags(self):
        args = _build_parser().parse_args(
            ["run", "https://example.com", "--backend", "lmstudio", "--model", "granite"]
        )
        assert args.backend == "lmstudio"
        assert args.model == "granite"

    def test_models_accepts_runtime_flags(self):
        args = _build_parser().parse_args(["models", "-b", "vllm", "--base-url", "http://x:1"])
        assert args.backend == "vllm"
        assert args.base_url == "http://x:1"

    def test_runtimes_subcommand_exists(self):
        assert _build_parser().parse_args(["runtimes"]).command == "runtimes"


class TestAbort:
    def test_abort_exits(self):
        with patch("protor.cli.sys.exit") as mock_exit:
            _abort("Test error", "Test hint")
            mock_exit.assert_called_once_with(1)
