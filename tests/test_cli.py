"""Unit tests for protor.cli module"""

import json
from unittest.mock import patch

import pytest

from protor.cli import _abort, _build_parser, _load_index
from protor.exceptions import DataFileNotFoundError, InvalidManifestError
from protor.models import SiteManifest
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
        # Manifests, not the raw dicts: _load_index routes every row through
        # SiteManifest.from_dict, which is what turns a wrong --file into a
        # sentence instead of an AttributeError from analyzer._site_header.
        assert [m.to_dict() for m in result] == [
            SiteManifest(domain="example.com", url="https://example.com").to_dict()
        ]

    def test_a_record_that_names_no_page_is_refused(self, tmp_path):
        """A dict without url/domain raised deep in the analyzer; it raises here."""
        index_file = tmp_path / "sites_index.json"
        index_file.write_text(json.dumps([{"note": "hi"}]))
        with pytest.raises(InvalidManifestError):
            _load_index(str(index_file))

    def test_a_file_that_is_not_a_list_of_manifests_is_refused(self, tmp_path):
        """The shape --file most often gets wrong: a dict, or a list of strings."""
        for payload in ('{"note": "hi"}', '["https://example.com"]'):
            index_file = tmp_path / "sites_index.json"
            index_file.write_text(payload)
            with pytest.raises(InvalidManifestError):
                _load_index(str(index_file))

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


class TestModuleEntryPoint:
    def test_importing_dunder_main_does_not_run_the_cli(self):
        """
        `protor/__main__.py` called `cli()` at module scope.

        `python -m protor` sets `__name__ == "__main__"`, so the guard changes
        nothing there — but `pkgutil.iter_modules` lists `__main__` among the
        package's submodules like any other, so anything that walks the package
        (an import-everything helper, a coverage sweep, a docs generator) ran the
        whole CLI on import and exited with whatever argv it was holding. It
        surfaced as `protor: error: argument <command>: invalid choice:
        'tests/test_isolation.py'`.
        """
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-c", "import protor.__main__; print('inert')"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "inert" in proc.stdout, proc.stdout

    def test_python_dash_m_still_runs_the_cli(self):
        """The other half of the guard: the documented entry point must survive."""
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "protor", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().startswith("protor "), proc.stdout
