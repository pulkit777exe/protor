"""Integration tests for protor CLI"""

import contextlib
import json
import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from protor.cli import cli


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI commands"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("protor.cli.scrape_multiple")
    def test_scrape_command(self, mock_scrape):
        """Test scrape command execution"""
        mock_scrape.return_value = os.path.join(self.temp_dir, "sites_index.json")

        # Simulate CLI call
        with (
            patch(
                "sys.argv", ["protor", "scrape", "https://example.com", "--output", self.temp_dir]
            ),
            contextlib.suppress(SystemExit),
        ):
            cli()

        mock_scrape.assert_called_once()

    @patch("protor.cli.list_runtime_models")
    def test_list_models_command(self, mock_list):
        """Test list-models command"""
        mock_list.return_value = None

        with patch("sys.argv", ["protor", "models"]), contextlib.suppress(SystemExit):
            cli()

        mock_list.assert_called_once_with("ollama", base_url=None, api_key=None)

    @patch("protor.cli.list_runtimes")
    def test_runtimes_command(self, mock_runtimes):
        mock_runtimes.return_value = None

        with patch("sys.argv", ["protor", "runtimes"]), contextlib.suppress(SystemExit):
            cli()

        mock_runtimes.assert_called_once()

    @patch("protor.cli.analyze_with_runtime")
    @patch("protor.cli.scrape_multiple")
    def test_run_command(self, mock_scrape_multiple, mock_analyze):
        """Test run command (scrape + analyze)"""

        site_dir = os.path.join(self.temp_dir, "example_com")
        os.makedirs(site_dir, exist_ok=True)
        json_path = os.path.join(site_dir, "sites_index.json")
        # A bare JSON *array* of manifests, which is what _write_manifest_index
        # produces. This fixture was a {"sites": [...]} object instead — a shape
        # nothing writes — and `_load_index` iterating it handed the key "sites"
        # to the analyzer. It passed only because analyze_with_runtime was mocked
        # out; with the mock in place the malformed index was never looked at.
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"url": "https://example.com", "domain": "example.com", "title": "Example"}], f
            )

        mock_scrape_multiple.return_value = json_path
        mock_analyze.return_value = "analysis.md"

        with (
            patch(
                "sys.argv",
                ["protor", "run", "https://example.com", "-m", "llama3", "--output", self.temp_dir],
            ),
            contextlib.suppress(SystemExit),
        ):
            cli()

        mock_scrape_multiple.assert_called_once()
        mock_analyze.assert_called_once()


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_scrape_and_save(self, fake_session):
        """End-to-end: fetch -> parse -> write HTML + manifest."""
        from pathlib import Path

        from protor.models import SiteManifest
        from protor.scraper import scrape_site_async
        from tests.conftest import FakeResponse

        sample_html = (
            "<html><head><title>Test</title></head>"
            "<body><p>Content</p><nav>skip me</nav></body></html>"
        )
        session = fake_session(
            routes={"https://example.com": FakeResponse(status=200, body=sample_html)}
        )

        row_state: dict = {}
        result = await scrape_site_async(
            session,
            "https://example.com",
            Path(self.temp_dir),
            download_js=False,
            row_state=row_state,
            check_robots=False,
        )

        assert isinstance(result, SiteManifest)
        assert result.success is True
        assert result.domain == "example.com"
        assert result.metadata.title == "Test"
        assert "Content" in result.text_content
        # The noise pass must have removed the nav, in both artefacts.
        assert "skip me" not in result.text_content
        assert "skip me" not in result.markdown_content
        # row_state is what a Live table renders from.
        assert row_state["status"] == "done"

        site_dir = Path(self.temp_dir) / "example.com"
        assert (site_dir / "index.html").exists()
        assert (site_dir / "manifest.json").exists()
        assert "Content" in (site_dir / "index.html").read_text(encoding="utf-8")
