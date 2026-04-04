"""Integration tests for protor CLI"""

import pytest
import os
import tempfile
import json
import shutil
from unittest.mock import patch, MagicMock
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
        with patch(
            "sys.argv", ["protor", "scrape", "https://example.com", "--output", self.temp_dir]
        ):
            try:
                cli()
            except SystemExit:
                pass

        mock_scrape.assert_called_once()

    @patch("protor.cli.list_ollama_models")
    def test_list_models_command(self, mock_list):
        """Test list-models command"""
        mock_list.return_value = None

        with patch("sys.argv", ["protor", "models"]):
            try:
                cli()
            except SystemExit:
                pass

        mock_list.assert_called_once()

    @patch("protor.cli.analyze_with_ollama")
    @patch("protor.cli.scrape_multiple")
    def test_run_command(self, mock_scrape_multiple, mock_analyze):
        """Test run command (scrape + analyze)"""

        site_dir = os.path.join(self.temp_dir, "example_com")
        os.makedirs(site_dir, exist_ok=True)
        json_path = os.path.join(site_dir, "sites_index.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"sites": [{"url": "https://example.com", "title": "Example"}]}, f)

        mock_scrape_multiple.return_value = json_path
        mock_analyze.return_value = "analysis.md"

        with patch(
            "sys.argv",
            ["protor", "run", "https://example.com", "-m", "llama3", "--output", self.temp_dir],
        ):
            try:
                cli()
            except SystemExit:
                pass

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

    def test_scrape_and_save(self):
        """Test complete scrape and save workflow"""
        import asyncio
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock
        from protor.scraper import scrape_site_async
        from protor.models import SiteManifest

        sample_html = """
        <html>
            <head><title>Test</title></head>
            <body><p>Content</p></body>
        </html>
        """

        async def run_test():
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=sample_html.encode("utf-8"))
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_session.get = MagicMock(return_value=mock_response)

            row_state = {}
            result = await scrape_site_async(
                mock_session,
                "https://example.com",
                Path(self.temp_dir),
                download_js=False,
                row_state=row_state,
            )

            assert result is not None
            assert isinstance(result, SiteManifest)
            assert result.success is True
            assert result.domain == "example.com"

        asyncio.run(run_test())
