"""Integration tests for protor CLI"""
import pytest
import os
import tempfile
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
    
    @patch('protor.scraper.scrape_website')
    def test_scrape_command(self, mock_scrape):
        """Test scrape command execution"""
        mock_scrape.return_value = {"success": True}
        
        # Simulate CLI call
        with patch('sys.argv', ['protor', 'scrape', 'https://example.com', '--output', self.temp_dir]):
            try:
                cli()
            except SystemExit:
                pass
        
        mock_scrape.assert_called_once()
    
    @patch('protor.analyzer.list_ollama_models')
    def test_list_models_command(self, mock_list):
        """Test list-models command"""
        mock_list.return_value = [
            {"name": "llama3", "size": 4661224192}
        ]
        
        with patch('sys.argv', ['protor', 'list-models']):
            try:
                cli()
            except SystemExit:
                pass
        
        mock_list.assert_called_once()
    
    @patch('protor.analyzer.analyze_with_ollama')
    @patch('protor.scraper.scrape_website')
    def test_run_command(self, mock_scrape, mock_analyze):
        """Test run command (scrape + analyze)"""
        mock_scrape.return_value = {"success": True}
        mock_analyze.return_value = "analysis.md"
        
        with patch('sys.argv', ['protor', 'run', 'https://example.com', '--output', self.temp_dir]):
            try:
                cli()
            except SystemExit:
                pass
        
        mock_scrape.assert_called_once()
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
    
    @patch('protor.scraper.fetch_with_curl')
    def test_scrape_and_save(self, mock_fetch):
        """Test complete scrape and save workflow"""
        sample_html = """
        <html>
            <head><title>Test</title></head>
            <body><p>Content</p></body>
        </html>
        """
        mock_fetch.return_value = (sample_html, True)
        
        from protor.scraper import scrape_website
        
        result = scrape_website("https://example.com", self.temp_dir, download_js=False)
        
        assert result is not None
        assert result['success'] is True
        assert os.path.exists(os.path.join(self.temp_dir, "example_com", "manifest.json"))
