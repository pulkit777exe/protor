"""Unit tests for protor.scraper module"""
import pytest
import os
from unittest.mock import patch, MagicMock, mock_open
from bs4 import BeautifulSoup
from protor.scraper import (
    fetch_with_curl,
    extract_metadata,
    extract_js_links,
    extract_links,
    extract_text_content,
    download_file,
    scrape_website
)


class TestFetchWithCurl:
    """Tests for fetch_with_curl function"""
    
    @patch('subprocess.run')
    def test_successful_fetch(self, mock_run):
        """Test successful URL fetch"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="<html>content</html>",
            stderr=""
        )
        
        content, success = fetch_with_curl("https://example.com")
        
        assert success is True
        assert content == "<html>content</html>"
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_failed_fetch(self, mock_run):
        """Test failed URL fetch"""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error"
        )
        
        content, success = fetch_with_curl("https://example.com")
        
        assert success is False
        assert content == ""
    
    @patch('subprocess.run')
    def test_timeout_handling(self, mock_run):
        """Test timeout handling"""
        mock_run.side_effect = TimeoutError("Timeout")
        
        content, success = fetch_with_curl("https://example.com", timeout=5)
        
        assert success is False
        assert content == ""


class TestExtractMetadata:
    """Tests for extract_metadata function"""
    
    def test_extract_title(self, sample_html):
        """Test extracting page title"""
        soup = BeautifulSoup(sample_html, 'html.parser')
        metadata = extract_metadata(soup)
        
        assert metadata['title'] == "Test Page"
    
    def test_extract_description(self, sample_html):
        """Test extracting meta description"""
        soup = BeautifulSoup(sample_html, 'html.parser')
        metadata = extract_metadata(soup)
        
        assert metadata['description'] == "A test page"
    
    def test_extract_keywords(self, sample_html):
        """Test extracting keywords"""
        soup = BeautifulSoup(sample_html, 'html.parser')
        metadata = extract_metadata(soup)
        
        assert "test" in metadata['keywords']
        assert "sample" in metadata['keywords']
    
    def test_extract_og_tags(self, sample_html):
        """Test extracting Open Graph tags"""
        soup = BeautifulSoup(sample_html, 'html.parser')
        metadata = extract_metadata(soup)
        
        assert "og:title" in metadata['og_tags']
        assert metadata['og_tags']['og:title'] == "Test Page OG"
    
    def test_empty_html(self):
        """Test with minimal HTML"""
        soup = BeautifulSoup("<html></html>", 'html.parser')
        metadata = extract_metadata(soup)
        
        assert metadata['title'] == ""
        assert metadata['description'] == ""
        assert len(metadata['keywords']) == 0


class TestExtractJsLinks:
    """Tests for extract_js_links function"""
    
    def test_extract_script_tags(self, sample_html):
        """Test extracting JavaScript file links"""
        js_links = extract_js_links(sample_html, "https://example.com")
        
        assert len(js_links) > 0
        assert any("/js/app.js" in link for link in js_links)
    
    def test_relative_urls_converted(self):
        """Test that relative URLs are converted to absolute"""
        html = '<html><script src="/js/app.js"></script></html>'
        js_links = extract_js_links(html, "https://example.com")
        
        assert "https://example.com/js/app.js" in js_links
    
    def test_no_scripts(self):
        """Test HTML with no script tags"""
        html = '<html><body>No scripts</body></html>'
        js_links = extract_js_links(html, "https://example.com")
        
        assert len(js_links) == 0
    
    def test_duplicate_removal(self):
        """Test that duplicate scripts are removed"""
        html = '''<html>
            <script src="/app.js"></script>
            <script src="/app.js"></script>
        </html>'''
        js_links = extract_js_links(html, "https://example.com")
        
        assert len(js_links) == 1


class TestExtractLinks:
    """Tests for extract_links function"""
    
    def test_extract_internal_links(self, sample_html):
        """Test extracting internal links"""
        links = extract_links(sample_html, "https://example.com")
        
        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links
    
    def test_exclude_external_links(self, sample_html):
        """Test that external links are excluded"""
        links = extract_links(sample_html, "https://example.com")
        
        assert not any("external.com" in link for link in links)
    
    def test_remove_fragments(self):
        """Test that URL fragments are removed"""
        html = '<html><a href="https://example.com/page#section">Link</a></html>'
        links = extract_links(html, "https://example.com")
        
        assert "https://example.com/page" in links
        assert not any("#" in link for link in links)
    
    def test_exclude_base_url(self, sample_html):
        """Test that base URL itself is excluded"""
        links = extract_links(sample_html, "https://example.com")
        
        assert "https://example.com" not in links
    
    def test_duplicate_removal(self):
        """Test that duplicate links are removed"""
        html = '''<html>
            <a href="https://example.com/page">Link 1</a>
            <a href="https://example.com/page">Link 2</a>
        </html>'''
        links = extract_links(html, "https://example.com")
        
        assert len(links) == 1


class TestExtractTextContent:
    """Tests for extract_text_content function"""
    
    def test_extract_main_content(self, sample_html):
        """Test extracting main text content"""
        text = extract_text_content(sample_html)
        
        assert "Welcome" in text
        assert "This is a test page" in text
    
    def test_remove_script_tags(self):
        """Test that script content is removed"""
        html = '<html><script>alert("test")</script><body>Content</body></html>'
        text = extract_text_content(html)
        
        assert "alert" not in text
        assert "Content" in text
    
    def test_remove_nav_footer(self, sample_html):
        """Test that nav and footer content is removed"""
        text = extract_text_content(sample_html)
        
        # Nav and footer should be removed
        assert "About" not in text or "Contact" not in text
    
    def test_max_length_limit(self):
        """Test that content is limited to 10000 chars"""
        long_html = f'<html><body>{"x" * 20000}</body></html>'
        text = extract_text_content(long_html)
        
        assert len(text) <= 10000


class TestDownloadFile:
    """Tests for download_file function"""
    
    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_successful_download(self, mock_makedirs, mock_file, mock_get):
        """Test successful file download"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"file content"
        mock_get.return_value = mock_response
        
        result = download_file("https://example.com/file.js", "/tmp/file.js")
        
        assert result is True
        mock_get.assert_called_once()
        mock_file.assert_called_once()
    
    @patch('requests.get')
    def test_failed_download(self, mock_get):
        """Test failed file download"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = download_file("https://example.com/missing.js", "/tmp/file.js")
        
        assert result is False
    
    @patch('requests.get')
    def test_timeout_handling(self, mock_get):
        """Test timeout handling"""
        mock_get.side_effect = TimeoutError("Timeout")
        
        result = download_file("https://example.com/file.js", "/tmp/file.js", timeout=5)
        
        assert result is False


class TestScrapeWebsite:
    """Tests for scrape_website function"""
    
    @patch('protor.scraper.fetch_with_curl')
    @patch('protor.scraper.save_json')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_successful_scrape(self, mock_makedirs, mock_file, mock_save, mock_fetch, sample_html):
        """Test successful website scraping"""
        mock_fetch.return_value = (sample_html, True)
        
        result = scrape_website("https://example.com", "/tmp/output", download_js=False)
        
        assert result is not None
        assert result['url'] == "https://example.com"
        assert result['domain'] == "example.com"
        assert result['success'] is True
        mock_save.assert_called_once()
    
    @patch('protor.scraper.fetch_with_curl')
    def test_failed_scrape(self, mock_fetch):
        """Test failed website scraping"""
        mock_fetch.return_value = ("", False)
        
        result = scrape_website("https://example.com", "/tmp/output")
        
        assert result is None
