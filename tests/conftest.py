"""Test fixtures and utilities for protor tests"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_html():
    """Sample HTML content for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <meta name="description" content="A test page">
        <meta name="keywords" content="test, sample, page">
        <meta property="og:title" content="Test Page OG">
        <script src="/js/app.js"></script>
    </head>
    <body>
        <header>
            <nav>
                <a href="/about">About</a>
                <a href="/contact">Contact</a>
            </nav>
        </header>
        <main>
            <h1>Welcome</h1>
            <p>This is a test page with some content.</p>
            <a href="https://example.com/page1">Internal Link 1</a>
            <a href="https://example.com/page2">Internal Link 2</a>
            <a href="https://external.com">External Link</a>
        </main>
        <footer>Footer content</footer>
    </body>
    </html>
    """


@pytest.fixture
def sample_manifest():
    """Sample manifest data for testing"""
    return {
        "url": "https://example.com",
        "domain": "example.com",
        "html_file": "/tmp/example_com/index.html",
        "metadata": {
            "title": "Test Page",
            "description": "A test page",
            "keywords": ["test", "sample"],
            "author": "",
            "og_tags": {"og:title": "Test Page OG"}
        },
        "text_content": "Welcome\nThis is a test page with some content.",
        "js_files": ["https://example.com/js/app.js"],
        "js_count": 1,
        "timestamp": "2024-01-01 12:00:00",
        "success": True
    }


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response"""
    return {
        "model": "llama3",
        "created_at": "2024-01-01T12:00:00Z",
        "response": "This is a test analysis response.",
        "done": True
    }
