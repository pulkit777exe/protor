"""
protor — async web scraper and AI analyzer.

Quickstart
----------
    from protor import scrape_multiple, analyze_with_ollama

    index = scrape_multiple(["https://example.com"])
    analyze_with_ollama(index, model="llama3", focus="general")
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("protor")
except PackageNotFoundError:
    __version__ = "dev"

__all__ = ["__version__"]