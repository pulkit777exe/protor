"""
protor — async web scraper and AI analyzer.

Quickstart
----------
    from protor.scraper import scrape_multiple
    from protor.analyzer import analyze
    from protor.utils import load_json

    index_path = scrape_multiple(["https://example.com"])
    analyze(load_json(index_path), model="llama3", focus="general")

``scrape_multiple`` returns the *path* to ``sites_index.json``; ``analyze``
takes the loaded list of manifests, not the path.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("protor")
except PackageNotFoundError:
    __version__ = "dev"

__all__ = ["__version__"]
