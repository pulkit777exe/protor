"""Shared pytest fixtures for protor tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from protor.models import SiteManifest, SiteMetadata

if TYPE_CHECKING:
    from pathlib import Path

# ── HTML fixtures ─────────────────────────────────────────────────────────────

SIMPLE_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <title>Test Site</title>
  <meta name="description" content="A test description.">
  <meta name="keywords" content="test, python, scraper">
  <meta name="author" content="Pulkit">
  <meta property="og:title" content="Test OG Title">
</head>
<body>
  <nav>Navigation</nav>
  <h1>Hello World</h1>
  <p>This is the main content of the page.</p>
  <a href="/about">About</a>
  <a href="/contact">Contact</a>
  <a href="https://external.com/page">External</a>
  <script src="/static/app.js"></script>
  <script src="https://cdn.example.com/lib.js"></script>
  <footer>Footer text</footer>
</body>
</html>
"""

EMPTY_HTML = "<html><body></body></html>"


# ── manifest fixture ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_manifest() -> SiteManifest:
    return SiteManifest(
        url="https://example.com",
        domain="example.com",
        html_file="/tmp/example/index.html",
        metadata=SiteMetadata(
            title="Example Domain",
            description="Example description",
            keywords=["example", "test"],
            author="",
            og_tags={},
        ),
        text_content="Example Domain\nThis domain is for use in examples.",
        js_files=[],
        js_count=0,
        bytes_received=1024,
        elapsed_ms=120,
        timestamp="2024-01-01 00:00:00",
        success=True,
    )


@pytest.fixture
def sample_manifests(sample_manifest: SiteManifest) -> list[dict]:
    return [sample_manifest.to_dict()]


# ── temp output dir ───────────────────────────────────────────────────────────


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    out = tmp_path / "protor_output"
    out.mkdir()
    return out


@pytest.fixture
def sites_index_file(tmp_path: Path, sample_manifests: list[dict]) -> Path:
    f = tmp_path / "sites_index.json"
    f.write_text(json.dumps(sample_manifests), encoding="utf-8")
    return f


@pytest.fixture
def mock_ollama_response() -> dict:
    return {"response": "Test analysis result", "done": False}
