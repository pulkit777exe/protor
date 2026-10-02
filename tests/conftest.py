"""Shared pytest fixtures for protor tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from protor.models import SiteManifest, SiteMetadata

if TYPE_CHECKING:
    from pathlib import Path


# ── HTTP test doubles ─────────────────────────────────────────────────────────
#
# These implement the *real* async context manager protocol. Hand-rolled
# AsyncMock sessions do not: `async with session.get(...)` never entered its
# body, so the code under test never ran and the tests passed through the
# `except Exception` fallback instead of asserting anything.
#
# `aioresponses` is not used because its latest release (0.7.9) is
# incompatible with aiohttp 3.14 (`ClientResponse.__init__` now requires
# `stream_writer`), which silently turned every assertion into the fallback.


class FakeResponse:
    """A minimal stand-in for ``aiohttp.ClientResponse``."""

    def __init__(self, status: int = 200, body: str = "", headers: dict | None = None) -> None:
        self.status = status
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self.headers = headers or {}

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def read(self) -> bytes:
        return self._body

    async def text(self) -> str:
        return self._body.decode("utf-8")


class FakeSession:
    """A session returning canned responses, and recording requested URLs.

    ``routes`` maps an exact URL to a :class:`FakeResponse`. ``default`` is
    returned for any unrouted URL.
    """

    def __init__(
        self,
        routes: dict[str, FakeResponse] | None = None,
        default: FakeResponse | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.routes = routes or {}
        self.default = default
        self.raises = raises
        self.requested: list[str] = []

    def get(self, url: str, **_kwargs: object) -> FakeResponse:
        self.requested.append(url)
        if self.raises is not None:
            raise self.raises
        if url in self.routes:
            return self.routes[url]
        if self.default is not None:
            return self.default
        raise AssertionError(f"FakeSession got an unrouted request: {url}")


@pytest.fixture
def fake_session():
    """Factory for :class:`FakeSession` doubles."""
    return FakeSession


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
