"""Shared pytest fixtures for protor tests."""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

import pytest

from protor.models import SiteManifest, SiteMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# ── test isolation ─────────────────────────────────────────────────────────────


def _import_every_submodule() -> None:
    """
    Import every ``protor`` submodule now, so the snapshot below is complete.

    Without this the detector only covers whichever modules happened to be
    imported by the time a test ran: patch a function in a module that nothing
    has touched yet, and the patch was invisible — which is the one case where a
    leak does the most damage, because the importing test is the one that owns
    the patched name.
    """
    import contextlib
    import importlib
    import pkgutil

    import protor

    for info in pkgutil.iter_modules(protor.__path__, prefix="protor."):
        # A submodule that will not import is a failure the tests themselves will
        # report; here it just means less coverage.
        with contextlib.suppress(Exception):
            importlib.import_module(info.name)


_import_every_submodule()


def _module_snapshot() -> dict[tuple[str, str], int]:
    """Identity of every mutable attribute of every imported protor module."""
    out: dict[tuple[str, str], int] = {}
    for name, module in list(sys.modules.items()):
        if name != "protor" and not name.startswith("protor."):
            continue
        try:
            namespace = vars(module)
        except TypeError:  # a namespace package has no __dict__
            continue
        for attr, value in list(namespace.items()):
            if attr.startswith("__") or attr.isupper():
                # Constants are the one thing a test may legitimately rebind and
                # leave; lazy caches are initialised on first use, not leaked.
                continue
            out[(name, attr)] = id(value)
    return out


@pytest.fixture(autouse=True)
def _no_leaked_module_patches() -> Iterator[None]:
    """
    Fail a test that leaves a protor module attribute swapped out.

    Off by default because it costs ~8% of the suite; CI runs a job with
    ``PROTOR_CHECK_ISOLATION=1`` to turn it on.

    It exists because of how badly this class of bug hides. A test that patches
    ``protor.engine.fetch`` and restores ``parse_html`` in a ``finally`` but not
    ``fetch`` passes in isolation, passes in its own file, and then fails a
    schema-extraction test in a different file with zero records — four hundred
    tests later, with nothing in between to connect the two. The failure reads
    as a product bug in the extractor. Naming the offending attribute at the
    point of the patch turns that into a one-line diagnosis.
    """
    if not os.environ.get("PROTOR_CHECK_ISOLATION"):
        yield
        return

    before = _module_snapshot()
    yield
    leaked = [
        f"{module}.{attr}"
        for (module, attr), ident in _module_snapshot().items()
        if before.get((module, attr), ident) != ident
    ]
    if leaked:
        pytest.fail(
            "this test leaked module patches: " + ", ".join(sorted(leaked)),
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _http_cache_in_tmp(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """
    Keep every ``HTTPCache`` in the suite out of the developer's real cache.

    Session-scoped in effect — one directory for the whole run, set before any
    test imports a cache — because the default is a *single* file at
    ``~/.cache/protor/http/index.json`` that every ``HTTPCache()`` with no
    argument opens. Two consequences, both observed rather than theorised:

    * An instance holds its own copy of the index in memory and writes the whole
      file back on ``flush``. The recrawl tests force a revalidation by ageing
      every entry in that file; any other test flushing afterwards restores the
      original timestamps, so the revalidation never happens and
      ``test_a_304_is_reported_as_unchanged`` fails — intermittently, depending on
      which test happened to run next. Demonstrated directly: age an index, let a
      second cache put one entry and flush, and the aged entries come back fresh.
    * The run leaves entries pointing at a test server's dead port in the user's
      real cache, which then grows without bound — 1,165 entries across 233 hosts
      here, none of them ever fetchable again.

    Set on ``protor.config`` rather than ``os.environ`` because
    ``HTTP_CACHE_DIR`` is read at import time, which has already happened by the
    time any fixture runs; rebinding the name is what the code actually reads.
    A test wanting a private cache can still pass ``cache_dir=`` explicitly, which
    takes precedence.
    """
    import protor.config
    import protor.http_cache

    previous = protor.config.HTTP_CACHE_DIR
    cache_dir = tmp_path_factory.mktemp("http-cache")
    protor.config.HTTP_CACHE_DIR = str(cache_dir)
    # Already bound into this module's namespace at import; rebind it too so the
    # fixture works even if the import order ever changes.
    protor.http_cache.HTTP_CACHE_DIR = str(cache_dir)
    try:
        yield
    finally:
        protor.config.HTTP_CACHE_DIR = previous
        protor.http_cache.HTTP_CACHE_DIR = previous


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

    def __init__(
        self,
        status: int = 200,
        body: str = "",
        headers: dict | None = None,
        url: str = "http://test.local/",
    ) -> None:
        self.status = status
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self.headers = headers or {}
        # Real ClientResponse carries the final URL, which redirect handling
        # needs to resolve a relative `Location` against. Absent here, every
        # redirect test failed with AttributeError instead of testing redirects.
        self.url = url

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
