"""The public surface that nothing else in the repo calls.

`analyze_with_ollama` and `extract_from_html` are exported from their modules'
``__all__`` but have no internal caller — they exist for library users. That is
a reasonable thing to ship, but an untested export is a liability: nothing
notices when it rots. These tests pin what each one promises, so removing or
breaking one fails here instead of in someone else's code.
"""

from __future__ import annotations

import pytest

from protor.analyzer import analyze_with_ollama, analyze_with_runtime
from protor.extractor import ExtractionSchema, extract_from_html, extract_from_soup

SCHEMA = ExtractionSchema.from_dict(
    {
        "name": "products",
        "base_selector": ".product",
        "fields": [
            {"name": "name", "selector": "h2", "type": "text"},
            {"name": "price", "selector": ".price", "type": "text"},
        ],
    }
)

HTML = (
    '<html><body><div class="product"><h2>Widget</h2>'
    '<span class="price">$10</span></div></body></html>'
)


class TestExtractFromHtml:
    """A convenience wrapper over Extractor.extract — exported, so pinned."""

    def test_extracts_records(self):
        records = extract_from_html(HTML, SCHEMA, base_url="https://shop.test/")
        assert records == [{"name": "Widget", "price": "$10"}]

    def test_resolves_relative_urls_against_the_base(self):
        schema = ExtractionSchema.from_dict(
            {
                "name": "links",
                "base_selector": ".product",
                "fields": [{"name": "href", "selector": "a", "type": "href"}],
            }
        )
        html = '<div class="product"><a href="/buy">buy</a></div>'
        assert extract_from_html(html, schema, "https://shop.test/c/") == [
            {"href": "https://shop.test/buy"}
        ]

    def test_returns_nothing_for_a_page_with_no_matches(self):
        assert extract_from_html("<html><body><p>nope</p></body></html>", SCHEMA) == []

    def test_agrees_with_the_class_based_api(self):
        """Two paths to the same answer must not drift apart."""
        from bs4 import BeautifulSoup

        via_function = extract_from_html(HTML, SCHEMA)
        via_class = extract_from_soup(BeautifulSoup(HTML, "html.parser"), SCHEMA)
        assert via_function == via_class


class TestAnalyzeWithOllama:
    """The pre-1.x entry point, kept working for library users."""

    @pytest.fixture
    def backend(self, monkeypatch):
        class Fake:
            model_name = "llama3"
            display_name = "Ollama"
            base_url = "http://localhost:11434"

            def check_available(self):
                return True

            def stream(self, prompt):
                yield "**Overview**\nA site."

        monkeypatch.setattr("protor.analyzer.create_backend", lambda *a, **k: Fake())
        monkeypatch.setattr(
            "protor.analyzer.console",
            __import__("rich.console", fromlist=["Console"]).Console(
                file=__import__("io").StringIO(), width=80, highlight=False
            ),
        )
        return Fake()

    def _site(self) -> dict:
        return {
            "url": "https://example.com/",
            "domain": "example.com",
            "metadata": {"title": "Example"},
            "text_content": "Content.",
            "js_count": 0,
        }

    def test_routes_to_the_ollama_backend(self, backend, monkeypatch):
        seen = {}

        def spy(backend_name, model, **kwargs):
            seen["backend"] = backend_name
            return backend

        monkeypatch.setattr("protor.analyzer.create_backend", spy)
        analyze_with_ollama([self._site()], output_dir="/tmp/protor-test-wrapper")
        assert seen["backend"] == "ollama"

    def test_writes_the_report(self, backend, tmp_path):
        result = analyze_with_ollama([self._site()], output_dir=tmp_path)
        assert (tmp_path / "analysis.json").exists()
        assert result.sites_analyzed == 1

    def test_matches_analyze_with_runtime_on_the_ollama_backend(self, backend, tmp_path):
        """The wrapper must be a shim, not a second implementation."""
        legacy = analyze_with_ollama([self._site()], output_dir=tmp_path / "a")
        direct = analyze_with_runtime([self._site()], "ollama", "llama3", "general", tmp_path / "b")
        assert legacy.analysis == direct.analysis
        assert legacy.sites_analyzed == direct.sites_analyzed
        assert legacy.model == direct.model

    def test_is_a_thin_wrapper_not_a_fork(self):
        """A shim that grows its own logic is a second implementation waiting."""
        import inspect

        source = inspect.getsource(analyze_with_ollama)
        assert source.count("analyze(") == 1, "the wrapper should delegate once"
        assert "if " not in source.split('"""')[-1], "no branching of its own"
        assert analyze_with_ollama.__doc__.strip().startswith("Backwards-compatible")
