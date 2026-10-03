"""Tests for the user-facing display commands.

These render terminal output, so a crash here breaks a CLI command outright.
They were the least-covered code in the project.
"""

import io

import pytest
from rich.console import Console

import protor.runtimes as _runtimes
from protor.analyzer import list_models, list_ollama_models, list_runtime_models, list_runtimes
from protor.exceptions import RuntimeUnavailableError
from protor.llm_backends import ModelInfo


@pytest.fixture
def captured(monkeypatch):
    """Capture whatever the analyzer prints, returning the text."""
    buf = io.StringIO()
    monkeypatch.setattr(
        "protor.analyzer.console",
        Console(file=buf, width=100, highlight=False, soft_wrap=True, legacy_windows=False),
    )
    return lambda: buf.getvalue()


def _stub_models(monkeypatch, names, *, available=True, size=1024):
    """
    Replace every backend's model listing with a canned result.

    Uses monkeypatch so the patch is undone after each test. Assigning directly
    to the class leaked into later modules and broke their real-backend tests.
    """
    import protor.llm_backends as lb

    def fake_list(self):
        return [ModelInfo(name=n, size_bytes=size, modified="2026-01-02") for n in names]

    def fake_available(self):
        return available

    for cls_name in ("OllamaBackend", "OpenAICompatBackend", "OpenAIBackend", "AnthropicBackend"):
        cls = getattr(lb, cls_name)
        monkeypatch.setattr(cls, "list_models", fake_list, raising=False)
        monkeypatch.setattr(cls, "check_available", fake_available, raising=False)


class TestListRuntimeModels:
    def test_lists_models_with_size_and_date(self, monkeypatch, captured):
        _stub_models(monkeypatch, ["llama3:latest", "mistral"], size=4 * 1024**3)
        list_runtime_models("ollama")
        out = captured()
        assert "llama3:latest" in out
        assert "mistral" in out
        assert "4.0 GB" in out
        assert "2026-01-02" in out

    def test_shows_the_resolved_url(self, monkeypatch, captured):
        _stub_models(monkeypatch, ["m"])
        list_runtime_models("lmstudio")
        assert "http://localhost:1234" in captured()

    def test_reports_a_stopped_runtime_with_a_start_hint(self, monkeypatch, captured):
        """
        The message is printed, and the command exits non-zero.

        Returning 0 meant `protor models` could not be told apart from a runtime
        with no models loaded, so a script saw success from a runtime that is not
        running. The typed error is what carries the URL out to the CLI.
        """
        _stub_models(monkeypatch, [], available=False)
        with pytest.raises(RuntimeUnavailableError) as exc:
            list_runtime_models("vllm")
        assert "vllm serve" in str(exc.value)
        out = captured()
        assert "not reachable" in out
        assert "vllm serve" in out, "must tell the user how to start it"

    def test_reports_an_empty_model_list(self, monkeypatch, captured):
        _stub_models(monkeypatch, [], available=True)
        list_runtime_models("ollama")
        out = captured()
        assert "No models" in out
        assert "ollama pull" in out

    def test_reports_a_non_ollama_empty_list_without_ollama_advice(self, monkeypatch, captured):
        _stub_models(monkeypatch, [], available=True)
        list_runtime_models("lmstudio")
        out = captured()
        assert "No models" in out
        assert "ollama pull" not in out

    def test_unknown_backend_does_not_raise(self, captured):
        """A bad name must print, not traceback."""
        list_runtime_models("not-a-runtime")
        assert "Unknown runtime" in captured()

    def test_listing_failure_is_reported(self, monkeypatch, captured):
        import protor.llm_backends as lb

        def boom(self):
            raise RuntimeError("socket died")

        monkeypatch.setattr(lb.OllamaBackend, "list_models", boom)
        monkeypatch.setattr(lb.OllamaBackend, "check_available", lambda self: True)
        list_runtime_models("ollama")
        assert "socket died" in captured()

    def test_shows_dash_when_no_size_is_reported(self, monkeypatch, captured):
        _stub_models(monkeypatch, ["m"], size=None)
        list_runtime_models("ollama")
        assert "—" in captured()

    def test_ollama_only_wrapper_still_works(self, monkeypatch, captured):
        _stub_models(monkeypatch, ["llama3"])
        list_ollama_models()
        assert "llama3" in captured()

    def test_module_level_helper_delegates(self, monkeypatch):
        _stub_models(monkeypatch, ["a", "b"])
        assert [m.name for m in list_models("ollama")] == ["a", "b"]


class TestListRuntimes:
    def test_reports_a_running_runtime(self, monkeypatch, captured):
        monkeypatch.setattr(
            "protor.analyzer.detect_runtimes",
            lambda: [__import__("protor.runtimes", fromlist=["RUNTIMES"]).RUNTIMES["vllm"]],
        )
        list_runtimes()
        out = captured()
        assert "running" in out
        assert "vllm" in out
        assert "--backend vllm" in out, "should suggest the follow-up command"

    def test_reports_nothing_running(self, monkeypatch, captured):
        monkeypatch.setattr("protor.analyzer.detect_runtimes", lambda: [])
        list_runtimes()
        out = captured()
        assert "No local runtime detected" in out
        assert "--backend openai" in out, "should mention the hosted alternative"

    def test_lists_every_registered_runtime(self, monkeypatch, captured):
        from protor.runtimes import RUNTIMES

        monkeypatch.setattr("protor.analyzer.detect_runtimes", lambda: [])
        list_runtimes()
        # rich folds a long cell by replacing the space with a newline, so
        # compare on collapsed whitespace rather than raw text.
        out = " ".join(captured().split())
        for runtime in RUNTIMES.values():
            assert runtime.label in out
            assert " ".join(runtime.start_hint.split()) in out

    def test_status_column_does_not_wrap(self, monkeypatch, captured):
        monkeypatch.setattr("protor.analyzer.detect_runtimes", lambda: [])
        list_runtimes()
        assert "stopped\n" not in captured()


class TestResponsiveLayout:
    """A terminal too narrow for four columns must lose the least useful one."""

    def _render(self, width):
        import io

        from rich.console import Console

        import protor.analyzer as analyzer

        original = analyzer.console
        analyzer.console = Console(file=io.StringIO(), width=width, highlight=False)
        analyzer.detect_runtimes = lambda *a, **k: []
        try:
            analyzer.list_runtimes()
            return analyzer.console.file.getvalue()
        finally:
            analyzer.console = original
            analyzer.detect_runtimes = _real_detect

    def test_wide_terminal_keeps_every_column(self):
        out = self._render(120)
        assert "URL" in out
        assert "http://localhost:11434" in out, "full URL, not clipped"
        assert "ollama serve" in out

    def test_narrow_terminal_does_not_clip_urls_mid_value(self):
        """
        "http://localhost:11434" used to render as "http://localhost:114" at
        60 columns, which reads as a different port entirely.
        """
        out = self._render(60)
        assert "localhost:114" not in out, "a clipped URL is a wrong URL"

    def test_narrow_terminal_keeps_the_start_commands(self):
        """The reason to run `protor runtimes` is the column that got dropped."""
        out = self._render(60)
        for hint in ("ollama serve", "vllm serve <model>", "localai run"):
            assert hint in out, f"lost an actionable hint: {hint}"

    def test_narrow_terminal_still_reports_status(self):
        out = self._render(60)
        assert "stopped" in out

    def test_works_at_a_very_narrow_width(self):
        out = self._render(40)
        assert "ollama serve" in out


_real_detect = _runtimes.detect_runtimes
