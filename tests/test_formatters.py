"""Tests for protor.formatters module."""

import pytest

from protor.formatters import FORMAT_CHOICES, format_output, write_output
from protor.models import AnalysisResult


@pytest.fixture
def sample_result():
    return AnalysisResult(
        model="llama3",
        focus="general",
        timestamp="2024-01-01 00:00:00",
        sites_analyzed=3,
        analysis="Test analysis content",
    )


class TestFormatChoices:
    """
    `json` joined the list because `analyze` already wrote `analysis.json`
    unconditionally, whatever `--format` said. The file existed; asking for it by
    name was an "invalid choice" error, and the two definitions of "the report" were
    separate code paths that could have drifted.
    """

    def test_expected_formats(self):
        for fmt in ("markdown", "json", "csv", "html", "text"):
            assert fmt in FORMAT_CHOICES, fmt

    def test_every_choice_can_be_formatted(self):
        """A choice argparse accepts must not raise from `format_output`."""
        for fmt in FORMAT_CHOICES:
            assert format_output(sample_result_for(fmt), fmt)

    def test_json_is_the_same_document_the_separate_write_produced(self):
        """
        One definition of the report.

        `_to_json` is `to_dict`, which is what the unconditional `save_json` used, so
        asking for json must produce that document and not a near relative.
        """
        import json

        from protor.formatters import format_output

        payload = json.loads(format_output(sample_result_for("json"), "json"))
        assert payload == sample_result_for("json").to_dict()


def sample_result_for(_fmt: str = "markdown"):
    """A minimal AnalysisResult, built per call so callers cannot mutate a shared one."""
    from protor.models import AnalysisResult

    return AnalysisResult(
        model="llama3",
        focus="general",
        timestamp="2026-01-01 00:00:00",
        sites_analyzed=1,
        analysis='# report\n\nBody with a quote: "hello".',
    )


class TestFormatOutput:
    def test_markdown_format(self, sample_result):
        result = format_output(sample_result, "markdown")
        assert "# Website Analysis Report" in result
        assert "llama3" in result
        assert "general" in result
        assert "3" in result
        assert "Test analysis content" in result

    def test_text_format(self, sample_result):
        result = format_output(sample_result, "text")
        assert "Website Analysis Report" in result
        assert "=" * 40 in result
        assert "llama3" in result
        assert "Test analysis content" in result

    def test_csv_format(self, sample_result):
        result = format_output(sample_result, "csv")
        assert "timestamp,model,focus,sites_analyzed,analysis" in result
        assert "2024-01-01 00:00:00" in result
        assert "llama3" in result

    def test_html_format(self, sample_result):
        result = format_output(sample_result, "html")
        assert "<!DOCTYPE html>" in result
        assert "Website Analysis Report" in result
        assert "llama3" in result
        assert "Test analysis content" in result

    def test_unknown_format_raises(self, sample_result):
        with pytest.raises(ValueError, match="Unknown format"):
            format_output(sample_result, "xml")


class TestWriteOutput:
    def test_write_markdown(self, sample_result, tmp_path):
        path = write_output(sample_result, tmp_path, "markdown")
        assert path.exists()
        assert path.name == "analysis.md"
        assert "# Website Analysis Report" in path.read_text()

    def test_write_text(self, sample_result, tmp_path):
        path = write_output(sample_result, tmp_path, "text")
        assert path.exists()
        assert path.name == "analysis.txt"

    def test_write_csv(self, sample_result, tmp_path):
        path = write_output(sample_result, tmp_path, "csv")
        assert path.exists()
        assert path.name == "analysis.csv"

    def test_write_html(self, sample_result, tmp_path):
        path = write_output(sample_result, tmp_path, "html")
        assert path.exists()
        assert path.name == "analysis.html"

    def test_creates_directory(self, sample_result, tmp_path):
        out_dir = tmp_path / "nested" / "dir"
        path = write_output(sample_result, out_dir)
        assert path.exists()
        assert path.parent == out_dir


class TestJsonIsWrittenOnce:
    """
    `--format json` used to write `analysis.json` twice.

    `analyze` wrote it unconditionally and separately, then `write_output` wrote the
    same document to the same path because that is what the format maps to. The file
    is byte-identical either way, so no content assertion can see it — the cost is a
    redundant write, and the visible half was the "saved" line naming one path twice.
    """

    def _analyze(self, fmt: str, tmp_path, monkeypatch):
        import protor.analyzer as analyzer_mod

        calls: list[dict] = []
        monkeypatch.setattr(analyzer_mod, "create_backend", lambda *a, **k: _FakeBackend())
        monkeypatch.setattr(analyzer_mod, "_stream_backend", lambda llm, prompt: "# report")
        monkeypatch.setattr(
            analyzer_mod,
            "save_json",
            lambda data, path: calls.append({"path": path, "data": data}),
        )
        analyzer_mod.analyze(
            [{"url": "https://ex.com", "domain": "ex.com", "text_content": "x"}],
            output_dir=tmp_path / fmt,
            fmt=fmt,
        )
        return calls

    def test_json_is_not_also_written_by_the_separate_path(self, tmp_path, monkeypatch):
        """One definition of the report, written once, whichever format asked for it."""
        assert self._analyze("json", tmp_path, monkeypatch) == []

    def test_another_format_still_gets_analysis_json(self, tmp_path, monkeypatch):
        """The control: `analysis.json` is a contract, not a side effect of `--format json`."""
        calls = self._analyze("markdown", tmp_path, monkeypatch)
        assert len(calls) == 1, calls
        assert calls[0]["path"].name == "analysis.json"

    def test_the_saved_line_names_it_once(self, tmp_path, monkeypatch):
        import io

        import protor.analyzer as analyzer_mod
        from protor import theme

        buf = io.StringIO()
        monkeypatch.setattr(
            analyzer_mod,
            "console",
            theme.ProtorConsole(file=buf, width=100, force_terminal=False, highlight=False),
        )
        monkeypatch.setattr(analyzer_mod, "create_backend", lambda *a, **k: _FakeBackend())
        monkeypatch.setattr(analyzer_mod, "_stream_backend", lambda llm, prompt: "# report")
        analyzer_mod.analyze(
            [{"url": "https://ex.com", "domain": "ex.com", "text_content": "x"}],
            output_dir=tmp_path / "j",
            fmt="json",
        )
        named = [line for line in buf.getvalue().splitlines() if "analysis.json" in line]
        assert len(named) == 1, buf.getvalue()


class _FakeBackend:
    display_name = "Fake"

    def check_available(self) -> bool:
        return True

    def start_hint(self) -> str:
        return ""
