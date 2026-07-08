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
    def test_choices_count(self):
        assert len(FORMAT_CHOICES) == 4

    def test_expected_formats(self):
        assert "markdown" in FORMAT_CHOICES
        assert "csv" in FORMAT_CHOICES
        assert "html" in FORMAT_CHOICES
        assert "text" in FORMAT_CHOICES


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
