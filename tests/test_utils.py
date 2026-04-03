"""Tests for protor.utils."""

from __future__ import annotations

from pathlib import Path

import pytest

from protor.utils import (
    get_default_output_dir,
    human_bytes,
    load_json,
    safe_filename,
    save_json,
    timestamp,
)


class TestSafeFilename:
    def test_simple_domain(self):
        assert safe_filename("example.com") == "example.com"

    def test_replaces_special_chars(self):
        result = safe_filename("hello world/path?query=1")
        assert " " not in result
        assert "/" not in result
        assert "?" not in result
        assert "=" not in result

    def test_preserves_alphanum_dots_dashes(self):
        assert safe_filename("my-file_name.js") == "my-file_name.js"

    def test_empty_string_returns_unnamed(self):
        assert safe_filename("") == "unnamed"

    def test_strips_leading_trailing_underscores(self):
        result = safe_filename("!hello!")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_url_encoded_chars(self):
        r = safe_filename("https://example.com/page")
        assert "https" in r
        assert "example" in r


class TestSaveLoadJson:
    def test_round_trip(self, tmp_path: Path):
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        path = tmp_path / "sub" / "data.json"
        save_json(data, path)
        assert path.exists()
        loaded = load_json(path)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "a" / "b" / "c" / "data.json"
        save_json({"x": 1}, path)
        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "missing.json")

    def test_save_unicode(self, tmp_path: Path):
        data = {"emoji": "🚀", "japanese": "日本語"}
        path = tmp_path / "unicode.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded["emoji"] == "🚀"
        assert loaded["japanese"] == "日本語"

    def test_pretty_printed(self, tmp_path: Path):
        path = tmp_path / "pretty.json"
        save_json({"a": 1}, path)
        raw = path.read_text()
        assert "\n" in raw  # indented


class TestTimestamp:
    def test_format(self):
        ts = timestamp()
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", ts)


class TestHumanBytes:
    @pytest.mark.parametrize("n,expected", [
        (0,       "0 B"),
        (512,     "512 B"),
        (1024,    "1.0 KB"),
        (1536,    "1.5 KB"),
        (1_048_576, "1.0 MB"),
        (1_073_741_824, "1.0 GB"),
    ])
    def test_units(self, n: int, expected: str):
        assert human_bytes(n) == expected


class TestDefaultOutputDir:
    def test_returns_path(self):
        p = get_default_output_dir()
        assert isinstance(p, Path)
        assert "protor" in p.parts
        assert "Downloads" in p.parts
