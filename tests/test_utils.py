"""Tests for protor.utils."""

from __future__ import annotations

from pathlib import Path

import pytest

from protor.utils import (
    canonicalize_url,
    get_default_output_dir,
    human_bytes,
    load_json,
    manifest_filename,
    page_filename,
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


class TestPageFilename:
    def test_root_uses_fallback(self):
        assert page_filename("https://example.com/") == "index.html"

    def test_root_path_page_is_index(self):
        assert page_filename("https://example.com/index.html") == "index.html"

    def test_keeps_path_segment(self):
        assert page_filename("https://example.com/about.html") == "about.html"

    def test_extensionless_page(self):
        assert page_filename("https://example.com/blog/post") == "blog-post"

    def test_ignores_query_and_fragment(self):
        assert page_filename("http://example.com/page?utm=1#top") == "page"

    def test_decodes_percent_encoding(self):
        assert page_filename("http://example.com/my%20page.html") == "my_page.html"

    def test_sanitizes_hostile_segments(self):
        name = page_filename("http://example.com/../../etc/passwd")
        assert ".." not in name
        assert "/" not in name
        assert name.endswith("passwd")

    def test_different_paths_do_not_collide(self):
        a = page_filename("http://example.com/index.html")
        b = page_filename("http://example.com/about.html")
        assert a != b


class TestCanonicalizeUrl:
    def test_index_html_collapses_to_root(self):
        assert canonicalize_url("http://example.com/index.html") == "http://example.com/"

    def test_subdir_index_collapses(self):
        assert canonicalize_url("http://example.com/blog/index.html") == "http://example.com/blog/"

    def test_fragment_stripped(self):
        assert canonicalize_url("http://example.com/about#top") == "http://example.com/about"

    def test_host_and_scheme_lowercased(self):
        assert canonicalize_url("HTTPS://EXAMPLE.com/Page") == "https://example.com/Page"

    def test_query_preserved(self):
        assert canonicalize_url("http://example.com/list?page=2&x=1#f") == (
            "http://example.com/list?page=2&x=1"
        )

    def test_trailing_index_variant_dedup(self):
        a = canonicalize_url("http://example.com/")
        b = canonicalize_url("http://example.com/index.html")
        assert a == b

    def test_normal_path_untouched(self):
        assert canonicalize_url("http://example.com/about.html") == "http://example.com/about.html"


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
    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, "0 B"),
            (512, "512 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1_048_576, "1.0 MB"),
            (1_073_741_824, "1.0 GB"),
        ],
    )
    def test_units(self, n: int, expected: str):
        assert human_bytes(n) == expected


class TestDefaultOutputDir:
    def test_returns_path(self):
        p = get_default_output_dir()
        assert isinstance(p, Path)
        assert "protor" in p.parts
        assert "Downloads" in p.parts


class TestManifestFilename:
    """A multi-page crawl must not overwrite earlier manifests."""

    def test_site_root_keeps_conventional_name(self):
        assert manifest_filename("https://x.com/") == "manifest.json"
        assert manifest_filename("https://x.com/index.html") == "manifest.json"

    def test_other_pages_are_namespaced(self):
        assert manifest_filename("https://x.com/docs/guide.html") == "docs-guide.manifest.json"

    def test_distinct_pages_never_collide(self):
        urls = [
            "https://x.com/",
            "https://x.com/docs/a.html",
            "https://x.com/blog/a.html",
            "https://x.com/b.html",
        ]
        names = [manifest_filename(u) for u in urls]
        # "/" and "/index.html" are the same page, so they may share a name;
        # everything else must be distinct.
        assert len(set(names[1:])) == 3
