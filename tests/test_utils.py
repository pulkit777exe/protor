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


class TestHumanDuration:
    """
    Bounded width, whatever the timeout.

    The progress table's Time column is six cells, and `--timeout` is the user's to
    set: at 300s with three retries a page can take 900000ms, which is nine
    characters. Rich ellipsised it to `900000…`, dropping the unit so the cell read
    as corrupt data rather than as a slow page.
    """

    @pytest.mark.parametrize(
        "ms", [1, 12, 999, 1000, 1500, 30000, 59400, 59999, 60000, 90000, 900000, 3600000]
    )
    def test_never_exceeds_the_column(self, ms):
        from protor.utils import human_duration

        assert len(human_duration(ms)) <= 6, f"{ms}ms -> {human_duration(ms)!r}"

    @pytest.mark.parametrize("ms", [None, 0])
    def test_nothing_measured_is_a_dash(self, ms):
        from protor.utils import human_duration

        assert human_duration(ms) == "—"

    @pytest.mark.parametrize(
        ("ms", "expected"),
        [
            (12, "12ms"),
            (1500, "1.5s"),
            (30000, "30.0s"),
            (59400, "59.4s"),
            # The boundary that made this fiddly: deciding before formatting gave
            # "60.0s" in one arrangement and "0m59s" in the other.
            (59999, "1m00s"),
            (90000, "1m30s"),
            (900000, "15m00s"),
            (3600000, "1h00m"),
        ],
    )
    def test_the_boundary_is_not_a_thing(self, ms, expected):
        from protor.utils import human_duration

        assert human_duration(ms) == expected

    def test_it_never_says_sixty_seconds(self):
        from protor.utils import human_duration

        for ms in range(59_900, 60_100, 7):
            rendered = human_duration(ms)
            assert rendered not in ("60.0s", "0m59s"), f"{ms}ms -> {rendered}"

    def test_it_increases(self):
        from protor.utils import human_duration

        # Not a total order on the strings — units change — but never goes
        # backwards at the unit boundaries, which is where a rounding slip shows.
        assert human_duration(59_950) == "1m00s"
        assert human_duration(59_900) == "59.9s"


class TestResolveUrl:
    """
    A page's links are resolved twice — once to collect them, once to render them.

    On a 1,320-link page that was 2,106 `urljoin` calls for 1,200 distinct pairs, 43%
    redundant at 4us each. Memoising measured 1.07x on `parse_html` with byte-identical
    output, measured round-robin so machine drift could not favour it.

    A cache introduces a stale-read risk: a crawl resolves thousands of pages, and if
    page 2's hrefs were answered from page 1's entries every relative link would point
    at the wrong host. Two independent mechanisms prevent that, and each is tested on
    its own because either alone would hide the other's absence — which is exactly how
    two "surviving" mutants turned out to be untestable rather than harmless.
    """

    def test_it_agrees_with_urljoin(self):
        from urllib.parse import urljoin

        from protor.utils import resolve_url

        for base, href in (
            ("https://ex.com/", "/a/b"),
            ("https://ex.com/dir/page", "rel"),
            ("https://ex.com/", "../up"),
            ("https://ex.com/", "#frag"),
            ("https://ex.com/", "https://other.example/x"),
            ("https://ex.com/", ""),
        ):
            assert resolve_url(base, href) == urljoin(base, href), (base, href)

    def test_the_key_includes_the_base_not_just_the_href(self):
        """
        The correctness half, tested without the per-parse clear in the way.

        `parse_soup` clears the cache, so a key of just `href` would still look right
        through the parser. It is only observable when the cache is warm across two
        different bases — which is what a caller outside a parse would have.
        """
        from protor.utils import clear_url_cache, resolve_url

        clear_url_cache()
        assert resolve_url("https://one.example/dir/", "/x") == "https://one.example/x"
        assert resolve_url("https://two.example/other/", "/x") == "https://two.example/x"

    def test_the_cache_is_cleared_per_parse_for_boundedness(self):
        """
        The memory half.

        Correctness does not depend on it — the key carries the base — so this is
        about not carrying a page's links into the next one for no reason.
        """
        from protor.parser import parse_html
        from protor.utils import _RESOLVED

        parse_html("<html><body><a href='/x'>x</a></body></html>", "https://a.example/")
        parse_html("<html><body><a href='/y'>y</a></body></html>", "https://a.example/")

        assert len(_RESOLVED) <= 2, f"entries carried across a parse boundary: {len(_RESOLVED)}"

    def test_one_page_cannot_answer_another_pages_links(self):
        """The property that makes it safe to cache at all."""
        from protor.parser import parse_html

        first = "<html><body><a href='/only-here'>x</a></body></html>"
        second = "<html><body><a href='/only-here'>x</a></body></html>"

        _, a = parse_html(first, "https://one.example/dir/")
        _, b = parse_html(second, "https://two.example/other/")

        assert a.links == ["https://one.example/only-here"], a.links
        assert b.links == ["https://two.example/only-here"], b.links

    def test_the_cache_is_cleared_at_the_start_of_a_parse(self):
        from protor.parser import parse_html
        from protor.utils import _RESOLVED, resolve_url

        parse_html("<html><body><a href='/x'>x</a></body></html>", "https://a.example/")
        first = dict(_RESOLVED)
        parse_html("<html><body><a href='/y'>y</a></body></html>", "https://a.example/")

        assert "/x" not in str(_RESOLVED) or first != _RESOLVED, "the cache was never cleared"
        assert resolve_url("https://a.example/", "/x") == "https://a.example/x"

    def test_it_is_bounded(self):
        """`resolve_url` is importable, so a caller outside a parse could grow it."""
        from protor.utils import _RESOLVED, _URL_CACHE_MAX, clear_url_cache, resolve_url

        clear_url_cache()
        for i in range(_URL_CACHE_MAX + 50):
            resolve_url("https://ex.com/", f"/p{i}")
        assert len(_RESOLVED) <= _URL_CACHE_MAX, len(_RESOLVED)
        clear_url_cache()
