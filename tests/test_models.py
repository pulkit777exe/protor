"""Tests for protor.models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from protor.exceptions import ProtorError
from protor.models import AnalysisResult, InvalidManifestError, SiteManifest, SiteMetadata

if TYPE_CHECKING:
    from pathlib import Path


class TestSiteMetadata:
    def test_defaults(self):
        m = SiteMetadata()
        assert m.title == ""
        assert m.keywords == []
        assert m.og_tags == {}

    def test_with_values(self):
        m = SiteMetadata(title="My Site", keywords=["a", "b"])
        assert m.title == "My Site"
        assert len(m.keywords) == 2


class TestSiteManifest:
    def test_to_dict_has_expected_keys(self, sample_manifest: SiteManifest):
        d = sample_manifest.to_dict()
        assert "url" in d
        assert "domain" in d
        assert "metadata" in d
        assert "text_content" in d
        assert isinstance(d["metadata"], dict)

    def test_to_dict_nests_every_metadata_field(self, sample_manifest: SiteManifest):
        """`asdict` already recurses; the nested block must not lose a field."""
        assert sample_manifest.to_dict()["metadata"] == {
            "title": "Example Domain",
            "description": "Example description",
            "keywords": ["example", "test"],
            "author": "",
            "og_tags": {},
        }

    def test_to_dict_result_is_a_copy(self, sample_manifest: SiteManifest):
        """Editing the returned dict must not reach back into the manifest."""
        d = sample_manifest.to_dict()
        d["metadata"]["title"] = "mutated"
        d["js_files"].append("sneaky.js")
        assert sample_manifest.metadata.title == "Example Domain"
        assert sample_manifest.js_files == []

    def test_round_trip(self, sample_manifest: SiteManifest):
        d = sample_manifest.to_dict()
        restored = SiteManifest.from_dict(d)
        assert restored.url == sample_manifest.url
        assert restored.domain == sample_manifest.domain
        assert restored.metadata.title == sample_manifest.metadata.title
        assert restored.bytes_received == sample_manifest.bytes_received

    def test_round_trip_is_exact(self, sample_manifest: SiteManifest):
        """to_dict/from_dict must not quietly drop a field on the way back."""
        assert SiteManifest.from_dict(sample_manifest.to_dict()) == sample_manifest

    def test_from_dict_with_legacy_bytes_key(self, sample_manifest: SiteManifest):
        d = sample_manifest.to_dict()
        d["bytes"] = d.pop("bytes_received")
        restored = SiteManifest.from_dict(d)
        assert restored.bytes_received == sample_manifest.bytes_received

    def test_from_dict_ignores_extra_keys(self, sample_manifest: SiteManifest):
        d = sample_manifest.to_dict()
        d["unknown_future_field"] = "value"
        # Should not raise
        restored = SiteManifest.from_dict(d)
        assert restored.url == sample_manifest.url

    def test_from_dict_leaves_caller_dict_untouched(self):
        """A caller's dict is theirs: keys are neither dropped nor rewritten."""
        d = {
            "url": "https://example.com",
            "domain": "example.com",
            "html_file": "f.html",
            "metadata": {"title": "T"},
            "text_content": "body",
            "js_files": [],
            "js_count": 0,
            "bytes": 42,  # legacy key
            "elapsed_ms": 1,
            "timestamp": "ts",
            "future_field": "keep me",
        }
        snapshot = dict(d)

        restored = SiteManifest.from_dict(d)

        assert d == snapshot
        assert "metadata" in d
        assert "bytes" in d
        assert "bytes_received" not in d
        assert restored.bytes_received == 42

    def test_from_dict_accepts_partial_record(self):
        """A row short a measurement still describes a page, so it loads."""
        m = SiteManifest.from_dict({"url": "https://example.com", "domain": "example.com"})

        assert m.url == "https://example.com"
        assert m.metadata == SiteMetadata()
        assert m.js_files == []
        assert m.js_count == 0
        assert m.bytes_received == 0
        assert m.elapsed_ms == 0
        assert m.html_file == ""
        assert m.text_content == ""
        assert m.timestamp == ""

    def test_partial_records_do_not_share_default_lists(self):
        """Two defaulted manifests must not hand out the same js_files list."""
        a = SiteManifest.from_dict({"url": "https://a.example", "domain": "a.example"})
        b = SiteManifest.from_dict({"url": "https://b.example", "domain": "b.example"})

        a.js_files.append("a.js")
        assert b.js_files == []

    def test_from_dict_rejects_dict_that_is_not_a_manifest(self):
        with pytest.raises(InvalidManifestError) as excinfo:
            SiteManifest.from_dict({"note": "hi", "score": 3})

        assert excinfo.value.missing_fields == ("url", "domain")
        assert "url" in str(excinfo.value)
        assert "domain" in str(excinfo.value)

    def test_from_dict_rejects_missing_domain_only(self):
        with pytest.raises(InvalidManifestError) as excinfo:
            SiteManifest.from_dict({"url": "https://example.com"})

        assert excinfo.value.missing_fields == ("domain",)

    def test_from_dict_rejects_blank_url(self):
        """A blank URL attributes the record to no page, so it is not a manifest."""
        with pytest.raises(InvalidManifestError):
            SiteManifest.from_dict({"url": "", "domain": "example.com"})

    def test_manifest_error_is_a_protor_error(self):
        """Callers catch ProtorError; the new error has to be one of them."""
        with pytest.raises(ProtorError):
            SiteManifest.from_dict({})

    def test_manifest_error_names_the_source_file(self):
        with pytest.raises(InvalidManifestError) as excinfo:
            SiteManifest.from_dict({"url": "u"}, source="sites_index.json")

        assert excinfo.value.source == "sites_index.json"
        assert str(excinfo.value).startswith("sites_index.json: ")
        assert "domain" in str(excinfo.value)

    def test_manifest_error_accepts_a_path_source(self, tmp_path: Path):
        index = tmp_path / "sites_index.json"
        with pytest.raises(InvalidManifestError) as excinfo:
            SiteManifest.from_dict({"text_content": "orphan"}, source=index)

        assert str(index) in str(excinfo.value)

    def test_from_dict_rejects_non_dict_payload(self):
        """`json.loads` can hand back a list or a number; say what was wrong."""
        payload: Any = ["https://example.com"]

        with pytest.raises(InvalidManifestError) as excinfo:
            SiteManifest.from_dict(payload)

        assert "list" in str(excinfo.value)

    def test_from_dict_rejects_non_object_metadata(self):
        with pytest.raises(InvalidManifestError) as excinfo:
            SiteManifest.from_dict({"url": "u", "domain": "d", "metadata": "Example Domain"})

        assert excinfo.value.missing_fields == ("metadata",)
        assert "str" in str(excinfo.value)

    def test_from_dict_accepts_null_metadata(self, sample_manifest: SiteManifest):
        d = sample_manifest.to_dict()
        d["metadata"] = None

        assert SiteManifest.from_dict(d).metadata == SiteMetadata()

    def test_from_dict_drops_unknown_metadata_keys(self):
        m = SiteManifest.from_dict(
            {
                "url": "u",
                "domain": "d",
                "html_file": "f",
                "metadata": {"title": "T", "nope": 1},
                "text_content": "t",
                "js_files": [],
                "js_count": 0,
                "bytes_received": 1,
                "elapsed_ms": 1,
                "timestamp": "ts",
            }
        )
        assert m.metadata.title == "T"
        assert not hasattr(m.metadata, "nope")


class TestAnalysisResult:
    def test_to_dict(self):
        r = AnalysisResult(
            model="llama3",
            focus="general",
            timestamp="2024-01-01 00:00:00",
            sites_analyzed=2,
            analysis="## Overview\nTest analysis.",
        )
        d = r.to_dict()
        assert d["model"] == "llama3"
        assert d["sites_analyzed"] == 2
        assert "analysis" in d
