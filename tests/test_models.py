"""Tests for protor.models."""

from __future__ import annotations

from protor.models import AnalysisResult, SiteManifest, SiteMetadata


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

    def test_round_trip(self, sample_manifest: SiteManifest):
        d = sample_manifest.to_dict()
        restored = SiteManifest.from_dict(d)
        assert restored.url == sample_manifest.url
        assert restored.domain == sample_manifest.domain
        assert restored.metadata.title == sample_manifest.metadata.title
        assert restored.bytes_received == sample_manifest.bytes_received

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
