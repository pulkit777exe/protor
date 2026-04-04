"""
Typed data models for protor.
All public structs are dataclasses so they're trivially serialisable,
comparable in tests, and self-documenting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SiteMetadata:
    title: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    author: str = ""
    og_tags: dict[str, str] = field(default_factory=dict)


@dataclass
class SiteManifest:
    url: str
    domain: str
    html_file: str
    metadata: SiteMetadata
    text_content: str
    js_files: list[str]
    js_count: int
    bytes_received: int
    elapsed_ms: int
    timestamp: str
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # flatten metadata for backwards-compatible JSON
        d["metadata"] = asdict(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SiteManifest:
        meta_raw = d.pop("metadata", {})
        meta = SiteMetadata(**{k: v for k, v in meta_raw.items() if k in SiteMetadata.__dataclass_fields__})
        # compat: old key name
        if "bytes" in d and "bytes_received" not in d:
            d["bytes_received"] = d.pop("bytes")
        return cls(
            metadata=meta,
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "metadata"},
        )


@dataclass
class AnalysisResult:
    model: str
    focus: str
    timestamp: str
    sites_analyzed: int
    analysis: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
