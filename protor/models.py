"""
Typed data models for protor.
All public structs are dataclasses so they're trivially serialisable,
comparable in tests, and self-documenting.

Reading a manifest back is deliberately two-tier. A record short a *measurement*
-- a run killed mid-write leaves exactly those -- loads with that measurement
empty, because the page it describes is still worth keeping. A record with no
``url``/``domain`` names no page at all, so it raises `InvalidManifestError`
listing what it lacked. Tolerating that too would turn a wrong file (an LLM
response, a list of URLs) into a wall of empty manifests that look valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

from .exceptions import InvalidManifestError

if TYPE_CHECKING:
    from pathlib import Path

#: Identity a dict must carry to be read as a manifest. Every other field is a
#: measurement of a page, so it defaults to empty when the record lacks it.
_MANIFEST_IDENTITY = ("url", "domain")


@dataclass
class SiteMetadata:
    """Page metadata harvested from ``<head>``."""

    title: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    author: str = ""
    og_tags: dict[str, str] = field(default_factory=dict)


@dataclass
class SiteManifest:
    """Everything recorded about one scraped page."""

    url: str = ""
    domain: str = ""
    html_file: str = ""
    metadata: SiteMetadata = field(default_factory=SiteMetadata)
    text_content: str = ""
    js_files: list[str] = field(default_factory=list)
    #: Kept in the saved JSON because `protor analyze` reads the index back as
    #: plain dicts. Drop it when the index format moves.
    js_count: int = 0
    markdown_content: str = ""
    bytes_received: int = 0
    elapsed_ms: int = 0
    timestamp: str = ""
    #: Never False today — engine only appends a manifest after a good fetch.
    #: Don't branch on it; use the absence of a manifest instead.
    success: bool = True
    #: Write-only in Python, but the only place `scrape --schema` output lands
    #: on disk, so it cannot be dropped without losing the schema feature.
    extracted_data: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise for JSON.

        ``asdict`` already recurses into nested dataclasses, so the metadata
        block comes back as a dict without being flattened by hand.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any, *, source: str | Path | None = None) -> SiteManifest:
        """
        Rebuild a manifest from *d*, never mutating it.

        Identity (``url``, ``domain``) is required: a record without it names no
        page, and tolerating that would turn a wrong file into a wall of empty
        manifests that look valid. Measurements default, because a run killed
        mid-write leaves exactly those missing and the page is still worth
        keeping. Raises :class:`InvalidManifestError` naming what was missing.
        """
        if not isinstance(d, dict):
            where = f"{source}: " if source is not None else ""
            raise InvalidManifestError(
                missing_fields=_MANIFEST_IDENTITY,
                source=source,
                detail=f"{where}expected a manifest object, got {type(d).__name__}",
            )

        data = dict(d)  # never pop from the caller's dict
        missing = tuple(
            key for key in _MANIFEST_IDENTITY if not str(data.get(key, "") or "").strip()
        )
        if missing:
            raise InvalidManifestError(missing_fields=missing, source=source)

        # Indexes written before the rename called it "bytes".
        if "bytes" in data and "bytes_received" not in data:
            data["bytes_received"] = data.pop("bytes")
        else:
            data.pop("bytes", None)

        raw_meta = data.pop("metadata", None)
        if raw_meta is not None and not isinstance(raw_meta, dict):
            where = f"{source}: " if source is not None else ""
            raise InvalidManifestError(
                missing_fields=("metadata",),
                source=source,
                detail=f"{where}metadata must be an object, got {type(raw_meta).__name__}",
            )
        # Unknown metadata keys are dropped rather than raising: an index written
        # by a newer protor must still load in an older one.
        meta_fields = {f.name for f in fields(SiteMetadata)}
        data["metadata"] = SiteMetadata(
            **{k: v for k, v in (raw_meta or {}).items() if k in meta_fields}
        )

        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class AnalysisResult:
    """The report produced by one analysis run."""

    model: str
    focus: str
    timestamp: str
    sites_analyzed: int
    analysis: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
