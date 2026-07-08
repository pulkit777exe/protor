"""
protor.extractor
~~~~~~~~~~~~~~~~
Schema-based structured data extraction from HTML.

Inspired by Firecrawl's Pydantic schema extraction and AutoScraper's
pattern-learning approach. Extracts structured JSON from HTML using
CSS selectors or XPath expressions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

__all__ = ["ExtractionSchema", "Extractor", "FieldSchema", "extract_from_html"]


@dataclass
class FieldSchema:
    """
    Schema for a single extraction field.

    Parameters
    ----------
    name:
        Output key name.
    selector:
        CSS selector to find the element.
    type:
        Extraction type: "text", "html", "attribute", "href", "src".
    attribute:
        Attribute name when type is "attribute".
    multiple:
        If True, extract all matches as a list.
    default:
        Default value when no match is found.
    """

    name: str
    selector: str
    type: str = "text"
    attribute: str = ""
    multiple: bool = False
    default: Any = None


@dataclass
class ExtractionSchema:
    """
    Schema definition for structured data extraction.

    Parameters
    ----------
    name:
        Name of the extraction schema.
    base_selector:
        Optional base CSS selector to scope all field selectors.
    fields:
        List of field schemas to extract.
    """

    name: str = "extraction"
    base_selector: str = ""
    fields: list[FieldSchema] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtractionSchema:
        """Create schema from a dictionary (e.g., JSON config)."""
        fields = []
        for f in d.get("fields", []):
            if isinstance(f, dict):
                fields.append(
                    FieldSchema(
                        name=f.get("name", "unknown"),
                        selector=f.get("selector", ""),
                        type=f.get("type", "text"),
                        attribute=f.get("attribute", ""),
                        multiple=f.get("multiple", False),
                        default=f.get("default"),
                    )
                )
            elif isinstance(f, FieldSchema):
                fields.append(f)
        return cls(
            name=d.get("name", "extraction"),
            base_selector=d.get("base_selector", ""),
            fields=fields,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ExtractionSchema:
        """Load schema from a JSON file."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_selector": self.base_selector,
            "fields": [
                {
                    "name": f.name,
                    "selector": f.selector,
                    "type": f.type,
                    "attribute": f.attribute,
                    "multiple": f.multiple,
                    "default": f.default,
                }
                for f in self.fields
            ],
        }


def _extract_field_value(element: Tag, field: FieldSchema, base_url: str) -> Any:
    """Extract a single value from an element based on field config."""
    if field.type == "text":
        return element.get_text(strip=True)
    elif field.type == "html":
        return str(element)
    elif field.type == "attribute" and field.attribute:
        val = element.get(field.attribute, field.default)
        return str(val) if val is not None else field.default
    elif field.type == "href":
        href = element.get("href", "")
        return urljoin(base_url, href) if href else field.default
    elif field.type == "src":
        src = element.get("src", "")
        return urljoin(base_url, src) if src else field.default
    elif field.type == "regex":
        text = element.get_text(strip=True)
        pattern = field.attribute  # reuse attribute for regex pattern
        if pattern:
            match = re.search(pattern, text)
            return match.group(1) if match else field.default
        return text
    return element.get_text(strip=True)


class Extractor:
    """
    Schema-based data extractor.

    Usage::

        schema = ExtractionSchema(
            name="products",
            base_selector=".product-card",
            fields=[
                FieldSchema(name="title", selector="h2", type="text"),
                FieldSchema(name="price", selector=".price", type="text"),
                FieldSchema(name="link", selector="a", type="href"),
                FieldSchema(name="image", selector="img", type="src"),
            ]
        )
        extractor = Extractor(schema)
        results = extractor.extract(html)

    Parameters
    ----------
    schema:
        ExtractionSchema defining what to extract.
    base_url:
        Base URL for resolving relative links.
    """

    def __init__(self, schema: ExtractionSchema, base_url: str = "") -> None:
        self.schema = schema
        self.base_url = base_url

    def extract(self, html: str) -> list[dict[str, Any]]:
        """
        Extract structured data from HTML.

        Returns a list of dicts, one per matched base element.
        If no base_selector is set, extracts one record from the whole page.
        """
        soup = BeautifulSoup(html, "lxml")
        results: list[dict[str, Any]] = []

        containers = soup.select(self.schema.base_selector) if self.schema.base_selector else [soup]

        for container in containers:
            record: dict[str, Any] = {}
            for f in self.schema.fields:
                try:
                    elements = container.select(f.selector)
                except Exception:
                    elements = []

                if not elements:
                    record[f.name] = f.default
                    continue

                if f.multiple:
                    record[f.name] = [_extract_field_value(el, f, self.base_url) for el in elements]
                else:
                    record[f.name] = _extract_field_value(elements[0], f, self.base_url)

            results.append(record)

        return results

    def extract_from_file(self, path: str | Path) -> list[dict[str, Any]]:
        """Extract from a local HTML file."""
        p = Path(path)
        html = p.read_text(encoding="utf-8")
        return self.extract(html)


def extract_from_html(
    html: str,
    schema: ExtractionSchema,
    base_url: str = "",
) -> list[dict[str, Any]]:
    """
    Convenience function: extract structured data from HTML using a schema.
    """
    extractor = Extractor(schema, base_url)
    return extractor.extract(html)
