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
from soupsieve import SelectorSyntaxError
from soupsieve import compile as compile_selector

from .exceptions import ConfigurationError, InvalidSelectorError

__all__ = [
    "ExtractionSchema",
    "Extractor",
    "FieldSchema",
    "InvalidSelectorError",
    "extract_from_html",
    "extract_from_soup",
]


#: The extraction types a schema field may declare. Validated on load so a typo
#: is refused rather than quietly falling through to plain text.
FIELD_TYPES = frozenset({"text", "html", "attribute", "href", "src", "regex"})


def _compile_checked(selector: str, *, where: str) -> None:
    """
    Compile *selector* now so a typo is reported once, naming its field.

    soupsieve's reason is multi-line (selector echo, caret); keep the first line
    so the message stays one line when a CLI prints it.
    """
    try:
        compile_selector(selector)
    except SelectorSyntaxError as exc:
        raise InvalidSelectorError(selector, where=where, reason=str(exc).splitlines()[0]) from exc


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
        Extraction type. One of ``text``, ``html``, ``attribute``, ``href``,
        ``src`` or ``regex``.
    attribute:
        Attribute name when type is ``attribute``; the pattern when it is
        ``regex``, which reuses this field rather than adding another.
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
        """
        Create schema from a dictionary (e.g., JSON config).

        Selectors are compiled here, so a schema file with a typo is rejected
        before it is applied to a single page.
        """
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
        schema = cls(
            name=d.get("name", "extraction"),
            base_selector=d.get("base_selector", ""),
            fields=fields,
        )
        schema.validate()
        return schema

    @classmethod
    def from_json(cls, path: str | Path) -> ExtractionSchema:
        """Load schema from a JSON file."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def validate(self) -> None:
        """
        Compile every selector, raising `InvalidSelectorError` on the first bad one.

        Called once at load time: waiting until extraction would repeat the same
        diagnostic for every page of every run, and an empty `base_selector`
        means "whole page" rather than a typo, so it is only checked when set.
        """
        if self.base_selector:
            _compile_checked(self.base_selector, where=f"base_selector in schema {self.name!r}")
        for f in self.fields:
            where = f"field {f.name!r} in schema {self.name!r}"
            _compile_checked(f.selector, where=where)
            if f.type not in FIELD_TYPES:
                # Checked here for the same reason as the selector: a typo like
                # "hrefs" used to fall through every branch and quietly extract
                # the element's *text* instead, so the run reported success with
                # the wrong data — a link's URL replaced by its label.
                # ConfigurationError, not InvalidSelectorError: the selector
                # compiled fine, and its message would have said otherwise. It is
                # also what the CLI renders with a hint, which is the right
                # outcome for a schema that cannot work.
                raise ConfigurationError(
                    f"{where}: unknown type {f.type!r}; "
                    f"expected one of {', '.join(sorted(FIELD_TYPES))}"
                )

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
        if val is None:
            return field.default
        # bs4 returns a *list* for multi-valued attributes (class, rel, headers,
        # accept-charset, accesskey, dropzone), and str() on that wrote a Python
        # repr into the JSON: "['product-title']". Join them instead.
        if isinstance(val, list):
            return " ".join(str(v) for v in val)
        return str(val)
    elif field.type == "href":
        href = element.get("href", "")
        return urljoin(base_url, str(href)) if href else field.default
    elif field.type == "src":
        src = element.get("src", "")
        return urljoin(base_url, str(src)) if src else field.default
    elif field.type == "regex":
        text = element.get_text(strip=True)
        pattern = field.attribute  # reuse attribute for regex pattern
        if pattern:
            match = re.search(pattern, text)
            if match is None:
                return field.default
            # A pattern with no capture group raised IndexError out of the
            # extractor — a traceback after the page was already fetched, and in
            # `scrape --schema` a recorded page failure. Fall back to the whole
            # match, which is what an author who wrote no group almost always
            # meant.
            return match.group(1) if match.re.groups else match.group(0)
        return text
    raise ValueError(f"field {field.name!r} has unknown type {field.type!r}")


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

    def _select(self, container: Tag | BeautifulSoup, selector: str, *, where: str) -> list[Tag]:
        """
        Run one selector, re-raising a syntax error as `InvalidSelectorError`.

        Schemas loaded from JSON are validated up front, so this only fires for
        a schema built in code -- still better than the old blanket `except`,
        which turned a typo into a field that was quietly ``None`` everywhere.
        """
        try:
            return container.select(selector)
        except SelectorSyntaxError as exc:
            raise InvalidSelectorError(
                selector, where=where, reason=str(exc).splitlines()[0]
            ) from exc

    def extract(self, html: str) -> list[dict[str, Any]]:
        """Extract structured data from an HTML string."""
        return self.extract_from_soup(BeautifulSoup(html, "lxml"))

    def extract_from_soup(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """
        Extract structured data from an already-parsed tree.

        Callers that have parsed the page (e.g. the crawl engine) should use
        this to avoid paying for a second lxml parse.

        Returns a list of dicts, one per matched base element.
        If no base_selector is set, extracts one record from the whole page.
        """
        results: list[dict[str, Any]] = []

        containers: list[Tag | BeautifulSoup]
        if self.schema.base_selector:
            containers = self._select(
                soup,
                self.schema.base_selector,
                where=f"base_selector in schema {self.schema.name!r}",
            )
        else:
            containers = [soup]

        for container in containers:
            record: dict[str, Any] = {}
            for f in self.schema.fields:
                elements = self._select(
                    container, f.selector, where=f"field {f.name!r} in schema {self.schema.name!r}"
                )

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


def extract_from_soup(
    soup: BeautifulSoup,
    schema: ExtractionSchema,
    base_url: str = "",
) -> list[dict[str, Any]]:
    """
    Extract structured data from an already-parsed tree.

    Use this when the HTML has already been parsed, so the page is not parsed
    twice.
    """
    return Extractor(schema, base_url).extract_from_soup(soup)
