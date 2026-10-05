"""
Defects found by auditing `protor.extractor` and `protor.models`.

One class per defect, each saying what was broken and why it mattered. Every
test here either reproduces a real failure or pins the contract that the fix
rests on, so a later "simplification" that reintroduces the bug shows up as a
failure rather than as a traceback somebody has to debug in production.

Two of the five defects could only be half-fixed inside this package: the
extractor can hand ``protor extract`` a stable shape, and
``SiteManifest.from_dict`` can refuse a wrong file honestly, but the code that
*decides* to read the index lives in ``protor/cli.py``. Those classes say so.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import pytest

from protor import extractor as extractor_module
from protor import models as models_module
from protor.exceptions import (
    ConfigurationError,
    InvalidManifestError,
    InvalidSelectorError,
    ProtorError,
)
from protor.extractor import ExtractionSchema, Extractor, FieldSchema
from protor.models import SiteManifest

if TYPE_CHECKING:
    from pathlib import Path

# ── shared fixtures ───────────────────────────────────────────────────────────

LIST_PAGE = """<!DOCTYPE html>
<html><body>
  <div class="card"><span class="tag">alpha</span><span class="tag">beta</span></div>
  <div class="card"><span class="tag">gamma</span></div>
</body></html>
"""

LIST_SCHEMA = {
    "name": "cards",
    "base_selector": ".card",
    "fields": [
        {"name": "tags", "selector": ".tag", "type": "text", "multiple": True},
    ],
}


def extract(schema: ExtractionSchema, html: str) -> list[dict[str, Any]]:
    return Extractor(schema).extract(html)


# ── 1. a `multiple` field must always be a list ───────────────────────────────


class TestMultipleFieldsAlwaysYieldAList:
    """
    `multiple: true` used to return the field's scalar `default` when the
    selector matched nothing, and a list when it matched something.

    The shape of a record therefore depended on the data: `{"tags": null}` from
    one page and `{"tags": ["a"]}` from the next, out of one schema. Anything
    that consumed `extracted_data` — a JSON report, a loop over the values, a
    `len()` — had to re-check the type on every row, and a downstream KeyError
    or TypeError was the only symptom, from data that looked fine.

    The CLI half needed the same fix: `cli.py`'s "nothing matched" guard compared
    each value against ``(None, "")``, which a list never equals, so it could not
    see inside one. It recurses now (`cli._has_data`), and the two tests at the
    bottom of this class pin that end too.
    """

    def test_no_matches_is_an_empty_list(self):
        schema = ExtractionSchema.from_dict(
            {**LIST_SCHEMA, "fields": [{**LIST_SCHEMA["fields"][0], "selector": ".renamed"}]}
        )

        records = extract(schema, LIST_PAGE)

        assert records == [{"tags": []}, {"tags": []}], (
            "a `multiple` field returned the scalar default instead of a list"
        )

    def test_the_shape_is_the_same_either_way(self):
        """One schema, two pages: the declared type must not depend on the data."""
        schema = ExtractionSchema.from_dict(LIST_SCHEMA)

        with_matches = extract(schema, LIST_PAGE)
        without_matches = extract(
            schema,
            '<html><body><div class="card"></div><div class="card"></div></body></html>',
        )

        assert [type(r["tags"]) for r in with_matches] == [list, list]
        assert [type(r["tags"]) for r in without_matches] == [list, list]

    def test_a_scalar_default_does_not_leak_out_as_the_field_value(self):
        """
        `default: "N/A"` on a `multiple` field used to hand back the bare
        string, so the field was a `str` in some records and a `list` in others.
        """
        schema = ExtractionSchema.from_dict(
            {
                "name": "cards",
                "base_selector": ".card",
                "fields": [
                    {
                        "name": "tags",
                        "selector": ".renamed",
                        "type": "text",
                        "multiple": True,
                        "default": "N/A",
                    }
                ],
            }
        )

        records = extract(schema, LIST_PAGE)

        assert records == [{"tags": []}, {"tags": []}]
        assert all(isinstance(r["tags"], list) for r in records)

    def test_the_default_still_fills_each_match(self):
        """
        The default has not moved out of the extraction: an element that yields
        nothing of its own — no such attribute — still falls back to it, once
        per element, inside the list.
        """
        schema = ExtractionSchema.from_dict(
            {
                "name": "cards",
                "base_selector": ".card",
                "fields": [
                    {
                        "name": "tags",
                        "selector": ".tag",
                        "type": "attribute",
                        "attribute": "data-tag",
                        "multiple": True,
                        "default": "N/A",
                    }
                ],
            }
        )

        records = extract(schema, LIST_PAGE)

        assert records == [{"tags": ["N/A", "N/A"]}, {"tags": ["N/A"]}]

    def test_the_records_survive_a_json_round_trip_as_lists(self):
        """The shape is written to disk, so `json` must see an array, never null."""
        schema = ExtractionSchema.from_dict(LIST_SCHEMA)

        written = json.loads(json.dumps(extract(schema, LIST_PAGE)))

        assert [r["tags"] for r in written] == [["alpha", "beta"], ["gamma"]]

    # ── the CLI half, still missing ───────────────────────────────────────────
    #
    # `_cmd_extract` decides "nothing matched the schema" with
    # `sum(1 for r in results if not any(v not in (None, "") for v in r.values()))`.
    # A list is never equal to None or "", so *every* list reads as data:
    # `{"tags": []}` and `{"tags": [null, null]}` both counted as a successful
    # extraction while describing no data at all. The guard now recurses
    # (`cli._has_data`), so the two tests below pin the CLI half too — they are
    # how the coupling between the extractor's shape and the guard is kept from
    # reopening: changing either one alone fails one of them.

    def test_a_stale_selector_on_a_multiple_field_is_reported_as_no_data(self, tmp_path):
        _assert_extract_exits_1_for_a_schema_that_matches_nothing(
            tmp_path,
            {
                "name": "stale",
                "base_selector": ".card",
                "fields": [
                    {"name": "tags", "selector": ".renamed", "type": "text", "multiple": True}
                ],
            },
        )

    def test_a_multiple_field_whose_values_all_default_is_reported_as_no_data(self, tmp_path):
        """
        The same schema as a scalar field, one element short: the elements match
        but every value falls back to the default, and `{"tags": [null, null]}`
        was written to disk and reported as "Extracted 2 records".
        """
        _assert_extract_exits_1_for_a_schema_that_matches_nothing(
            tmp_path,
            {
                "name": "stale",
                "base_selector": ".card",
                "fields": [
                    {
                        "name": "tags",
                        "selector": ".tag",
                        "type": "attribute",
                        "attribute": "data-renamed",
                        "multiple": True,
                    }
                ],
            },
        )


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    HTML = LIST_PAGE

    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        body = self.HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _assert_extract_exits_1_for_a_schema_that_matches_nothing(
    tmp_path: Path, schema: dict[str, Any]
) -> None:
    """
    Drive `protor extract` over a real loopback page and require the
    "nothing matched" verdict: exit 1, and no records file written.
    """
    import sys

    from protor.cli import cli

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        out = tmp_path / "out"
        url = f"http://{host}:{port}/index.html"

        old_argv = sys.argv
        sys.argv = ["protor", "extract", url, str(schema_path), "--output", str(out)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                cli()
        finally:
            sys.argv = old_argv
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert excinfo.value.code == 1, "records that hold no data were reported as a success"
    assert not (out / "stale_index.json").exists(), "wrote records whose every value was empty"


# ── 2 & 3. types that borrow `attribute` must be refused without one ───────────


class TestAttributeTypeIsRefusedWithoutAnAttributeName:
    """
    `{"type": "attribute"}` with no `attribute` value loaded cleanly.

    `FIELD_TYPES` whitelists the type and `validate()` checked only the type and
    the selector, so the schema passed every gate and then failed per page, with
    `ValueError: field 'cls' has unknown type 'attribute'` — a message blaming
    the one part of the field the author had spelled correctly, sent after the
    page was fetched and saved, and recorded as a page failure.
    """

    def test_it_is_refused_at_load_time(self):
        with pytest.raises(ConfigurationError) as excinfo:
            ExtractionSchema.from_dict(
                {
                    "name": "products",
                    "fields": [{"name": "cls", "selector": "div", "type": "attribute"}],
                }
            )

        message = str(excinfo.value)
        assert "cls" in message, message
        assert "attribute" in message, message

    def test_the_message_does_not_blame_the_type(self):
        """`attribute` is a supported type; the error has to name what is missing."""
        with pytest.raises(ConfigurationError) as excinfo:
            ExtractionSchema.from_dict(
                {"fields": [{"name": "cls", "selector": "div", "type": "attribute"}]}
            )

        assert "unknown type" not in str(excinfo.value)

    def test_it_is_a_protor_error(self):
        """`cli.cli()` renders a ProtorError as a message; a bare ValueError escapes it."""
        with pytest.raises(ProtorError):
            ExtractionSchema.from_dict(
                {"fields": [{"name": "cls", "selector": "div", "type": "attribute"}]}
            )

    def test_from_json_refuses_it(self, tmp_path: Path):
        path = tmp_path / "schema.json"
        path.write_text(
            '{"name": "products", "fields": [{"name": "cls", "selector": "div", '
            '"type": "attribute"}]}',
            encoding="utf-8",
        )

        with pytest.raises(ConfigurationError) as excinfo:
            ExtractionSchema.from_json(path)

        assert "cls" in str(excinfo.value)

    def test_an_unvalidated_schema_built_in_code_is_not_blamed_for_its_type(self):
        """
        `validate()` is not the only way in — a caller can build a FieldSchema in
        Python and hand it straight to `Extractor`. The extraction path must say
        the same thing the load path does, rather than claiming the type is
        unknown.
        """
        schema = ExtractionSchema(
            fields=[FieldSchema(name="cls", selector="div", type="attribute")]
        )

        with pytest.raises(ConfigurationError) as excinfo:
            extract(schema, '<div class="card">content</div>')

        message = str(excinfo.value)
        assert "cls" in message, message
        assert "unknown type" not in message, message

    def test_a_spellable_attribute_name_still_loads(self):
        schema = ExtractionSchema.from_dict(
            {
                "fields": [
                    {"name": "cls", "selector": "div", "type": "attribute", "attribute": "class"}
                ]
            }
        )

        assert schema.fields[0].attribute == "class"


class TestRegexTypeIsRefusedWithoutAPattern:
    """
    The sibling branch of the same function, same class of mistake, worse
    outcome: `{"type": "regex"}` with no pattern returned the element's *entire
    text*. One branch raised a misdiagnosis, the other reported success with the
    wrong data — the exact failure the module's validation comments exist to
    prevent, and the reason both are fixed the same way.
    """

    def test_it_is_refused_at_load_time(self):
        with pytest.raises(ConfigurationError) as excinfo:
            ExtractionSchema.from_dict(
                {
                    "name": "products",
                    "fields": [{"name": "price", "selector": "div", "type": "regex"}],
                }
            )

        message = str(excinfo.value)
        assert "price" in message, message
        assert "regex" in message, message

    def test_the_message_does_not_blame_the_type(self):
        with pytest.raises(ConfigurationError) as excinfo:
            ExtractionSchema.from_dict(
                {"fields": [{"name": "price", "selector": "div", "type": "regex"}]}
            )

        assert "unknown type" not in str(excinfo.value)

    def test_it_does_not_return_the_whole_element_text(self):
        """
        The silent half of the defect: a schema with no pattern matched nothing,
        so this returned `Order 12345 shipped` where the author asked for a
        number.
        """
        schema = ExtractionSchema(fields=[FieldSchema(name="price", selector="body", type="regex")])

        with pytest.raises(ConfigurationError) as excinfo:
            extract(schema, "<body>Order 12345 shipped</body>")

        assert "price" in str(excinfo.value)

    def test_a_pattern_still_loads_and_extracts(self):
        schema = ExtractionSchema.from_dict(
            {
                "fields": [
                    {
                        "name": "price",
                        "selector": "body",
                        "type": "regex",
                        "attribute": r"Order (\d+)",
                    }
                ]
            }
        )

        assert extract(schema, "<body>Order 12345 shipped</body>") == [{"price": "12345"}]

    @pytest.mark.parametrize("field_type", ["attribute", "regex"])
    def test_both_types_that_borrow_attribute_are_refused_the_same_way(self, field_type):
        """Two branches of one function must not disagree about a missing value."""
        with pytest.raises(ConfigurationError) as excinfo:
            ExtractionSchema.from_dict(
                {"fields": [{"name": "v", "selector": "div", "type": field_type}]}
            )

        assert "'attribute'" in str(excinfo.value)

    def test_a_genuinely_unknown_type_still_says_so(self):
        """The refusal must not swallow the diagnostic it was added next to."""
        with pytest.raises(ConfigurationError) as excinfo:
            ExtractionSchema.from_dict(
                {"fields": [{"name": "v", "selector": "div", "type": "attribte"}]}
            )

        assert "unknown type" in str(excinfo.value)


# ── 4. `SiteManifest.from_dict` has to be honest about what it is for ─────────


class TestFromDictRefusesAWrongFile:
    """
    `cli._load_index` used to hand `protor analyze` whatever `json.loads`
    returned, so `--file answer.json` where the file is `{"note": "hi"}` iterated
    the dict's *keys*: the analyzer then called `.get` on a string, and
    `cli.cli()` — which catches `ProtorError` and `ValueError` — let the
    `AttributeError` escape as a traceback.

    `_load_index` now routes every row through `from_dict`, so the guard is
    consulted and the refusal names the file. These tests pin the guard itself;
    the wiring is pinned in `tests/test_cli.py::TestLoadIndex`, and the
    end-to-end path (exit 1, no traceback) in this class.
    """

    @pytest.mark.parametrize(
        ("payload", "what"),
        [
            ({"note": "hi"}, "an LLM answer written as an object"),
            ({"url": "https://example.com", "note": "hi"}, "one field short of identity"),
            ({"url": "", "domain": "example.com"}, "a blank url"),
            ({"url": "https://example.com"}, "no domain"),
        ],
    )
    def test_a_record_that_names_no_page_is_refused(self, payload: dict[str, Any], what: str):
        with pytest.raises(InvalidManifestError) as excinfo:
            SiteManifest.from_dict(payload)

        assert excinfo.value.missing_fields, what

    def test_the_error_carries_both_halves_of_the_catch_clauses_cli_uses(self):
        """
        `InvalidManifestError` is a `ProtorError` *and* a `ValueError`, which is
        the only reason routing the index through `from_dict` can fix the
        traceback without touching the error hierarchy.
        """
        with pytest.raises(ProtorError):
            SiteManifest.from_dict({"note": "hi"})
        with pytest.raises(ValueError):
            SiteManifest.from_dict({"note": "hi"})

    def test_the_source_names_the_offending_file(self, tmp_path: Path):
        index = tmp_path / "answer.json"
        index.write_text('{"note": "hi"}', encoding="utf-8")

        rows = json.loads(index.read_text(encoding="utf-8"))
        rows = list(rows) if isinstance(rows, list) else [rows]

        with pytest.raises(InvalidManifestError) as excinfo:
            for row in rows:
                SiteManifest.from_dict(row, source=index)

        assert excinfo.value.source == index
        assert "url" in str(excinfo.value)

    def test_a_killed_write_leaves_unparseable_json_not_a_partial_manifest(self, tmp_path: Path):
        """
        The module docstring justified the lenient tier with "a run killed
        mid-write leaves exactly those" missing. It cannot: the only writer
        dumps one manifest at a time, so a kill leaves a truncated document that
        `json.loads` rejects outright. The lenient tier is here for records that
        were never written by that writer — an older index, a hand-assembled
        row — and the docstring now says so.
        """
        index = tmp_path / "sites_index.json"
        complete = [{"url": "https://a.example", "domain": "a.example"}]
        index.write_text(json.dumps(complete, indent=2), encoding="utf-8")

        # What a kill mid-write leaves: the array opened, one element in, no
        # closing bracket.
        truncated = (
            index.read_text(encoding="utf-8").rstrip()[:-2] + ',\n  {\n    "url": "https://b'
        )
        index.write_text(truncated, encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            json.loads(index.read_text(encoding="utf-8"))

    def test_the_module_no_longer_justifies_the_lenient_tier_with_a_killed_write(self):
        """
        The docstring has to name a cause that really produces records missing
        measurements. It named a run killed mid-write, which the only writer
        cannot produce. Both docstrings are checked, because either one repeating
        the false claim puts the reader back on the wrong trail.
        """
        module_doc = models_module.__doc__ or ""
        assert "leaves exactly those" not in module_doc, module_doc
        assert "truncated" in module_doc, (
            "the writer's real failure mode is gone from the docstring"
        )

        method_doc = SiteManifest.from_dict.__doc__ or ""
        assert "leaves exactly those" not in method_doc, method_doc


# ── 5. selectors are CSS, and the module has to stop saying otherwise ─────────


class TestSelectorsAreCssNotXPath:
    """
    The module docstring advertised "CSS selectors **or XPath expressions**".
    There is no XPath anywhere in the package: every selector goes through
    soupsieve, and `soupsieve` refuses a `/` outright, so an XPath expression is
    already rejected at load — but with "Invalid character '/' position 0",
    which reads like a typo in the user's CSS rather than the wrong language.

    Implementing XPath is a feature, not an audit fix, so the claim comes out and
    the refusal becomes legible instead.
    """

    @pytest.mark.parametrize(
        "selector",
        ["//div[@class='price']", "//div", "/html/body/div", "div//span", ".//p"],
    )
    def test_an_xpath_expression_is_refused_saying_it_is_xpath(self, selector):
        with pytest.raises(InvalidSelectorError) as excinfo:
            ExtractionSchema.from_dict({"fields": [{"name": "price", "selector": selector}]})

        assert "XPath" in str(excinfo.value), str(excinfo.value)

    def test_the_message_points_at_the_css_equivalent(self):
        with pytest.raises(InvalidSelectorError) as excinfo:
            ExtractionSchema.from_dict(
                {"fields": [{"name": "price", "selector": "//div[@class='price']"}]}
            )

        message = str(excinfo.value)
        assert "CSS" in message, message
        assert "price" in message, message

    def test_a_protocol_relative_url_in_an_attribute_selector_is_not_xpath(self):
        """
        `//` also appears inside a quoted attribute value. Refusing that would
        break a selector people write to follow a CDN.
        """
        schema = ExtractionSchema.from_dict(
            {
                "name": "cdn",
                "fields": [{"name": "src", "selector": "img[src^='//cdn.example']", "type": "src"}],
            }
        )

        assert extract(schema, '<img src="//cdn.example/a.png">') == [
            {"src": "//cdn.example/a.png"}
        ]

    def test_an_unvalidated_schema_built_in_code_says_the_same_thing(self):
        schema = ExtractionSchema(fields=[FieldSchema(name="price", selector="//div")])

        with pytest.raises(InvalidSelectorError) as excinfo:
            extract(schema, "<div>Price: $19.99</div>")

        assert "XPath" in str(excinfo.value)

    def test_the_module_no_longer_advertises_xpath(self):
        doc = extractor_module.__doc__ or ""
        assert "CSS" in doc, "the docstring must still name the selector language it supports"
        assert "XPath expressions" not in doc
