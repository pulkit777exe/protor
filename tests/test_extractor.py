from pathlib import Path

import pytest

from protor.exceptions import ProtorError
from protor.extractor import (
    ExtractionSchema,
    Extractor,
    FieldSchema,
    InvalidSelectorError,
)


def test_extract_text():
    schema = ExtractionSchema(fields=[FieldSchema(name="title", selector="h1", type="text")])
    ext = Extractor(schema)
    result = ext.extract("<html><body><h1>Hello World</h1></body></html>")
    assert result[0]["title"] == "Hello World"


def test_extract_href():
    schema = ExtractionSchema(fields=[FieldSchema(name="url", selector="a", type="href")])
    ext = Extractor(schema)
    result = ext.extract('<a href="https://example.com">link</a>')
    assert result[0]["url"] == "https://example.com"


def test_extract_image_src():
    schema = ExtractionSchema(fields=[FieldSchema(name="img", selector="img", type="src")])
    ext = Extractor(schema)
    result = ext.extract('<img src="photo.jpg" alt="pic">')
    assert result[0]["img"] == "photo.jpg"


def test_extract_html():
    schema = ExtractionSchema(fields=[FieldSchema(name="content", selector="div", type="html")])
    ext = Extractor(schema)
    result = ext.extract("<div><b>bold</b></div>")
    assert "<b>bold</b>" in result[0]["content"]


def test_extract_regex():
    schema = ExtractionSchema(
        fields=[
            FieldSchema(
                name="price",
                selector="body",
                type="regex",
                attribute=r"\$(\d+\.\d{2})",
            )
        ]
    )
    ext = Extractor(schema)
    result = ext.extract("<body>Price: $19.99</body>")
    assert result[0]["price"] == "19.99"


def test_extract_multiple_fields():
    schema = ExtractionSchema(
        fields=[
            FieldSchema(name="title", selector="h1", type="text"),
            FieldSchema(name="desc", selector="p", type="text"),
        ]
    )
    ext = Extractor(schema)
    html = "<h1>Title</h1><p>Description</p>"
    result = ext.extract(html)
    assert result[0]["title"] == "Title"
    assert result[0]["desc"] == "Description"


def test_extract_all_matches():
    schema = ExtractionSchema(
        fields=[FieldSchema(name="links", selector="a", type="href", multiple=True)]
    )
    ext = Extractor(schema)
    html = '<a href="a.html">A</a><a href="b.html">B</a>'
    result = ext.extract(html)
    assert result[0]["links"] == ["a.html", "b.html"]


def test_extract_not_found():
    schema = ExtractionSchema(
        fields=[FieldSchema(name="missing", selector=".nonexistent", type="text")]
    )
    ext = Extractor(schema)
    result = ext.extract("<p>no match</p>")
    assert result[0]["missing"] is None


def test_extract_attribute():
    schema = ExtractionSchema(
        fields=[FieldSchema(name="cls", selector="div", type="attribute", attribute="class")]
    )
    ext = Extractor(schema)
    result = ext.extract('<div class="special">content</div>')
    assert "special" in str(result[0]["cls"])


def test_extract_attribute_missing():
    schema = ExtractionSchema(
        fields=[FieldSchema(name="data", selector="div", type="attribute", attribute="data-x")]
    )
    ext = Extractor(schema)
    result = ext.extract("<div>content</div>")
    assert result[0]["data"] is None


def test_empty_html():
    schema = ExtractionSchema(fields=[FieldSchema(name="title", selector="h1", type="text")])
    ext = Extractor(schema)
    result = ext.extract("")
    assert result[0]["title"] is None


def test_regex_no_match():
    schema = ExtractionSchema(
        fields=[
            FieldSchema(
                name="price",
                selector="body",
                type="regex",
                attribute=r"\$(\d+\.\d{2})",
            )
        ]
    )
    ext = Extractor(schema)
    result = ext.extract("<body>no price here</body>")
    assert result[0]["price"] is None


def test_complex_schema():
    schema = ExtractionSchema(
        fields=[
            FieldSchema(name="title", selector="h1", type="text"),
            FieldSchema(name="author", selector=".author", type="text"),
            FieldSchema(
                name="date",
                selector="time",
                type="attribute",
                attribute="datetime",
            ),
            FieldSchema(name="tags", selector=".tag", type="text", multiple=True),
        ]
    )
    ext = Extractor(schema)
    html = """
    <article>
        <h1>My Post</h1>
        <span class="author">Jane</span>
        <time datetime="2024-01-15">Jan 15</time>
        <span class="tag">python</span>
        <span class="tag">web</span>
    </article>
    """
    result = ext.extract(html)
    assert result[0]["title"] == "My Post"
    assert result[0]["author"] == "Jane"
    assert result[0]["date"] == "2024-01-15"
    assert result[0]["tags"] == ["python", "web"]


def test_from_dict():
    d = {
        "fields": [
            {"name": "title", "selector": "h1", "type": "text"},
        ]
    }
    schema = ExtractionSchema.from_dict(d)
    assert len(schema.fields) == 1
    assert schema.fields[0].name == "title"


def test_to_dict():
    schema = ExtractionSchema(fields=[FieldSchema(name="title", selector="h1", type="text")])
    d = schema.to_dict()
    assert d["fields"][0]["name"] == "title"
    assert d["fields"][0]["type"] == "text"


def test_field_default_value():
    schema = ExtractionSchema(
        fields=[FieldSchema(name="missing", selector=".nope", type="text", default="N/A")]
    )
    ext = Extractor(schema)
    result = ext.extract("<p>text</p>")
    assert result[0]["missing"] == "N/A"


def test_text_all_matches():
    schema = ExtractionSchema(
        fields=[FieldSchema(name="items", selector="li", type="text", multiple=True)]
    )
    ext = Extractor(schema)
    html = "<ul><li>a</li><li>b</li><li>c</li></ul>"
    result = ext.extract(html)
    assert result[0]["items"] == ["a", "b", "c"]


# ── bad selectors must fail loudly, not extract nothing ──────────────────────


def test_bad_selector_names_the_field():
    """A typo used to yield `None` for every record and report a good scrape."""
    schema = ExtractionSchema(
        name="products",
        fields=[FieldSchema(name="price", selector="[[[bad")],
    )

    with pytest.raises(InvalidSelectorError) as excinfo:
        Extractor(schema).extract("<div>Price: $19.99</div>")

    err = excinfo.value
    assert err.selector == "[[[bad"
    assert err.where == "field 'price' in schema 'products'"
    assert "price" in str(err)
    assert "[[[bad" in str(err)


def test_bad_selector_is_a_protor_error():
    schema = ExtractionSchema(fields=[FieldSchema(name="price", selector="[[[bad")])
    with pytest.raises(ProtorError):
        Extractor(schema).extract("<div>x</div>")


def test_bad_selector_on_a_valid_schema_still_raises():
    """Only the broken field fails; the error must not name a healthy one."""
    schema = ExtractionSchema(
        fields=[
            FieldSchema(name="title", selector="h1"),
            FieldSchema(name="price", selector="h1 >>>"),
        ]
    )
    with pytest.raises(InvalidSelectorError) as excinfo:
        Extractor(schema).extract("<h1>Title</h1>")
    assert "price" in excinfo.value.where


def test_bad_base_selector_names_the_schema():
    schema = ExtractionSchema(
        name="listing",
        base_selector=".card >>>",
        fields=[FieldSchema(name="title", selector="h2")],
    )
    with pytest.raises(InvalidSelectorError) as excinfo:
        Extractor(schema).extract("<div class='card'><h2>T</h2></div>")
    assert "base_selector" in excinfo.value.where
    assert "listing" in excinfo.value.where


def test_from_dict_rejects_bad_selector_at_load_time(tmp_path):
    """Loading is the cheap moment to fail; scraping 500 pages is not."""
    d = {"name": "products", "fields": [{"name": "price", "selector": "[[[bad"}]}

    with pytest.raises(InvalidSelectorError) as excinfo:
        ExtractionSchema.from_dict(d)

    assert excinfo.value.selector == "[[[bad"
    assert "price" in str(excinfo.value)


def test_from_json_rejects_bad_selector(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(
        '{"name": "products", "fields": [{"name": "price", "selector": "[[[bad"}]}',
        encoding="utf-8",
    )

    with pytest.raises(InvalidSelectorError) as excinfo:
        ExtractionSchema.from_json(path)

    assert "price" in str(excinfo.value)


def test_from_dict_rejects_bad_base_selector():
    d = {"base_selector": "[[[bad", "fields": [{"name": "title", "selector": "h1"}]}
    with pytest.raises(InvalidSelectorError) as excinfo:
        ExtractionSchema.from_dict(d)
    assert "base_selector" in excinfo.value.where


def test_field_without_a_selector_is_rejected():
    """An omitted selector means an empty one, which matches nothing by design."""
    with pytest.raises(InvalidSelectorError) as excinfo:
        ExtractionSchema.from_dict({"fields": [{"name": "title"}]})

    assert excinfo.value.where == "field 'title' in schema 'extraction'"


def test_valid_schema_with_no_base_selector_passes_validation():
    schema = ExtractionSchema.from_dict({"fields": [{"name": "title", "selector": "h1"}]})
    schema.validate()  # must not raise


def test_bundled_schemas_are_valid():
    """Every schema shipped in schemas/ must compile, or `protor extract` aborts."""
    schema_dir = Path(__file__).resolve().parent.parent / "schemas"
    files = sorted(schema_dir.glob("*.json"))
    assert files, "no bundled schemas found"
    for path in files:
        ExtractionSchema.from_json(path).validate()
