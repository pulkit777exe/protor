from protor.extractor import ExtractionSchema, Extractor, FieldSchema


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
