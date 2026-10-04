from pathlib import Path

import pytest
from bs4 import BeautifulSoup

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


def _schema_with(field_type: str, attribute: str = "") -> ExtractionSchema:
    """A one-field schema in the ordinary container-then-field shape."""
    return ExtractionSchema.from_dict(
        {
            "name": "probe",
            "base_selector": ".row",
            "fields": [
                {"name": "v", "selector": ".row > *", "type": field_type, "attribute": attribute}
            ],
        }
    )


def extract(soup: BeautifulSoup, schema: ExtractionSchema) -> list[dict]:
    from protor.extractor import extract_from_soup

    return extract_from_soup(soup, schema, base_url="https://x.example/")


class TestSchemaTyposAreRefusedNotGuessed:
    """
    A schema that cannot work is rejected at load, not guessed at per page.

    An unrecognised ``type`` used to fall through every branch and quietly
    extract the element's *text*: a field declared ``"hrefs"`` returned the link's
    label where its URL belonged, and the run reported success.
    """

    @pytest.mark.parametrize("field_type", ["text", "html", "href", "src", "regex"])
    def test_every_supported_type_is_accepted(self, field_type):
        schema = ExtractionSchema.from_dict(
            {
                "name": "ok",
                "base_selector": "a",
                "fields": [{"name": "v", "selector": "a", "type": field_type, "attribute": "x"}],
            }
        )
        assert schema.fields[0].type == field_type

    @pytest.mark.parametrize("field_type", ["hrefs", "attribte", "", "TEXT", "regexes"])
    def test_an_unknown_type_is_refused_at_load(self, field_type):
        from protor.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as exc:
            ExtractionSchema.from_dict(
                {
                    "name": "typo",
                    "base_selector": "a",
                    "fields": [{"name": "v", "selector": "a", "type": field_type}],
                }
            )
        assert field_type in str(exc.value) or field_type == ""

    def test_the_message_names_the_supported_types(self):
        from protor.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as exc:
            ExtractionSchema.from_dict(
                {
                    "name": "typo",
                    "base_selector": "a",
                    "fields": [{"name": "v", "selector": "a", "type": "hrefs"}],
                }
            )
        message = str(exc.value)
        assert "regex" in message and "attribute" in message, message


class TestFieldValuesThatUsedToBeWrong:
    def test_a_multi_valued_attribute_is_joined_not_repred(self):
        """bs4 returns a list for `class`, `rel`, `headers`, and str() leaked it."""
        soup = BeautifulSoup(
            '<div class="row"><a class="product-title special" href="/buy/42">Buy</a></div>',
            "lxml",
        )
        result = extract(soup, _schema_with("attribute", "class"))
        assert result[0]["v"] == "product-title special"

    def test_a_regex_without_a_capture_group_returns_the_match(self):
        """It raised IndexError — a traceback after the page was already fetched."""
        soup = BeautifulSoup('<div class="row"><p>Order 12345 shipped</p></div>', "lxml")
        result = extract(soup, _schema_with("regex", r"Order \d+"))
        assert result[0]["v"] == "Order 12345"

    def test_a_regex_with_a_group_still_returns_the_group(self):
        soup = BeautifulSoup('<div class="row"><p>Order 12345 shipped</p></div>', "lxml")
        result = extract(soup, _schema_with("regex", r"Order (\d+)"))
        assert result[0]["v"] == "12345"

    def test_a_regex_that_does_not_match_falls_back_to_the_default(self):
        soup = BeautifulSoup('<div class="row"><p>nothing here</p></div>', "lxml")
        result = extract(soup, _schema_with("regex", r"Order (\d+)"))
        assert result[0]["v"] is None


class TestOutputPathIsConfinedToTheOutputDirectory:
    """
    The output filename is built from the schema name and the URL's stem.

    Both come from outside, so both go through ``safe_filename``: a schema named
    ``"../../pwned"`` wrote the file two directories above the directory that was
    asked for, and an absolute path in the name raised instead of being written.
    """

    @pytest.mark.parametrize("name", ["../../pwned", "/etc/passwd", "a/b", "..", "."])
    def test_a_hostile_schema_name_stays_inside(self, name, tmp_path):
        from protor.utils import safe_filename

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out_file = out_dir / f"{safe_filename(name)}_{safe_filename('page')}.json"

        assert out_file.parent == out_dir, f"{name!r} escaped to {out_file.parent}"
        assert out_file.resolve().is_relative_to(out_dir.resolve())

    def test_the_real_command_writes_inside_the_output_directory(self, tmp_path):
        """End to end through `protor extract`, with a traversal name."""
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = b"<html><body><p>hello</p></body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            from protor.cli import _build_parser, _cmd_extract

            schema_path = tmp_path / "schema.json"
            schema_path.write_text(
                json.dumps(
                    {
                        "name": "../../pwned",
                        "base_selector": "body",
                        "fields": [{"name": "t", "selector": "p"}],
                    }
                ),
                encoding="utf-8",
            )
            out_dir = tmp_path / "out"
            url = f"http://127.0.0.1:{server.server_port}/page.html"
            args = _build_parser().parse_args(
                ["extract", url, str(schema_path), "--output", str(out_dir)]
            )
            _cmd_extract(args)
        finally:
            server.shutdown()

        written = list(out_dir.rglob("*.json"))
        assert written, "nothing was written"
        for path in written:
            assert path.resolve().is_relative_to(out_dir.resolve()), path
        assert not list(tmp_path.parent.glob("pwned*.json")), "wrote outside the output dir"


class TestBlocklistSeesTheRealHost:
    """
    The domain check must read the host, not the netloc.

    ``netloc`` keeps any userinfo, so ``https://user@doubleclick.net/pixel``
    compared as ``user@doubleclick.net`` — matching no tracker — and a hostile
    page's script walked straight through a blocklist meant to stop exactly
    that. The hand-rolled port strip also mangled IPv6.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "https://doubleclick.net/pixel",
            "https://user@doubleclick.net/pixel",
            "https://user:pass@doubleclick.net/pixel",
            "https://doubleclick.net:443/pixel",
        ],
    )
    def test_a_tracker_is_blocked_however_it_is_spelled(self, url):
        from protor.blocklist import Blocklist

        assert Blocklist().is_url_blocked(url) is True, url

    @pytest.mark.parametrize(
        ("url", "blocked"),
        [
            # The host really is cdn.example; the tracker name is userinfo.
            ("https://doubleclick.net@cdn.example/pixel", False),
            ("https://example.com/page", False),
            ("http://[::1]:8080/x", False),
        ],
    )
    def test_ordinary_urls_are_not_blocked(self, url, blocked):
        from protor.blocklist import Blocklist

        assert Blocklist().is_url_blocked(url) is blocked, url
