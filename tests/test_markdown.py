from protor.markdown import extract_clean_markdown, html_to_markdown


def test_basic_paragraph():
    assert "Hello world" in html_to_markdown("<p>Hello world</p>")


def test_headings():
    md = html_to_markdown("<h1>Title</h1><h2>Sub</h2>")
    assert "# Title" in md
    assert "## Sub" in md


def test_links():
    md = html_to_markdown('<a href="https://example.com">click</a>')
    assert "[click](https://example.com)" in md


def test_images():
    md = html_to_markdown('<img src="pic.jpg" alt="photo">')
    assert "![photo](pic.jpg)" in md


def test_lists():
    md = html_to_markdown("<ul><li>one</li><li>two</li></ul>")
    assert "- one" in md
    assert "- two" in md


def test_ordered_list():
    md = html_to_markdown("<ol><li>first</li><li>second</li></ol>")
    assert "1. first" in md
    assert "2. second" in md


def test_bold_text():
    md = html_to_markdown("<b>bold</b>")
    assert "bold" in md


def test_italic_text():
    md = html_to_markdown("<i>italic</i>")
    assert "italic" in md


def test_code_block():
    html = '<pre><code class="language-python">print("hi")</code></pre>'
    md = html_to_markdown(html)
    assert "```python" in md
    assert 'print("hi")' in md


def test_inline_code():
    md = html_to_markdown("use <code>x</code> here")
    assert "x" in md


def test_blockquote():
    md = html_to_markdown("<blockquote>wise words</blockquote>")
    assert "> wise words" in md


def test_table():
    html = """<table>
    <tr><th>A</th><th>B</th></tr>
    <tr><td>1</td><td>2</td></tr>
    </table>"""
    md = html_to_markdown(html)
    assert "| A | B |" in md
    assert "| 1 | 2 |" in md


def test_strips_script_and_style():
    html = "<p>text</p><script>alert('x')</script><style>.x{color:red}</style>"
    md = html_to_markdown(html)
    assert "alert" not in md
    assert "color:red" not in md
    assert "text" in md


def test_strips_noise_tags():
    html = "<nav>nav</nav><footer>foot</footer><p>content</p>"
    md = html_to_markdown(html)
    assert "content" in md


def test_empty_input():
    assert html_to_markdown("") == ""


def test_plain_text():
    assert "just text" in html_to_markdown("just text")


def test_br_tag():
    md = html_to_markdown("line1<br>line2")
    assert "line1" in md
    assert "line2" in md


def test_hr():
    md = html_to_markdown("<hr>")
    assert "---" in md


def test_multiple_paragraphs():
    html = "<p>first</p><p>second</p>"
    md = html_to_markdown(html)
    assert "first" in md
    assert "second" in md


def test_nested_elements():
    html = "<div><p>nested bold text</p></div>"
    md = html_to_markdown(html)
    assert "nested" in md
    assert "bold" in md


def test_entities():
    md = html_to_markdown("&amp; &lt; &gt;")
    assert "&" in md
    assert "<" in md
    assert ">" in md


def test_long_content_truncation():
    long = "<p>" + "word " * 600 + "</p>"
    md = html_to_markdown(long)
    assert len(md) < len(long)


def test_whitespace_cleanup():
    md = html_to_markdown("<p>  spaces  </p>")
    assert "spaces" in md


def test_nested_lists():
    html = "<ul><li>outer<ul><li>inner</li></ul></li></ul>"
    md = html_to_markdown(html)
    assert "outer" in md
    assert "inner" in md


def test_description_list():
    html = "<dl><dt>term</dt><dd>definition</dd></dl>"
    md = html_to_markdown(html)
    assert "term" in md
    assert "definition" in md


def test_figure_caption():
    html = "<figure><img src='a.jpg'><figcaption>caption</figcaption></figure>"
    md = html_to_markdown(html)
    assert "caption" in md


def test_extract_clean_markdown():
    html = "<h1>Title</h1><p>Content here</p>"
    md = extract_clean_markdown(html)
    assert "Title" in md
    assert "Content here" in md


def test_extract_clean_markdown_max_chars():
    html = "<p>" + "word " * 200 + "</p>"
    md = extract_clean_markdown(html, max_chars=50)
    assert len(md) <= 70


def test_extract_clean_markdown_base_url():
    html = '<a href="/relative">link</a>'
    md = extract_clean_markdown(html, base_url="https://example.com")
    assert "https://example.com/relative" in md


def test_span_elements():
    md = html_to_markdown('<span class="highlight">important</span>')
    assert "important" in md


def test_heading_levels():
    for i in range(1, 7):
        md = html_to_markdown(f"<h{i}>H{i}</h{i}>")
        assert f"{'#' * i} H{i}" in md
