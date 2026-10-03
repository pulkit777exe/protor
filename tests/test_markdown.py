from bs4 import BeautifulSoup

from protor.markdown import (
    _clean_markdown,
    _is_noise,
    _Lines,
    clean_soup,
    extract_clean_markdown,
    html_to_markdown,
    soup_to_markdown,
)


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


# ── clean_soup ────────────────────────────────────────────────────────────────


def test_nested_noise_elements_do_not_crash_clean_soup():
    """
    A noise element containing other elements used to raise AttributeError.

    decompose() clears the __dict__ of everything inside the removed element, so
    the next tag in the pre-materialised list had attrs=None and the noise check
    raised - which the engine recorded as a scrape error. Real pages hit this on
    every <nav>, <header>, <footer> and <aside>.
    """
    html = (
        "<html><body>"
        "<header><div><span>brand</span></div></header>"
        "<nav><a href='/a'>menu</a></nav>"
        "<main><h1>Title</h1><p>Real content.</p></main>"
        "<aside><b>promo</b></aside>"
        "<footer><div>colophon</div></footer>"
        "<script>track()</script>"
        "</body></html>"
    )
    md = html_to_markdown(html)
    assert "Real content." in md
    assert "track()" not in md


def test_deeply_nested_noise_is_removed():
    html = "<nav><div><ul><li><a href='/deep'>x</a></li></ul></div></nav><p>kept</p>"
    md = html_to_markdown(html)
    assert "kept" in md
    assert "deep" not in md


def test_clean_soup_is_idempotent():
    soup = BeautifulSoup("<nav><a href='/a'>menu</a></nav><p>kept</p>", "html.parser")
    clean_soup(soup)
    first = soup.get_text()
    clean_soup(soup)
    assert soup.get_text() == first


def test_clean_soup_does_not_materialise_the_whole_document():
    """
    find_all(True) builds a list of every tag in the page before the loop starts,
    which measured 447 KiB of transient list and attribute objects for a 177 KiB
    document — 2,258 KiB for a 213 KiB page. A level-at-a-time descent keeps it
    at 9 KiB, and makes a removed subtree unreachable rather than merely
    detectable, which is what the _is_decomposed guard exists for.
    """
    real = BeautifulSoup.find_all
    selectors: list = []

    def spy(self, name=None, attrs=None, **kwargs):
        selectors.append(name)
        return real(self, name, attrs, **kwargs)

    soup = BeautifulSoup(
        "<html><body>"
        + "<nav><a href='/n'>n</a></nav><div><p>kept</p></div>" * 50
        + "</body></html>",
        "lxml",
    )
    BeautifulSoup.find_all = spy
    try:
        clean_soup(soup)
    finally:
        BeautifulSoup.find_all = real

    assert selectors == [], f"clean_soup must not search, got selectors {selectors}"
    assert "kept" in soup.get_text()


def test_noise_subtrees_are_stripped_whole():
    """A noise element buried inside another one still goes, descendants included."""
    soup = BeautifulSoup(
        "<html><body>"
        "<header><footer><aside><form><input><button>b</button></form></aside></footer></header>"
        "<div class='cookie-banner'><div class='modal'><div id='popup'>gone</div></div></div>"
        "<p>kept</p>"
        "</body></html>",
        "lxml",
    )
    clean_soup(soup)
    assert "kept" in soup.get_text()
    assert "gone" not in soup.get_text()
    assert "b" not in soup.get_text()


# ── character budget ──────────────────────────────────────────────────────────


def _page(n_blocks: int) -> str:
    """A page of *n_blocks* sections, each far enough to blow past any test cap."""
    return (
        "<html><body><main>"
        + "".join(
            f"<section><h2>Section {i}</h2>"
            f"<p>Paragraph {i} of prose that runs on for a little while.</p></section>"
            for i in range(n_blocks)
        )
        + "</main></body></html>"
    )


def test_budget_stops_the_walk_rather_than_truncating_afterwards():
    """
    MAX_MARKDOWN_CHARS was applied to the finished string, so the whole document
    was walked and joined only for the tail to be discarded: a 1200-block page
    paid for 460 kB of output to keep 40 kB.

    Counting renderer calls is the only way to tell stopping early from slicing
    afterwards: both produce the same truncated string.
    """
    import protor.markdown as markdown

    html = _page(4000)
    soup = BeautifulSoup(html, "lxml")
    clean_soup(soup)

    calls = 0
    real = markdown._render_inline_children

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    markdown._render_inline_children = counting
    try:
        full = markdown.soup_to_markdown(soup)
        uncapped_calls = calls
        calls = 0
        budgeted = markdown.soup_to_markdown(soup, max_chars=40_000)
        budgeted_calls = calls
    finally:
        markdown._render_inline_children = real

    assert len(full) > 40_000, "fixture must exceed the cap to be meaningful"
    assert len(budgeted) <= 40_000 + 32
    assert budgeted.endswith("[truncated]")
    # Same content up to the cut, so the cap changes the length and nothing else.
    assert full.startswith(budgeted[: budgeted.index("[truncated]")].rstrip())

    # The whole point: the tail is never rendered, not merely dropped afterwards.
    # The cap keeps ~40 kB of a ~190 kB render, so well under half the work is
    # left; without the early stop both numbers are identical.
    assert budgeted_calls < uncapped_calls / 2, (
        f"budgeted render made {budgeted_calls} calls vs {uncapped_calls} uncapped"
    )


def test_under_cap_pages_are_byte_identical_with_and_without_a_budget():
    """A page that fits must come back the same whether or not a cap is passed."""
    for n_blocks in (1, 5, 30):
        soup = BeautifulSoup(_page(n_blocks), "lxml")
        clean_soup(soup)
        assert soup_to_markdown(soup, max_chars=40_000) == soup_to_markdown(soup)
        assert "[truncated]" not in soup_to_markdown(soup, max_chars=40_000)


def test_zero_budget_means_no_limit():
    soup = BeautifulSoup(_page(400), "lxml")
    clean_soup(soup)
    assert len(soup_to_markdown(soup, max_chars=0)) == len(soup_to_markdown(soup))
    assert "[truncated]" not in soup_to_markdown(soup, max_chars=0)


def test_budget_counts_what_the_finished_text_will_contain():
    """
    Blank-line runs collapse to a single pair, and the budget charges them that
    way. Charging them raw would let the accounting run ahead of the cap and cut
    a page that would have fitted.
    """
    # A document that is almost all blank lines between its words.
    lines = ["w"] + [""] * 200 + ["w"] * 200 + [""] * 200 + ["w"] * 200
    budget = _Lines(len(_clean_markdown("\n".join(lines))) + 1)
    for line in lines:
        budget.append(line)
    assert not budget.capped, "a document this size must fit the cap it was given"
    assert budget.used == len(_clean_markdown("\n".join(lines)))


def test_budget_stops_inside_a_long_list():
    """
    Lists emit one line per item, so a long one can exhaust the budget on its
    own. Without the check in the item loop, every remaining item is rendered
    even though none of them will be kept.
    """
    import protor.markdown as markdown

    items = "".join(f"<li>Item number {i} of the list.</li>" for i in range(3000))
    soup = BeautifulSoup(f"<html><body><ul>{items}</ul></body></html>", "lxml")
    clean_soup(soup)

    calls = 0
    real = markdown._render_inline_children

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    markdown._render_inline_children = counting
    try:
        markdown.soup_to_markdown(soup, max_chars=2_000)
    finally:
        markdown._render_inline_children = real

    # 3,000 items at ~25 characters each; a 2,000-character cap keeps ~80.
    assert calls < 300, f"rendered {calls} of 3,000 items past a 2,000-char cap"


def test_rendered_page_is_capped_end_to_end():
    """parse_html used to cap with a separate slice over the whole render."""
    from protor.config import MAX_MARKDOWN_CHARS
    from protor.parser import parse_html

    _, page = parse_html(_page(1500), "https://example.com/")
    assert len(page.markdown_content) <= MAX_MARKDOWN_CHARS + 32
    assert page.markdown_content.endswith("[truncated]")


# ── the noise check ───────────────────────────────────────────────────────────


def test_noise_check_skips_tags_without_a_class_or_id():
    """
    The pattern can only match a class or an id, and most tags have neither, so
    searching for those tags was pure overhead — 1.19x on the noise pass, which
    runs both in clean_soup and for every element the renderer visits.
    """
    import protor.markdown as markdown

    real_pattern = markdown._NOISE_PATTERN
    searched: list[str] = []

    class _Spy:
        def search(self, text):
            searched.append(text)
            return real_pattern.search(text)

    markdown._NOISE_PATTERN = _Spy()
    try:
        soup = BeautifulSoup("<div><p>plain</p><span class='ok'>x</span></div>", "lxml")
        markdown.clean_soup(soup)
    finally:
        markdown._NOISE_PATTERN = real_pattern

    # One search, for the only tag carrying a class to search.
    assert len(searched) == 1, searched


def test_noise_check_still_matches_on_class_or_id():
    soup = BeautifulSoup(
        "<div class='sidebar'>a</div><div id='cookie'>b</div><div id='ad-slot'>c</div>"
        "<div class='content'>d</div><p>e</p>",
        "lxml",
    )
    tags = soup.find_all("div")
    assert [_is_noise(t) for t in tags] == [True, True, True, False]
    assert _is_noise(soup.find("p")) is False
