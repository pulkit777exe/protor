"""Throwaway audit probes for protor/parser.py and protor/blocklist.py. Not kept."""

from __future__ import annotations

from urllib.parse import urlparse

from protor.blocklist import Blocklist, is_blocked
from protor.parser import extract_links, looks_like_html, parse_html, parse_soup


def hdr(n: str) -> None:
    print("\n" + "=" * 78)
    print(f"PROBE {n}")
    print("=" * 78)


# ── 1: strip_guessed_noise honoured by clean_soup but NOT by the markdown path ──
CLASSIFIEDS = (
    "<html><body><div class='classifieds'>"
    "<div class='ad-card'><h3>Blue widget</h3><span class='ad-price'>$5</span></div>"
    "<div class='ad-card'><h3>Red widget</h3><span class='ad-price'>$7</span></div>"
    "</div>"
    "<div class='cookie-banner'><p>accept cookies</p></div>"
    "</body></html>"
)


def test_p1_guessed_noise_strip_guessed_noise_false():
    hdr("1  strip_guessed_noise=False : text keeps the content, markdown does not")
    _, page = parse_html(CLASSIFIEDS, "https://x.example/", strip_guessed_noise=False)
    print(f"text_content    : {page.text_content!r}")
    print(f"markdown_content: {page.markdown_content!r}")
    print(f"'Blue widget' in text_content    -> {'Blue widget' in page.text_content}")
    print(f"'Blue widget' in markdown_content -> {'Blue widget' in page.markdown_content}")

    # and the default (guesses on) for comparison
    _, on = parse_html(CLASSIFIEDS, "https://x.example/")
    print(f"default markdown: {on.markdown_content!r}")


# ── 2: <title> with more than one child ───────────────────────────────────────
TITLES = [
    "<html><head><title>Plain Title</title></head><body>x</body></html>",
    "<html><head><title>Caf&eacute; &amp; Bar</title></head><body>x</body></html>",
    "<html><head><title>Foo <b>bold</b> tail</title></head><body>x</body></html>",
    "<html><head><title>Foo<!-- gtm --> tail</title></head><body>x</body></html>",
    "<html><head><title><![CDATA[Foo & Bar]]></title></head><body>x</body></html>",
    "<html><head><title>Foo\n   Bar</title></head><body>x</body></html>",
    "<html><head><title></title></head><body>x</body></html>",
    "<html><head><title>  Padded  </title></head><body>x</body></html>",
]


def test_p2_title():
    hdr("2  <title> harvest: title.string is None when the element has >1 child")
    for html in TITLES:
        _, page = parse_html(html, "https://x.example/")
        print(f"{html[:60]:62} -> title={page.metadata.title!r}")


# ── 3: base-domain comparison is case/port sensitive ─────────────────────────
def test_p3_link_case():
    hdr("3  _Harvest compares raw urlparse().netloc; engine lowercases its own")
    html = (
        "<html><body>"
        "<a href='/rel'>rel</a>"
        "<a href='https://example.com/abs'>abs lowercase</a>"
        "<a href='https://EXAMPLE.com/abs2'>abs uppercase</a>"
        "<a href='https://example.com:8080/abs3'>abs with port</a>"
        "</body></html>"
    )
    for base in ("https://example.com/", "https://EXAMPLE.com/", "https://example.com:8080/"):
        print(f"base={base!r:34} -> {extract_links(html, base)}")
    print()
    print("engine's own comparison, for contrast:")
    from protor.engine import CrawlEngine  # noqa: F401

    print("  CrawlEngine._allowed_domain is lowercased (engine.py:343)")
    print("  Blocklist.is_domain_blocked is lowercased (blocklist.py:187)")
    print("  parser._Harvest._base_domain is NOT (parser.py:215)")
    print()
    print("canonicalize_url lowercases netloc:", urlparse("https://EXAMPLE.com/x").netloc.lower())


# ── 4: urlparse ValueError from an untrusted href ────────────────────────────
def test_p4_urlparse_raises():
    hdr("4  an untrusted href that urlparse rejects aborts the entire parse")
    for href in ("http://[::1", "//[::1", "http://[::1]:80/x"):
        html = f"<html><head><title>Kept</title></head><body><p>real content</p>" f"<a href='{href}'>x</a></body></html>"
        try:
            _, page = parse_html(html, "https://example.com/")
            print(f"href={href!r:22} -> OK links={page.links}")
        except Exception as exc:
            print(f"href={href!r:22} -> {type(exc).__name__}: {exc}")
    print()
    print("Same href through extract_links (public API):")
    try:
        print(extract_links("<a href='http://[::1'>x</a>", "https://example.com/"))
    except Exception as exc:
        print(f"  {type(exc).__name__}: {exc}")
    print()
    print("engine._process_one wraps parse_html in `except Exception` -> _fail():")
    print("  so the whole page becomes a failed scrape, metadata and all.")


# ── 5: _extract_text truncation ───────────────────────────────────────────────
def test_p5_text_truncation():
    hdr("5  _extract_text truncates with no marker; an over-long FIRST line empties it")
    big = "x" * 20_000
    html = f"<html><body><p>{big}</p></body></html>"
    _, page = parse_html(html, "https://example.com/")
    print(f"text len={len(page.text_content)} markdown len={len(page.markdown_content)}")
    print(f"text ends with marker? {'[truncated]' in page.text_content}")
    print(f"markdown ends with marker? {'[truncated]' in page.markdown_content}")

    html2 = (
        "<html><body>"
        f"<p>{'a' * 20_000}</p>"
        "<p>THIS REAL ARTICLE TEXT COMES AFTER THE LONG LINE</p>"
        "</body></html>"
    )
    _, p2 = parse_html(html2, "https://example.com/")
    print(f"\nlong line then real article: text={p2.text_content!r}")
    print(f"                            markdown keeps it? "
          f"{'THIS REAL ARTICLE' in p2.markdown_content}")

    html3 = "<html><body>" + "".join(
        f"<p>Paragraph number {i} with some text in it.</p>" for i in range(400)
    ) + "</body></html>"
    _, p3 = parse_html(html3, "https://example.com/")
    print(f"\n400 paragraphs: text len={len(p3.text_content)} marker="
          f"{'[truncated]' in p3.text_content} last kept="
          f"{p3.text_content.splitlines()[-1]!r}")
    print(f"                md  len={len(p3.markdown_content)} marker="
          f"{'[truncated]' in p3.markdown_content}")


# ── 6: _add_script's startswith('http') vs _add_link's scheme check ───────────
def test_p6_script_scheme():
    hdr("6  sibling decisions about 'is this a fetchable URL?' differ")
    html = (
        "<html><body>"
        "<script src='https://cdn.example/a.js'></script>"
        "<script src='httpx://evil.example/b.js'></script>"
        "<script src='HTTP://UP.example/c.js'></script>"
        "<script src='httpjavascript:demo.js'></script>"
        "<script src='/rel.js?v=1#frag'></script>"
        "<a href='httpx://evil.example/b'>x</a>"
        "</body></html>"
    )
    _, page = parse_html(html, "https://example.com/")
    print("js_links:", page.js_links)
    print("links   :", page.links)
    print()
    from protor.utils import urljoin

    print("urljoin(base,'httpx://evil.example/b.js') ->", urljoin("https://example.com/", "httpx://evil.example/b.js"))


# ── 7: looks_like_html sniffing ──────────────────────────────────────────────
def test_p7_sniff():
    hdr("7  looks_like_html: unknown content-type + real markup")
    cases = [
        ("", "<!DOCTYPE html><html><body>hi</body></html>"),
        ("application/octet-stream", "<!DOCTYPE html><html><body>hi</body></html>"),
        ("application/octet-stream", "\ufeff<!DOCTYPE html><html><body>hi</body></html>"),
        ("application/octet-stream", "<!-- a comment -->\n<!DOCTYPE html><html>hi</html>"),
        ("", "\ufeff<!DOCTYPE html><html><body>hi</body></html>"),
        ("application/octet-stream", "<span>fragment</span>"),
        ("application/octet-stream", "   \n\n\t<html><body>hi</body></html>"),
        ("application/octet-stream", "<p>paragraph first</p>"),
        ("application/octet-stream", "<HTML><BODY>hi</BODY></HTML>"),
        ("text/html; charset=utf-8", "whatever"),
        ("", ""),
        ("", "plain text, definitely a page"),
    ]
    for ctype, body in cases:
        print(f"ctype={ctype!r:28} body={body[:42]!r:46} -> {looks_like_html(ctype, body)}")


# ── 8: blocklist checks ─────────────────────────────────────────────────────
def test_p8_blocklist():
    hdr("8  blocklist: docstring says ~3,500 domains")
    bl = Blocklist()
    print("blocked_count:", bl.blocked_count)
    print("len(_blocked_domains):", len(bl._blocked_domains))

    hdr("8b blocklist: hostname shapes")
    urls = [
        "https://doubleclick.net/x",
        "https://DOUBLECLICK.NET/x",
        "https://doubleclick.net.:8080/x",
        "https://doubleclick.net:443/x",
        "http://doubleclick.net",
        "//doubleclick.net/x",
        "https://sub.doubleclick.net/x",
        "https://notdoubleclick.net/x",
        "https://doubleclick.net@evil.example/x",
        "https://evil.example/?u=doubleclick.net",
        "https://%64oubleclick.net/x",
        "https://doubleclick%2enet/x",
        "https://xn--doubleclick-2we.net/x",
        "https://googleadservices.com/pagead/js",
        "https://www.googleadservices.com/pagead/js",
        "https://googleadservices.com:8443/x",
        "https://facebook.com/",
        "https://t.co/abc",
        "https://clarity.ms/tag.js",
        "https://app.posthog.com/i/e",
        "https://bam.nr-data.net/1/x",
        "https://hm.baidu.com/hm.js",
        "https://mc.yandex.ru/metrika/watch.js",
        "https://pixel.wp.com/g.gif",
        "https://www.googletagservices.com/tag/js/gpt.js",
        "https://gql.facebook.com/x",
        "https://edge-star-min.shoptify.com/x",
        "https://cdn.matomo.cloud/x",
        "https://in.appcenter.ms/x",
        "https://api.amplitude.com/2/httpapi",
        "",
        "not a url at all",
    ]
    for u in urls:
        try:
            r = is_blocked(u)
        except Exception as exc:
            r = f"{type(exc).__name__}: {exc}"
        print(f"{u!r:52} -> {r}")

    hdr("8c blocklist: from_file shapes")
    import tempfile
    from pathlib import Path

    p = Path(tempfile.mkdtemp()) / "hosts.txt"
    p.write_text(
        "# comment\n"
        "doubleclick.net\n"
        "  spaced.example  \n"
        "*.wildcard.example\n"
        "||adblock-syntax.example^\n"
        "0.0.0.0 hostsfile.example\n"
        "UPPER.EXAMPLE\n"
        "http://withscheme.example\n",
        encoding="utf-8",
    )
    bl2 = Blocklist.from_file(p)
    print("blocked_count:", bl2.blocked_count)
    for d in (
        "spaced.example",
        "wildcard.example",
        "a.wildcard.example",
        "adblock-syntax.example",
        "hostsfile.example",
        "upper.example",
        "withscheme.example",
    ):
        print(f"  is_domain_blocked({d!r:26}) -> {bl2.is_domain_blocked(d)}")
    print("\n  url form of the same file entries:")
    for u in (
        "https://spaced.example/x",
        "https://wildcard.example/x",
        "https://a.wildcard.example/x",
        "https://adblock-syntax.example/x",
        "https://hostsfile.example/x",
    ):
        print(f"  is_url_blocked({u!r:42}) -> {bl2.is_url_blocked(u)}")

    hdr("8d blocklist: missing file is silent")
    bl3 = Blocklist.from_file("/nonexistent/definitely/not/here.txt")
    print("blocked_count:", bl3.blocked_count, "-> no error, no warning")

    hdr("8e blocklist: is_domain_blocked with a port")
    print("is_domain_blocked('doubleclick.net:443') ->", bl.is_domain_blocked("doubleclick.net:443"))
    print("is_domain_blocked('  doubleclick.net  ') ->", bl.is_domain_blocked("  doubleclick.net  "))
    print("is_url_blocked('  https://doubleclick.net/x  ') ->",
          bl.is_url_blocked("  https://doubleclick.net/x  "))

    hdr("8f blocklist: requested-host bypass vs hostname comparison")
    print("engine compares parsed.netloc.lower() (keeps port+userinfo)")
    print("  requested_hosts from scraper = {urlparse(url).netloc}")
    for u in ("https://facebook.com:443/", "https://FACEBOOK.com/", "https://facebook.com@x/"):
        print(f"  urlparse({u!r:30}).netloc.lower() = {urlparse(u).netloc.lower()!r}")
        try:
            print(f"    is_url_blocked -> {bl.is_url_blocked(u)}")
        except Exception as exc:
            print(f"    is_url_blocked -> {type(exc).__name__}: {exc}")


# ── 9: parse_html drops max_markdown_chars ───────────────────────────────────
def test_p9_markdown_budget():
    hdr("9  parse_html cannot forward max_markdown_chars (parse_soup can)")
    import inspect

    print("parse_html :", inspect.signature(parse_html))
    print("parse_soup :", inspect.signature(parse_soup))
    big = "<html><body>" + "".join(f"<p>para {i} " + "y" * 200 + "</p>" for i in range(500)) + "</body></html>"
    _, page = parse_html(big, "https://example.com/")
    print(f"markdown len={len(page.markdown_content)} marker={'[truncated]' in page.markdown_content}")
    soup_pg = parse_soup.__wrapped__ if hasattr(parse_soup, "__wrapped__") else None  # type: ignore[attr-defined]
    from protor.parser import ParsedPage  # noqa: F401
    import protor.parser as P

    _, page2 = P.parse_html(big, "https://example.com/")
    print("same via parse_html:", len(page2.markdown_content))
    from bs4 import BeautifulSoup
    from protor.markdown import clean_soup

    s = BeautifulSoup(big, "lxml")
    clean_soup(s)
    p3 = P.parse_soup(s, "https://example.com/", max_markdown_chars=200_000)
    print("parse_soup with a 200k markdown budget:", len(p3.markdown_content), soup_pg)


# ── 10: self-links / fragment-only hrefs ─────────────────────────────────────
def test_p10_fragment():
    hdr("10  fragment-only hrefs become a self-link; js src keeps its fragment")
    html = (
        "<html><body>"
        "<a href='#top'>top</a>"
        "<a href=''>empty</a>"
        "<a href='javascript:void(0)'>js</a>"
        "<a href='mailto:a@b.c'>mail</a>"
        "<a href='tel:+1'>tel</a>"
        "<script src='/a.js#frag'></script>"
        "</body></html>"
    )
    _, page = parse_html(html, "https://example.com/dir/page.html")
    print("links   :", page.links)
    print("js_links:", page.js_links)


# ── 11: meta elif-chain ──────────────────────────────────────────────────────
def test_p11_meta():
    hdr("11  _add_meta elif-chain: name= and property= on one tag")
    html = (
        "<html><head>"
        "<meta name='author' property='og:author' content='Ann'>"
        "<meta name='description' content='D'>"
        "<meta name='description' content='second wins?'>"
        "<meta property='og:title' content='OG'>"
        "<meta property='OG:image' content='i'>"
        "<meta name='Description' content='case'>"
        "<meta content='nameless'>"
        "</head><body>x</body></html>"
    )
    _, page = parse_html(html, "https://example.com/")
    print(page.metadata)

# ===== merged from probe2 =====


# ── 12: <base href> is ignored ───────────────────────────────────────────────
def test_p12_base_href():
    hdr("12  <base href> is ignored; every relative URL resolves against the page URL")
    html = (
        "<html><head><base href='/docs/'></head><body>"
        "<a href='guide.html'>guide</a>"
        "<a href='../top.html'>top</a>"
        "</body></html>"
    )
    _, page = parse_html(html, "https://example.com/index.html")
    print("links:", page.links)
    print("markdown:", repr(page.markdown_content))
    print()
    print("what the browser would do: guide.html -> https://example.com/docs/guide.html")
    print("  and ../top.html          -> https://example.com/top.html")
    print()
    print("also ignored by the markdown renderer (same resolve_url(base_url, href)):")


# ── 13: <title> is in text_content but not in markdown_content ───────────────
def test_p13_title_in_text():
    hdr("13  text_content and markdown_content disagree about <title>")
    html = (
        "<html><head><title>The Page Title</title>"
        "<meta name='description' content='d'></head>"
        "<body><p>Body text.</p></body></html>"
    )
    _, page = parse_html(html, "https://example.com/")
    print(f"text_content    : {page.text_content!r}")
    print(f"markdown_content: {page.markdown_content!r}")
    print(f"title in text    -> {'The Page Title' in page.text_content}")
    print(f"title in markdown-> {'The Page Title' in page.markdown_content}")

    frag = "<div class='x'><title>Frag Title</title><p>Frag body</p></div>"
    _, page2 = parse_html(frag, "https://example.com/")
    print(f"\nno <body> (a fragment):")
    print(f"  text    : {page2.text_content!r}")
    print(f"  markdown: {page2.markdown_content!r}")


# ── 14: one hostile href fails the whole page, end to end ────────────────────
HOSTILE = (
    "<html><head><title>Perfectly Good Page</title>"
    "<meta name='description' content='great content'></head>"
    "<body><h1>Real Article</h1><p>Lots of good text here.</p>"
    "<a href='http://[::1'>one bad link</a>"
    "</body></html>"
)


def test_p14_engine_end_to_end(tmp_path, monkeypatch):
    hdr("14  END TO END: one malformed href turns a good page into a scrape ERROR")
    import asyncio

    import protor.engine as engine_mod
    from protor.engine import CrawlEngine, StaticQueue, StaticSource
    from protor.fetcher import FetchResult

    async def fake_fetch(session, url, **kwargs):
        return FetchResult(text=HOSTILE, nbytes=len(HOSTILE), status=200, content_type="text/html")

    monkeypatch.setattr(engine_mod, "fetch", fake_fetch)

    async def run(body):
        engine = CrawlEngine(
            queue=StaticQueue(["https://example.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=1,
        )
        stats = await engine.arun()
        return stats, engine.manifests, list(engine._rows)

    stats, manifests, rows = asyncio.run(run(HOSTILE))
    print(f"stats        : scraped={stats.scraped} errors={stats.errors} total={stats.total}")
    print(f"manifests    : {manifests}")
    print(f"row          : {rows}")
    print()
    print("the HTML file on disk (protor saves it BEFORE parsing):")
    for p in sorted(tmp_path.rglob("*.html")):
        print(f"  {p.name}: {p.read_text()[:60]}...")
    print()
    print("=> success=False, no manifest written, the run reports an error,")
    print("   and the perfectly good metadata/text/markdown of the page are lost.")


def test_p14b_control(tmp_path, monkeypatch):
    """Same page with the hostile link removed, to show it is the link alone."""
    hdr("14b CONTROL: same page, hostile href removed")
    import asyncio

    import protor.engine as engine_mod
    from protor.engine import CrawlEngine, StaticQueue, StaticSource
    from protor.fetcher import FetchResult

    good = HOSTILE.replace("<a href='http://[::1'>one bad link</a>", "<a href='/ok'>ok</a>")

    async def fake_fetch(session, url, **kwargs):
        return FetchResult(text=good, nbytes=len(good), status=200, content_type="text/html")

    monkeypatch.setattr(engine_mod, "fetch", fake_fetch)

    async def run():
        engine = CrawlEngine(
            queue=StaticQueue(["https://example.com/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=1,
        )
        stats = await engine.arun()
        return stats, engine.manifests, list(engine._rows)

    stats, manifests, rows = asyncio.run(run())
    print(f"stats     : scraped={stats.scraped} errors={stats.errors}")
    print(f"manifests : {[(m.metadata.title, m.text_content) for m in manifests]}")


# ── 15: is the percent-encoded host a real bypass? ──────────────────────────
def test_p15_percent_encoded_host():
    hdr("15  is %64oubleclick.net a real bypass for this fetcher?")
    from yarl import URL

    for u in ("https://%64oubleclick.net/x", "https://doubleclick.net/x"):
        y = URL(u)
        print(f"{u!r:32} yarl.host={y.host!r:26} is_absolute={y.is_absolute()}")
    print()
    print("=> yarl keeps the percent-encoding, so a fetch would ask DNS for the")
    print("   literal host and fail. Not an exploitable bypass in protor.")


# ── 16: does a non-http js url actually cost a request? ─────────────────────
def test_p16_js_scheme():
    hdr("16  what a non-http js url costs at download time")
    import asyncio

    from protor.fetcher import download_file

    async def go():
        for u in ("httpx://evil.example/b.js", "httpjavascript:demo.js"):
            try:
                r = await download_file(None, u, tmpdest)  # type: ignore[arg-type]
                print(f"{u!r:32} download_file -> {r}")
            except Exception as exc:
                print(f"{u!r:32} download_file -> {type(exc).__name__}: {exc}")

    import pathlib
    import tempfile

    tmpdest = pathlib.Path(tempfile.mkdtemp()) / "b.js"
    asyncio.run(go())
    print()
    print("engine._download_js_file does not catch, but asyncio.wait + task.exception()")
    print("swallows it, so the run survives -- the cost is a reserved filename and")
    print("a js_count that does not match anything on disk.")


# ===== merged from probe3 =====


# ── 17: schema run, markdown side of the strip_guessed_noise split ────────────
CLASSIFIEDS = (
    "<html><body><div class='classifieds'>"
    "<div class='ad-card'><h3>Blue widget</h3><span class='ad-price'>$5</span></div>"
    "<div class='ad-card'><h3>Red widget</h3><span class='ad-price'>$7</span></div>"
    "</div></body></html>"
)


def test_p17_schema_markdown(tmp_path, monkeypatch):
    hdr("17  END TO END: --schema recovers the text but the manifest's markdown is empty")
    import asyncio

    import protor.engine as engine_mod
    from protor.engine import CrawlEngine, StaticQueue, StaticSource
    from protor.extractor import ExtractionSchema
    from protor.fetcher import FetchResult

    async def fake_fetch(session, url, **kwargs):
        return FetchResult(
            text=CLASSIFIEDS, nbytes=len(CLASSIFIEDS), status=200, content_type="text/html"
        )

    monkeypatch.setattr(engine_mod, "fetch", fake_fetch)

    schema = ExtractionSchema.from_dict(
        {
            "name": "ads",
            "base_selector": ".ad-card",
            "fields": [
                {"name": "title", "selector": "h3"},
                {"name": "price", "selector": ".ad-price"},
            ],
        }
    )

    async def run():
        engine = CrawlEngine(
            queue=StaticQueue(["https://x.example/"]),
            link_source=StaticSource(),
            output_dir=tmp_path,
            max_targets=1,
            extraction_schema=schema,
        )
        return await engine.arun(), engine.manifests

    _, manifests = asyncio.run(run())
    m = manifests[0]
    print(f"extracted_data  : {m.extracted_data}")
    print(f"text_content    : {m.text_content!r}")
    print(f"markdown_content: {m.markdown_content!r}")
    print()
    print("protor.config.MAX_TEXT_CHARS = 10000, MAX_MARKDOWN_CHARS = 40000")
    print("=> the schema recovered both records, the text preview has both widgets,")
    print("   and markdown_content -- which is what gets written to the manifest and")
    print("   to the .md file -- is an empty string. The fix was applied at the")
    print("   clean_soup call site only.")


# ── 18: which malformed hrefs raise, and how innocently ─────────────────────
def test_p18_malformed():
    hdr("18  how innocently can a page's own href make urlparse raise?")
    hrefs = [
        "http://[::1",  # the classic payload
        "//[::1",
        "http://exa]mple.com/",  # one stray bracket in a hostname
        "http://exa[mple.com/",
        "https://en.wikipedia.org/wiki/A_(b)",
        "/wiki/Foo_(bar)",
        "https://例え.jp/",  # IDN, should be fine
        "http://[2001:db8::1]/x",  # valid IPv6
        "mailto:a@b.c",
        "#",
        "?q=1",
        "",
        "  ",
        "http://",
        "https://example.com:99999999999999/x",  # port out of range
        "tel:+441234567890",
        "data:text/html,<b>hi</b>",
    ]
    for h in hrefs:
        try:
            out = extract_links(f"<a href='{h}'>x</a>", "https://example.com/")
            verdict = f"ok -> {out}"
        except Exception as exc:
            verdict = f"RAISES {type(exc).__name__}: {exc}"
        print(f"{h!r:44} {verdict}")

    print()
    print("and urlparse alone:")
    for u in ("http://exa]mple.com/", "http://exa[mple.com/", "https://example.com:99999999999999/x"):
        try:
            p = urlparse(u)
            print(f"  {u!r:44} hostname={p.hostname!r}")
        except Exception as exc:
            print(f"  {u!r:44} RAISES {type(exc).__name__}: {exc}")


# ===== merged from probe4 =====


def test_p19_case_sensitivity_under_a_normal_crawl():
    hdr("19  a page that spells its own host in mixed case loses those absolute links")
    # The crawl queue canonicalises, so this is what parse_html really receives.
    base = "https://example.com/"
    html = (
        "<html><body>"
        "<a href='/relative'>relative</a>"
        "<a href='https://example.com/lowercase'>lowercase</a>"
        "<a href='https://EXAMPLE.com/uppercase'>uppercase</a>"
        "<a href='https://Example.Com/mixed'>mixed</a>"
        "</body></html>"
    )
    print(f"base = {base!r}  (what canonicalize_url produces)")
    for out in extract_links(html, base):
        print(f"  kept: {out}")
    print()
    print("engine.py:339-343 says the same decision is made this way:")
    print("  'Lowercased because the queue canonicalises the host while this")
    print("   arrives from urlparse(), which keeps the case the user typed.'")
    print()
    print("parser.py:215  self._base_domain = urlparse(base_url).netloc   # not lowercased")
    print("parser.py:246  p.netloc == self._base_domain                 # raw comparison")


def test_p20_docstring_claim():
    hdr("20  the module docstring's 'single noise-filtering pass' claim")
    import inspect

    from protor.markdown import _is_noise

    print("parser.py:4-7 claims:")
    print('  "The single noise-filtering pass lives in protor.markdown;')
    print('   this module applies it once..."')
    print()
    print("clean_soup call sites in the parse path:")
    src = inspect.getsource(__import__("protor.parser", fromlist=["x"]))
    for i, line in enumerate(src.splitlines(), 1):
        if "clean_soup(" in line or "soup_to_markdown(" in line:
            print(f"  parser.py:{i}: {line.strip()}")
    print()
    print("_is_noise is called again inside the renderer, with the default:")
    print("  markdown.py:321  if _is_noise(tag):")
    print("  -> strip_guessed_noise is NOT threaded through soup_to_markdown")
    print("  signature:", inspect.signature(__import__("protor.markdown", fromlist=["x"]).soup_to_markdown))
    print()
    print("and _is_noise defaults to:", inspect.signature(_is_noise))


def test_p21_dead_helpers():
    hdr("21  helpers that only tests call any more")
    import protor.parser as P
    import subprocess

    out = subprocess.run(
        ["grep", "-rn", "_extract_metadata\|_extract_js_links\|_extract_internal_links", "protor/"],
        capture_output=True,
        text=True,
        cwd="/home/pulkit/personal/protor",
    )
    print(out.stdout or "(none)")
    print("P._extract_js_links pokes a private attribute instead of using harvest():")
    import inspect

    print(inspect.getsource(P._extract_js_links))
    print("P._extract_internal_links uses harvest():")
    print(inspect.getsource(P._extract_internal_links))
