"""Sitemap reading, against real servers rather than fixtures.

A sitemap is the one input a crawler trusts to describe its own site, so the
shapes worth testing are the ones seen in the wild: namespaced documents, a
``<sitemapindex>`` pointing at more sitemaps, gzip served with the wrong
Content-Type, and documents that are not XML at all.
"""

import gzip

import pytest
from aiohttp import web

from protor.sitemap import default_sitemap, discover_sitemap_urls, iter_sitemap_urls

NS = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'


def urlset(entries: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<url><loc>{loc}</loc>" + (f"<lastmod>{mod}</lastmod>" if mod else "") + "</url>"
        for loc, mod in entries
    )
    return f'<?xml version="1.0"?><urlset {NS}>{body}</urlset>'


def sitemapindex(locs: list[str]) -> str:
    body = "".join(f"<sitemap><loc>{loc}</loc></sitemap>" for loc in locs)
    return f'<?xml version="1.0"?><sitemapindex {NS}>{body}</sitemapindex>'


class Site:
    """A server serving whatever routes the test puts on it."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, str, bytes | str]] = {}

    def add(self, path: str, body: str, *, status: int = 200, gz: bool = False) -> None:
        payload = gzip.compress(body.encode()) if gz else body
        self.routes[path] = (status, "application/gzip" if gz else "application/xml", payload)

    async def handler(self, request: web.Request) -> web.Response:
        entry = self.routes.get(request.path)
        if entry is None:
            return web.Response(status=404, text="not found")
        status, ctype, payload = entry
        return web.Response(status=status, body=payload, content_type=ctype)

    async def start(self) -> str:
        app = web.Application()
        app.router.add_get("/{tail:.*}", self.handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.runner = runner
        self.port = runner.addresses[0][1]
        return self.base

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"


@pytest.fixture
async def site():
    s = Site()
    await s.start()
    try:
        yield s
    finally:
        await s.runner.cleanup()


@pytest.mark.asyncio
async def test_a_plain_urlset_is_read(site):
    site.add(
        "/sitemap.xml", urlset([("https://x.example/a", ""), ("https://x.example/b", "2026-01-02")])
    )
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = [e async for e in iter_sitemap_urls(session, [site.url("/sitemap.xml")])]

    assert found == [("https://x.example/a", ""), ("https://x.example/b", "2026-01-02")]


@pytest.mark.asyncio
async def test_a_namespaced_document_is_read(site):
    """The namespace prefix varies by site, so matching must use the local name."""
    prefixed = (
        '<?xml version="1.0"?>'
        '<sm:urlset xmlns:sm="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<sm:url><sm:loc>https://x.example/a</sm:loc></sm:url>"
        "</sm:urlset>"
    )
    site.add("/sitemap.xml", prefixed)
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = [e async for e in iter_sitemap_urls(session, [site.url("/sitemap.xml")])]

    assert found == [("https://x.example/a", "")]


@pytest.mark.asyncio
async def test_a_sitemapindex_is_followed(site):
    site.add("/sitemap.xml", sitemapindex([site.url("/s1.xml"), site.url("/s2.xml")]))
    site.add("/s1.xml", urlset([("https://x.example/one", "")]))
    site.add("/s2.xml", urlset([("https://x.example/two", "")]))
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = [e async for e in iter_sitemap_urls(session, [site.url("/sitemap.xml")])]

    assert [u for u, _ in found] == ["https://x.example/one", "https://x.example/two"]


@pytest.mark.asyncio
async def test_a_gzipped_sitemap_is_read_despite_the_content_type(site):
    """
    `.xml.gz` is routinely served as application/xml, so the magic bytes decide.

    Trusting the header here means handing gzip to the XML parser, which fails
    to parse and silently yields nothing.
    """
    site.add("/sitemap.xml.gz", urlset([("https://x.example/a", "")]), gz=True)
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = [e async for e in iter_sitemap_urls(session, [site.url("/sitemap.xml.gz")])]

    assert found == [("https://x.example/a", "")]


@pytest.mark.asyncio
async def test_a_gzipped_sitemap_served_as_plain_xml_is_read(site):
    """The header lies; the bytes do not."""
    site.add("/sitemap.xml", urlset([("https://x.example/a", "")]), gz=True)
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = [e async for e in iter_sitemap_urls(session, [site.url("/sitemap.xml")])]

    assert found == [("https://x.example/a", "")]


@pytest.mark.asyncio
async def test_a_missing_sitemap_yields_nothing_rather_than_raising(site):
    """Discovering pages is an optimisation, never a precondition."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = [e async for e in iter_sitemap_urls(session, [site.url("/nope.xml")])]
    assert found == []


@pytest.mark.asyncio
async def test_a_document_that_is_not_xml_yields_nothing(site):
    site.add("/sitemap.xml", "<html><body>not a sitemap</body></html>")
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = [e async for e in iter_sitemap_urls(session, [site.url("/sitemap.xml")])]
    assert found == []


@pytest.mark.asyncio
async def test_an_error_status_yields_nothing(site):
    site.add("/sitemap.xml", urlset([("https://x.example/a", "")]), status=503)
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = [e async for e in iter_sitemap_urls(session, [site.url("/sitemap.xml")])]
    assert found == []


@pytest.mark.asyncio
async def test_two_indexes_naming_each_other_terminate(site):
    """Depth-capped, so a cycle costs a few requests rather than hanging."""
    site.add("/a.xml", sitemapindex([site.url("/b.xml")]))
    site.add("/b.xml", sitemapindex([site.url("/a.xml"), site.url("/leaf.xml")]))
    site.add("/leaf.xml", urlset([("https://x.example/leaf", "")]))
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = [e async for e in iter_sitemap_urls(session, [site.url("/a.xml")])]

    assert [u for u, _ in found] == ["https://x.example/leaf"]


@pytest.mark.asyncio
async def test_the_limit_stops_the_generator_early(site):
    """A crawl has its own page ceiling; parsing 50,000 URLs to return 10 is waste."""
    site.add(
        "/sitemap.xml",
        urlset([(f"https://x.example/p{i}", "") for i in range(500)]),
    )
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = [e async for e in iter_sitemap_urls(session, [site.url("/sitemap.xml")], limit=5)]

    assert len(found) == 5


@pytest.mark.asyncio
async def test_relative_locations_are_resolved(site):
    site.add("/sitemap.xml", urlset([("/relative/page", "")]))
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = [e async for e in iter_sitemap_urls(session, [site.url("/sitemap.xml")])]

    assert found == [(site.url("/relative/page"), "")]


@pytest.mark.asyncio
async def test_robots_sitemaps_are_preferred_over_the_default(site):
    class _Robots:
        def __init__(self, maps):
            self._maps = maps

        async def sitemaps(self, base, session):
            return list(self._maps)

    site.add("/from-robots.xml", urlset([("https://x.example/robots-listed", "")]))
    site.add("/sitemap.xml", urlset([("https://x.example/default", "")]))
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = await discover_sitemap_urls(
            session, site.base, robots=_Robots([site.url("/from-robots.xml")])
        )
    assert [u for u, _ in found] == ["https://x.example/robots-listed"]


@pytest.mark.asyncio
async def test_the_conventional_sitemap_is_the_fallback(site):
    site.add("/sitemap.xml", urlset([("https://x.example/default", "")]))
    import aiohttp

    async with aiohttp.ClientSession() as session:
        found = await discover_sitemap_urls(session, site.base, robots=None)
    assert [u for u, _ in found] == ["https://x.example/default"]


def test_default_sitemap_is_the_origin_root():
    assert default_sitemap("https://x.example/deep/page?a=1") == "https://x.example/sitemap.xml"
    assert default_sitemap("not a url") == ""
