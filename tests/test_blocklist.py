import tempfile

from protor.blocklist import Blocklist, is_blocked


def test_default_blocked_domains():
    bl = Blocklist()
    assert bl.is_domain_blocked("doubleclick.net")
    assert bl.is_domain_blocked("google-analytics.com")
    assert bl.is_domain_blocked("facebook.com")


def test_not_blocked():
    bl = Blocklist()
    assert not bl.is_domain_blocked("github.com")
    assert not bl.is_domain_blocked("example.com")


def test_is_url_blocked():
    bl = Blocklist()
    assert bl.is_url_blocked("https://doubleclick.net/ad")
    assert not bl.is_url_blocked("https://github.com/repo")


def test_add_domain():
    bl = Blocklist()
    bl.add_domain("myblock.com")
    assert bl.is_domain_blocked("myblock.com")


def test_remove_domain():
    bl = Blocklist()
    bl.add_domain("myblock.com")
    assert bl.is_domain_blocked("myblock.com")
    bl.remove_domain("myblock.com")
    assert not bl.is_domain_blocked("myblock.com")


def test_from_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("badsite.com\nanotherbad.org\n")
        f.flush()
        bl = Blocklist.from_file(f.name)
        assert bl.is_domain_blocked("badsite.com")
        assert bl.is_domain_blocked("anotherbad.org")
        assert not bl.is_domain_blocked("goodsite.com")


def test_from_file_not_found():
    bl = Blocklist.from_file("/nonexistent/file.txt")
    assert not bl.is_domain_blocked("anything.com")


def test_blocked_count():
    bl = Blocklist(block_ads=False)
    assert bl.blocked_count == 0
    bl.add_domain("test1.com")
    bl.add_domain("test2.com")
    assert bl.blocked_count == 2


def test_add_pattern():
    bl = Blocklist(custom_patterns=[r"ad[s]?\.\w+"])
    assert bl.is_url_blocked("https://example.com/ad.gif")
    assert not bl.is_url_blocked("https://example.com/page.html")


def test_is_blocked_function():
    assert is_blocked("https://doubleclick.net/ad")
    assert not is_blocked("https://github.com/repo")


def test_subdomain_matching():
    bl = Blocklist()
    assert bl.is_domain_blocked("sub.doubleclick.net")
    assert bl.is_domain_blocked("tracking.google-analytics.com")


def test_custom_blocklist_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("# comment\n\nbadsite.com\n\nanotherbad.org\n")
        f.flush()
        bl = Blocklist.from_file(f.name)
        assert bl.is_domain_blocked("badsite.com")
        assert bl.is_domain_blocked("anotherbad.org")
        assert not bl.is_domain_blocked("goodsite.com")


def test_block_ads_disabled():
    bl = Blocklist(block_ads=False)
    assert not bl.is_domain_blocked("doubleclick.net")


def test_extra_blocked():
    bl = Blocklist(extra_blocked=["mybad.com"], block_ads=False)
    assert bl.is_domain_blocked("mybad.com")


def test_empty_url():
    bl = Blocklist()
    assert not bl.is_url_blocked("")


def test_is_blocked_function_with_block_ads_false():
    assert not is_blocked("https://doubleclick.net/ad", block_ads=False)
