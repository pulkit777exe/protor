"""
protor.blocklist
~~~~~~~~~~~~~~~~
Domain and ad/tracker request blocking.

Inspired by Scrapling's built-in ad blocking (~3,500 known domains)
and Crawlee's request filtering capabilities.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Blocklist", "is_blocked"]

# ── built-in ad/tracker domains ──────────────────────────────────────────────
# Curated from popular blocklists (EasyList, AdGuard, uBlock Origin patterns)

_ADS_TRACKERS: set[str] = {
    # Major ad networks
    "doubleclick.net",
    "googleadservices.com",
    "googlesyndication.com",
    "google-analytics.com",
    "googletagmanager.com",
    "googletagservices.com",
    "adservice.google.com",
    "pagead2.googlesyndication.com",
    "ad.doubleclick.net",
    "stats.g.doubleclick.net",
    # Facebook tracking
    "facebook.com",
    "facebook.net",
    "fbcdn.net",
    "connect.facebook.net",
    "pixel.facebook.com",
    # Social trackers
    "twitter.com",
    "platform.twitter.com",
    "syndication.twitter.com",
    "analytics.twitter.com",
    "t.co",
    "linkedin.com",
    "snap.licdn.com",
    # Analytics
    "hotjar.com",
    "script.hotjar.com",
    "vars.hotjar.com",
    "segment.io",
    "cdn.segment.com",
    "amplitude.com",
    "api.amplitude.com",
    "mixpanel.com",
    "cdn.mxpnl.com",
    "heap.io",
    "heapanalytics.com",
    "optimizely.com",
    "cdn.optimizely.com",
    "fullstory.com",
    "rs.fullstory.com",
    "mouseflow.com",
    "cdn.mouseflow.com",
    "crazyegg.com",
    "script.crazyegg.com",
    "clicktale.com",
    "static.clicktale.com",
    # Ad tech
    "amazon-adsystem.com",
    "aax.amazon-adsystem.com",
    "ads-twitter.com",
    "outbrain.com",
    "widgets.outbrain.com",
    "taboola.com",
    "cdn.taboola.com",
    "scorecardresearch.com",
    "sb.scorecardresearch.com",
    "moatads.com",
    "cdn.moatads.com",
    "quantserve.com",
    "secure.quantserve.com",
    "bluekai.com",
    "stags.bluekai.com",
    "demdex.net",
    "adobedemdex.com",
    "rubiconproject.com",
    "fastlane.rubiconproject.com",
    "pubmatic.com",
    "ads.pubmatic.com",
    "openx.net",
    "tagid.openx.net",
    "casalemedia.com",
    "ib.adnxs.com",
    "criteo.com",
    "ads.criteo.com",
    "bidswitch.net",
    "rtbsystem.com",
    # Misc trackers
    "newrelic.com",
    "js-agent.newrelic.com",
    "sentry.io",
    "browser.sentry-cdn.com",
    "chartbeat.com",
    "static.chartbeat.com",
    "parsely.com",
    "api.parsely.com",
    "bounceexchange.com",
    "cdn.bounceexchange.com",
    "zarget.com",
    "cdn.zarget.com",
    "vero.com",
    "d10lpsik1i8c69.cloudfront.net",
    "intercom.io",
    "widget.intercom.io",
    "drift.com",
    "js.driftt.com",
    "hubspot.com",
    "js.hs-scripts.com",
    "marketo.com",
    "app-ab14.marketo.com",
    "pardot.com",
    "pi.pardot.com",
    # Cryptomining
    "coinhive.com",
    "coin-hive.com",
    "cryptoloot.pro",
    "crypto-loot.com",
}

# Patterns for ad-related URL paths
_AD_PATH_PATTERNS = re.compile(
    r"/(ads?|ad[-_]vert|banner|pixel|track|beacon|analytics|统计|stat|log)/",
    re.IGNORECASE,
)

# Patterns for common ad-related file extensions
_AD_FILE_PATTERNS = re.compile(
    r"\.(gif|png|jpg|jpeg|webp)(\?.*)?$",
    re.IGNORECASE,
)


class Blocklist:
    """
    Domain and URL blocklist for filtering requests.

    Supports domain-level blocking, pattern-based blocking,
    and custom rules.

    Parameters
    ----------
    extra_blocked:
        Additional domains to block beyond the built-in list.
    block_ads:
        Whether to enable the built-in ad/tracker blocklist.
    custom_patterns:
        Additional regex patterns for URL-level blocking.
    """

    def __init__(
        self,
        extra_blocked: Sequence[str] | None = None,
        block_ads: bool = True,
        custom_patterns: Sequence[str] | None = None,
    ) -> None:
        self._block_ads = block_ads
        self._blocked_domains: set[str] = set(_ADS_TRACKERS) if block_ads else set()
        self._patterns: list[re.Pattern[str]] = []

        if extra_blocked:
            for d in extra_blocked:
                self._blocked_domains.add(d.lower().strip("."))

        if custom_patterns:
            for p in custom_patterns:
                self._patterns.append(re.compile(p, re.IGNORECASE))

    def is_domain_blocked(self, domain: str) -> bool:
        """Check if a domain (or its parent) is blocked."""
        domain = domain.lower().strip(".")
        # Check exact match and parent domains
        parts = domain.split(".")
        for i in range(len(parts)):
            candidate = ".".join(parts[i:])
            if candidate in self._blocked_domains:
                return True
        return False

    def is_url_blocked(self, url: str) -> bool:
        """Check if a URL should be blocked based on domain or pattern rules."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Strip port
        if ":" in domain:
            domain = domain.rsplit(":", 1)[0]

        if self.is_domain_blocked(domain):
            return True

        return any(pattern.search(url) for pattern in self._patterns)

    @property
    def blocked_count(self) -> int:
        """Number of blocked domains."""
        return len(self._blocked_domains)

    def add_domain(self, domain: str) -> None:
        """Add a domain to the blocklist at runtime."""
        self._blocked_domains.add(domain.lower().strip("."))

    def remove_domain(self, domain: str) -> None:
        """Remove a domain from the blocklist."""
        self._blocked_domains.discard(domain.lower().strip("."))

    def add_pattern(self, pattern: str) -> None:
        """Add a URL pattern to block."""
        self._patterns.append(re.compile(pattern, re.IGNORECASE))

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> Blocklist:
        """
        Load a blocklist from a file (one domain per line).

        Lines starting with # are treated as comments.
        """
        blocked: list[str] = []
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    blocked.append(line)
        return cls(extra_blocked=blocked, **kwargs)


def is_blocked(url: str, block_ads: bool = True) -> bool:
    """Quick check if a URL is from a known ad/tracker domain."""
    bl = Blocklist(block_ads=block_ads)
    return bl.is_url_blocked(url)
