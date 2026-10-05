"""
Which redirect targets a scraper should refuse to follow.

A fetcher that follows redirects unconditionally will fetch whatever a remote
server tells it to. Scraping a site that answers with ``302 Location:
http://169.254.169.254/latest/meta-data/`` sends the request to the host's
cloud metadata service and stores the credentials it returns in the output
directory as if they were a web page. Nothing in the output distinguishes that
from a successful scrape.

The guard is deliberately narrow, because protor exists to scrape local and
intranet sites — ``http://localhost:8000`` and ``http://wiki.internal/`` are
use cases, not mistakes. So it refuses only targets that are never a legitimate
scrape destination:

* **Link-local addresses** (169.254.0.0/16, fe80::/10), which is where cloud
  metadata lives.
* **The metadata addresses themselves**, named explicitly so a redirect to an
  IPv6 or otherwise unusual form of one cannot slip past the range check.
* **Non-HTTP schemes**, so a redirect cannot move the fetch to ``file:``,
  ``ftp:`` or ``gopher:``.

Loopback and RFC 1918 space are allowed. Blocking those would break the tool's
own documented purpose, and a user who types an internal URL has already made
that decision.

Residual risk, stated plainly: a redirect to a *hostname* that resolves to an
internal address is not caught. Catching it means resolving DNS on every
redirect, which costs a lookup per hop and is itself defeatable by rebinding
the name between the check and the connection. That trade is not worth making
here; the range check catches the addresses that are actually written as
addresses in an attack, which is how link-local targets are named in practice.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

__all__ = [
    "BLOCKED_SCHEMES",
    "describe_block",
    "is_blocked_redirect",
    "is_metadata_host",
]

#: Schemes a redirect may not switch the fetch to.
BLOCKED_SCHEMES = frozenset({"file", "ftp", "gopher", "data", "jar", "netdoc"})

#: Host names that exist to hand out credentials.
_METADATA_HOSTS = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "metadata",
    }
)

#: Addresses that serve instance metadata regardless of the cloud.
_METADATA_ADDRESSES = frozenset({"169.254.169.254", "169.254.170.2", "fd00:ec2::254"})


def _address_of(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """
    Parse *host* as an IP literal, or None when it is a name.

    Normalised first — brackets stripped, lowercased, any trailing dot dropped —
    because the literal and its trailing-dot spelling are the same address, and
    ``ipaddress`` rejects the second: ``169.254.1.1.`` raised, so the link-local
    range check silently did not run and ``http://169.254.1.1./`` was allowed where
    ``http://169.254.1.1/`` was refused. The divergence sat exactly at the boundary
    this module exists to enforce.

    Normalising here rather than at the call site fixes every caller at once;
    ``is_metadata_host`` already did it, which is why the named-address check was
    never the thing that let a variant through.
    """
    candidate = host.strip("[]").lower().rstrip(".")
    if not candidate:
        return None
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        return None


def is_metadata_host(host: str) -> bool:
    """
    True when *host* is a known metadata endpoint, by name or by address.

    Four hostnames and a handful of literal addresses — **no name resolution**.
    That is a real limit and worth stating plainly: a hostname that *would* resolve
    to ``169.254.169.254`` is not caught here, and cannot be, because resolving it
    is itself the DNS request an SSRF filter must not make. The module header
    records the consequence, which is that a redirect to such a hostname is only
    caught once it has been resolved and re-checked as an address.
    """
    if not host:
        return False
    name = host.strip("[]").lower().rstrip(".")
    if name in _METADATA_HOSTS:
        return True
    addr = _address_of(name)
    return addr is not None and str(addr) in _METADATA_ADDRESSES


def describe_block(url: str) -> str | None:
    """
    Return why *url* must not be followed as a redirect, or None if it is fine.

    The message is written to be shown to a user, so it names the address and
    the reason rather than just refusing.
    """
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        return f"it is not a valid URL ({exc})"

    scheme = (parsed.scheme or "").lower()
    if scheme in BLOCKED_SCHEMES:
        return f"it redirects to the {scheme!r} scheme, which is never a web page"
    if scheme not in ("http", "https"):
        return f"it redirects to the unsupported scheme {scheme or '(none)'!r}"

    host = parsed.hostname or ""
    if is_metadata_host(host):
        return "it redirects to a cloud instance-metadata endpoint"

    addr = _address_of(host)
    if addr is not None and addr.is_link_local:
        return f"it redirects to the link-local address {addr}"

    return None


def is_blocked_redirect(url: str) -> bool:
    """True when *url* must not be followed as a redirect."""
    return describe_block(url) is not None
