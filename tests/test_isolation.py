"""
The leak detector guarding the test suite.

`_no_leaked_module_patches` in `tests/conftest.py` exists because a test once
patched `protor.engine.fetch` and restored `parse_html` in a `finally` but not
`fetch`. That passed on its own, and then failed a schema-extraction test in
another file with zero records, four hundred tests later. The failure reads as a
product bug in the extractor.

A detector that nobody checks still detects: it can stop finding things the day
the module layout changes, and it will report a clean suite while doing it.
"""

from __future__ import annotations

from tests.conftest import _module_snapshot


def test_the_snapshot_notices_a_swapped_module_attribute():
    """
    The whole mechanism is comparing identities before and after a test, so
    that comparison is what has to be pinned.
    """
    import protor.engine as engine

    real = engine.parse_html
    before = _module_snapshot()
    assert before[("protor.engine", "parse_html")] == id(real)

    engine.parse_html = lambda *a, **k: None  # type: ignore[assignment]
    try:
        after = _module_snapshot()
        changed = [key for key, ident in after.items() if before.get(key, ident) != ident]
        assert ("protor.engine", "parse_html") in changed, changed
    finally:
        engine.parse_html = real

    assert _module_snapshot()[("protor.engine", "parse_html")] == id(real)


def test_the_snapshot_ignores_constants():
    """
    Constants are the one thing a test may legitimately rebind and leave, and
    caches are initialised on first use rather than leaked. Excluding both is what
    keeps the detector quiet enough to be worth running.
    """
    import protor.engine as engine
    from protor.config import RETRY_BACKOFF_BASE

    assert not any(key[1].isupper() for key in _module_snapshot()), "an uppercase attr slipped in"

    engine.RETRY_BACKOFF_BASE = RETRY_BACKOFF_BASE + 1
    try:
        assert not any(key[1].isupper() for key in _module_snapshot())
    finally:
        engine.RETRY_BACKOFF_BASE = RETRY_BACKOFF_BASE


def test_the_snapshot_covers_the_modules_that_get_patched():
    """If it misses a module, a leak in that module goes unreported."""
    names = {key[0] for key in _module_snapshot()}
    for module in ("protor.engine", "protor.scraper", "protor.markdown", "protor.parser"):
        assert module in names, f"{module} is not covered"
