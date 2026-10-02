# Changelog

## [v2.7.0] — 2026-10-02

A review pass over the whole pipeline, driven by measurements rather than
inspection. Every change below was reproduced with a benchmark or a failing
test first.

### Fixed — output quality

- **Streamed LLM output was being corrupted.** Analysis chunks were printed with
  rich's default `markup=True`, so a Markdown link `[docs](url)` rendered as
  `(url)`, `` `[code]` `` vanished, and an unbalanced `[/tag]` raised
  `MarkupError` — losing the entire report *after* the model had been paid for.
  Now printed with `markup=False, highlight=False`, so the terminal matches the
  saved file byte for byte.
- **Analysis silently dropped sites.** `_prepare_context` used a flat 1,500-char
  preview plus a global 8,000-char cut, so 10 sites sent only 5 while the header
  still claimed "Sites: 10". The budget is now split evenly across sites.
- **Markdown shredded paragraphs.** `<p>Protor is a <strong>fast</strong>
  scraper…</p>` became five separate lines. Inline content is now folded into a
  single line, with emphasis, links, and code preserved.
- **Adjacent lists merged.** A `<ul>` followed by an `<ol>` had no blank line
  between them, so the ordered list rendered as part of the bullet list.
- **Document order was violated.** Loose inline content was emitted before
  nested blocks, hoisting trailing links above earlier headings.
- **JavaScript files overwrote each other.** Two origins serving `vendor.js`
  collapsed to one file while the manifest still listed both URLs. Names now
  carry a short URL hash when the basename is taken.
- **Crawled pages overwrote each other.** `/docs/a.html` and `/blog/a.html`
  both wrote `a.html` and `a.manifest.json`. Filenames are now derived from the
  full path (`docs-a.html`).
- **`--max-pages` was not a real limit.** The ceiling counted only successes, so
  `--max-pages 10` issued 14 requests. Failures and blocked pages now count.
- **Checkpoints fired before any work**, and again every round while the count
  was unchanged (`scraped % n == 0` is true at zero).
- **`protor scrape` followed by `protor analyze` failed.** `analyze` defaulted
  to `data/sites_index.json` while `scrape` writes to `~/Downloads/protor`.
  The defaults now agree.
- **`run` silently ignored `--format`**, and lacked `--format`, `--prompt`, and
  `--prompt-file`, which `analyze` had. Both now share one flag helper.
- **`--prompt-file` with a bad path raised an uncaught traceback.**
- **Off-domain URLs vanished** with no stat and no log line; they are now
  recorded as `skipped`.
- **The crawl progress bar was one cell per page**, so `--max-pages 500`
  rendered a 500-character bar. Now scaled to a fixed width.
- **Cached pages reported `0 B`** and rendered as `—` in the results table.
- **`Accept-Encoding` advertised brotli** without the dependency installed, so
  servers that honoured it returned bodies aiohttp could not decode. aiohttp now
  negotiates on its own.

### Fixed — performance

- **The noise filter ran 3× per page** (once in `parse_soup`, once in
  `_extract_text`, once in `soup_to_markdown`). Now exactly once:
  **87 ms → 44 ms** per 14 KB page.
- **The rate limiter did not rate limit.** It read a timestamp, slept, then
  wrote the new one, so every concurrent task to one domain read the same stale
  value and fired together: 8 requests with a 0.5 s delay completed in 0.50 s
  instead of 3.5 s. Each waiter now reserves its own slot under a per-domain
  lock; different domains still never block each other.
- **`page_delay` slept the whole crawl loop** after every completion round,
  costing 3.7× throughput (2.21 s vs 0.68 s for 24 pages). Removed; the
  crawler now uses the per-domain rate limiter like the scraper does.
- **The HTTP cache rewrote its entire index on every page**, bodies included —
  quadratic. 60 pages of 50 KB wrote ~172 MB and blocked the event loop for
  15 ms per page. Bodies now live in one file per URL and the index is written
  once per run: **15.4 ms → 0.10 ms** per put. Caching is now opt-in
  (`--cache`) instead of always on.
- **robots.txt was fetched once per concurrent page** because the cache was
  checked before the await: 6 concurrent checks meant 6 requests. Now
  single-flight, so one crawl fetches it once.
- **The crawl queue fsynced three times per page** on the event loop. Now WAL
  journal, batched commits, and in-memory counters instead of `COUNT(*)` per
  admission check: **765 µs → 47 µs** per page.
- **Schema extraction re-parsed the HTML** with lxml even though the engine
  already held the tree. Added `extract_from_soup`.
- **The noise regex was recompiled per tag.** Hoisted to module scope.

### Changed

- `scrape_site_async` is now a thin wrapper over `CrawlEngine`. It was a second
  copy of the pipeline that had already drifted: it ignored `block_ads`, skipped
  the robots check, wrote a hard-coded `manifest.json`, and dropped `timeout`.
- The engine accepts an optional `session`, so callers holding a connection pool
  no longer open a second one.
- `page_delay` removed from `CrawlEngine`; use `rate_limiter`.
- Dropped the `aioresponses` dev dependency — 0.7.9 (latest) is incompatible
  with aiohttp 3.14. Tests use purpose-built fakes in `tests/conftest.py`.

### Tests

- **344 passing, zero warnings** (was 317 with 4 warnings). Coverage 85%.
- **Four tests were vacuous**: `AsyncMock` used as a *sync* context manager never
  entered its body, so `async with session.get(...)` never ran and the tests
  passed through the `except Exception` fallback. Rewritten against real async
  context-manager doubles — which immediately exposed that nothing had been
  verifying robots.txt rules at all.
- New `tests/test_regressions.py` pins each defect fixed above.
- Removed tests that asserted removed behaviour (`to_dict` carrying bodies).

### Removed

- Dead config: `SCALING_ENABLED` was never read; `ANALYSIS_TIMEOUT` and
  `OLLAMA_CHECK_TIMEOUT` were hardcoded at their use sites and are now the single
  source of truth.

## [v2.6.0] — 2026-07-08

### New Features
- Add Markdown output via `--markdown` flag — clean, readable conversion with noise removal
- Add crawl checkpoint/resume — `--resume` picks up where a crawl left off
- Add content filtering — `--content-filter` strips ads, nav, footer, and boilerplate
- Add SQLite-backed crawl queue — BFS ordering with persistent visited tracking
- Add User-Agent rotation — randomizes UA per request from a pool of 15 real browsers
- Add schema-based extraction — `--schema schema.json` extracts structured JSON from HTML
- Add domain/ad blocking — `--block-ads` blocks 100+ known ad/tracker domains out of the box
- Add hook system — `before_fetch`, `after_fetch`, `before_parse`, `after_parse` callbacks
- Add auto-scaling concurrency — `--auto-scale` dynamically adjusts workers based on latency
- Add `extract` command for standalone schema-based extraction from saved HTML files

### Open Source
- Add CONTRIBUTING.md with development setup, testing, and contribution guidelines
- Add GitHub issue templates (bug report + feature request)
- Add CODE_OF_CONDUCT.md
- Fix license format to SPDX expression in pyproject.toml
- Add Python 3.13 to classifiers

### Test Improvements
- Add 72 new tests for markdown, blocklist, and extractor modules (210 → 282 total)
- Increase test coverage from 69% to 79%

### Code Quality
- Fix all ruff lint errors and format issues
- Clean up unused imports across all test files

## [v2.5.0] — 2026-04-04

### New Features
- Add multi-LLM backend support: OpenAI and Anthropic alongside Ollama
- Add `llm_backends.py` module with abstract `LLMBackend` class and concrete implementations
- Add factory function `create_backend()` for easy backend switching

### Test Improvements
- Add 97 new tests (113 → 210 total), covering formatters, HTTP cache, robots.txt, rate limiter, LLM backends, scraper internals, and CLI handlers
- Increase test coverage from 61% to 84%
- Fix all pre-existing test failures and mock path issues

### Code Quality
- Fix all ruff lint issues (89 auto-fixed + 8 manual)
- Add proper exception chaining (`raise ... from exc`) throughout codebase
- Remove duplicate code (ROBOTS_PATCH, RATE_LIMIT_DELAY)
- Sort all import blocks, remove unused imports, add trailing newlines
- Replace ambiguous variable names, remove empty TYPE_CHECKING blocks

### Bug Fixes
- Fix duplicate optional dependency entries in pyproject.toml
- Fix `test_integration.py` mock paths for refactored CLI imports
- Fix `test_analyzer.py` to use `_stream_backend` instead of removed `_stream`
- Fix end-to-end test to use actual async `scrape_site_async` API

## [v2.4.0] — 2026-04-03

### New Features
- Add progress callbacks for library users
- Add custom headers support for auth tokens
- Add HTTP caching to avoid re-fetching unchanged pages
- Add custom prompts via CLI `--prompt` flag or file `--prompt-file`

## [v2.2.0] — 2026-04-03

### New Features
- Add `update` command to check for and install updates from PyPI
- Add per-domain rate limiting with configurable politeness delays
- Add robots.txt support — blocks scraping of disallowed URLs
- Add `python -m protor` support via `__main__.py`
- Add URL validation at CLI entry points with helpful error messages

### Improvements
- Migrate all hardcoded values to `config.py` (crawler delay, concurrency, headers, Ollama base, max data chars)
- Improve `human_bytes` precision using float division
- Update CI workflow: replace flake8/black with ruff, add mypy type-checking
- Support Python 3.11, 3.12, 3.13 in CI matrix

### Bug Fixes
- Fix pre-existing test failures in `test_scraper.py`, `test_crawler.py`, `test_cli.py`, `test_analyzer.py`
- Fix `_abort` test that incorrectly expected `SystemExit` when `sys.exit` was mocked
- Add missing `mock_ollama_response` fixture to `conftest.py`
- Register `asyncio` marker in pytest configuration

## [v2.1.0] — 2026-04-03

### New Features
- Add `update` command to check for and install updates from PyPI
- Support `--check` flag for version info only
- Support `-y` flag to skip confirmation prompt
- Detect editable installs and show appropriate update instructions

## [v2.0.0] — 2026-04-02

### Breaking Changes
- Migrate from curl-based scraping to async aiohttp
- Python 3.11+ required (was 3.8+)
- Default output directory changed to `~/Downloads/protor`

### New Features
- Add crawler with BFS algorithm and live progress display
- Add typed dataclasses: `SiteManifest`, `SiteMetadata`, `AnalysisResult`
- Implement proper exception hierarchy with typed errors
- Add Rich theme system for consistent CLI output
- Refactor CLI to use subcommand pattern with proper error handling
- Add `version` command
- Add concurrency control with `--concurrency` flag

### Security Fixes
- Enable SSL certificate verification (was disabled)
- Add prompt injection protection in analyzer prompts

### Bug Fixes
- Fix asyncio.gather re-await pattern in scraper orchestrator
- Fix silent error swallowing in crawler — errors now logged with details
- Fix cross-platform path resolution using `Path.home()`
- Fix error count reporting in scraper summary

### Improvements
- Modernize pyproject.toml with full tool configurations (ruff, mypy, pytest, coverage)
- Update test suite with proper fixtures and async support
- Add Dockerfile for containerized deployment
- Add comprehensive docstrings across all modules
- Cleaner, more maintainable codebase with proper type hints

### Internal
- Remove `requirements-dev.txt` — use `pyproject.toml` optional deps
- Add `protor/__init__.py` with version detection
- Add `protor/exceptions.py` for typed errors
- Add `protor/models.py` for data classes
- Add `protor/theme.py` for Rich console theming
