# Changelog

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
