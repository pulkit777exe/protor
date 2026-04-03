# Changelog

## [v2.3.0] — 2026-04-03

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
