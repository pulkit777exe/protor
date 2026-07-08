# Contributing to protor

Thanks for wanting to help out. Here's how to get started.

## Development setup

```bash
# clone the repo
git clone https://github.com/pulkit777exe/protor.git
cd protor

# install in editable mode with dev dependencies
pip install -e ".[dev]"

# verify everything works
protor version
pytest
```

## Running tests

```bash
# run all tests
pytest

# run a specific test file
pytest tests/test_scraper.py

# run with coverage report
pytest --cov=protor --cov-report=term-missing

# skip slow tests
pytest -m "not slow"
```

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# check for lint errors
ruff check

# auto-fix lint errors
ruff check --fix

# format code
ruff format

# check formatting without changing files
ruff format --check
```

All code must pass `ruff check` and `ruff format --check` before merging.

## Type checking

```bash
mypy protor/
```

## Project structure

```
protor/
├── cli.py          # CLI entry point and subcommands
├── scraper.py      # async HTML + JS scraping
├── crawler.py      # BFS site crawler with checkpointing
├── analyzer.py     # LLM analysis (Ollama/OpenAI/Anthropic)
├── extractor.py    # schema-based structured data extraction
├── markdown.py     # HTML to Markdown converter
├── blocklist.py    # ad/tracker domain blocking
├── models.py       # typed dataclasses
├── exceptions.py   # error hierarchy
├── config.py       # centralized constants
├── llm_backends.py # multi-backend LLM abstraction
├── theme.py        # Rich console theming
├── http_cache.py   # conditional HTTP caching
├── robots.py       # robots.txt support
├── rate_limiter.py # per-domain rate limiting
├── updater.py      # PyPI update checker
├── formatters.py   # output formatting
└── utils.py        # helper functions
```

## Adding a feature

1. Create a branch: `git checkout -b my-feature`
2. Write tests first if possible
3. Implement the feature
4. Run `ruff check --fix && ruff format`
5. Run `pytest` to make sure everything passes
6. Commit with a clear message
7. Open a PR

## Adding tests

Tests live in `tests/`. We use `pytest` with `pytest-asyncio` for async tests.

```bash
# run just the new tests
pytest tests/test_my_module.py -v
```

Name test files `test_<module>.py` and test functions `test_<what_it_tests>`.

## Reporting bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your OS and Python version

## Code of conduct

Be respectful. We're all here to build cool stuff.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
