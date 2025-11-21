# Protor Testing

This directory contains the test suite for the Protor web scraper project.

## Test Structure

```
tests/
├── __init__.py           # Test package initialization
├── conftest.py           # Pytest fixtures and configuration
├── test_utils.py         # Tests for utility functions
├── test_scraper.py       # Tests for scraping functionality
├── test_analyzer.py      # Tests for analysis functionality
└── test_integration.py   # Integration and end-to-end tests
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=protor --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/test_utils.py
```

### Run Specific Test Class

```bash
pytest tests/test_utils.py::TestSafeFilename
```

### Run Specific Test

```bash
pytest tests/test_utils.py::TestSafeFilename::test_basic_alphanumeric
```

### Run Only Integration Tests

```bash
pytest -m integration
```

### Run Excluding Slow Tests

```bash
pytest -m "not slow"
```

## Test Coverage

The test suite aims for high coverage across all modules:

- **utils.py**: Utility functions (safe_filename, save_json, etc.)
- **scraper.py**: Web scraping functionality
- **analyzer.py**: LLM analysis integration
- **crawler.py**: Recursive crawling
- **cli.py**: Command-line interface

## Writing Tests

### Test Naming Convention

- Test files: `test_<module>.py`
- Test classes: `Test<Functionality>`
- Test methods: `test_<specific_behavior>`

### Using Fixtures

Common fixtures are defined in `conftest.py`:

```python
def test_example(temp_dir, sample_html):
    # temp_dir provides a temporary directory
    # sample_html provides sample HTML content
    pass
```

### Mocking External Dependencies

```python
@patch('protor.scraper.fetch_with_curl')
def test_with_mock(mock_fetch):
    mock_fetch.return_value = ("<html></html>", True)
    # Test code here
```

## Continuous Integration

Tests run automatically on:

- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

See `.github/workflows/ci.yml` for CI configuration.
