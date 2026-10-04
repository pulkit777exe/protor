"""Allow running protor as a module: python -m protor."""

from .cli import cli

# Under a `__name__ == "__main__"` guard rather than at module scope. Running the
# package as a module sets that name, so `python -m protor` behaves exactly as
# before — but importing `protor.__main__` no longer parses argv and exits. It
# did: `pkgutil.iter_modules` lists `__main__` among the package's submodules, so
# any tooling that walks the package — an import-everything helper, a coverage
# sweep, a docs generator — ran the CLI on import, and with whatever argv it
# happened to be holding. It surfaced as `protor: error: argument <command>:
# invalid choice: 'tests/test_isolation.py'`.
if __name__ == "__main__":
    cli()
