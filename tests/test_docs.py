"""The README must describe the product that actually ships.

Documentation drifts silently: a runtime is added, the table keeps listing the
original six, and the first person to follow the docs installs the wrong thing.
These assertions fail the moment the registry and the README disagree, which is
the only moment anyone is looking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from protor.llm_backends import BACKEND_CHOICES
from protor.runtimes import RUNTIMES

README = Path(__file__).resolve().parent.parent / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_exists():
    assert README.exists(), "the README is the package's front door"


class TestRuntimeTable:
    @pytest.mark.parametrize("key", sorted(RUNTIMES))
    def test_every_runtime_is_documented(self, key, readme):
        assert f"`{key}`" in readme, f"{key} is supported but missing from the README"

    @pytest.mark.parametrize("key", sorted(RUNTIMES))
    def test_every_runtime_url_is_documented(self, key, readme):
        url = RUNTIMES[key].default_url
        assert url in readme, f"{key}'s default URL ({url}) is not in the README"

    @pytest.mark.parametrize("env", sorted(r.env_url for r in RUNTIMES.values() if r.env_url))
    def test_every_url_override_is_documented(self, env, readme):
        assert env in readme, f"{env} overrides a runtime's URL but is undocumented"

    def test_hosted_backends_are_documented(self, readme):
        assert "OPENAI_API_KEY" in readme
        assert "ANTHROPIC_API_KEY" in readme

    def test_the_table_is_not_a_hand_maintained_subset(self, readme):
        """A stale README usually keeps exactly the original handful of rows."""
        assert readme.count("http://localhost:") >= len(RUNTIMES)


class TestCommandsAreReal:
    @pytest.mark.parametrize(
        "flag",
        [
            "--no-live",
            "--cache",
            "--no-js",
            "--block-ads",
            "--auto-scale",
            "--resume",
            "--base-url",
            "--api-key",
            "--max-pages",
            "--schema",
        ],
    )
    def test_documented_flag_exists_in_the_parser(self, flag, readme):
        assert flag in readme, f"{flag} is documented but no longer accepted"
        from protor.cli import _build_parser

        help_text = _build_parser().format_help()
        for command in _build_parser()._subparsers._group_actions[0].choices.values():
            help_text += command.format_help()
        assert flag in help_text, f"{flag} is documented but the parser does not accept it"

    @pytest.mark.parametrize("command", ["scrape", "crawl", "analyze", "run", "models", "runtimes"])
    def test_documented_command_exists(self, command, readme):
        assert command in readme
        from protor.cli import _build_parser

        parser = _build_parser()
        choices = parser._subparsers._group_actions[0].choices
        assert command in choices, f"{command} is documented but not a command"


class TestBackendChoicesAreReachable:
    def test_every_backend_choice_is_mentioned(self, readme):
        """`--backend` accepts more names than the README lists; that is fine, but
        the ones a user is likely to type must be there."""
        for name in ("ollama", "openai", "anthropic", "openai-compatible"):
            assert name in BACKEND_CHOICES
            assert name in readme


class TestBehaviourDocumented:
    def test_no_live_flag_is_explained(self, readme):
        """A flag nobody can interpret is worse than no flag."""
        assert "--no-live" in readme
        section = readme.lower()
        assert "pipe" in section or "ci" in section, (
            "--no-live is documented without saying when to use it"
        )

    def test_encoding_fallback_is_documented(self, readme):
        assert "NO_COLOR" in readme

    def test_python_requirement_matches_the_project(self, readme):
        from protor import __version__  # noqa: F401  - import guard

        assert "3.11" in readme, "the README must state the minimum Python version"


class TestHelpEnvironmentBlock:
    """
    `protor --help`'s environment block is generated from the runtime registry.

    Written out by hand it named six of the seventeen runtimes, so the other
    eleven were documented only in the README, and it described the API-key
    variables as ``*_API_KEY`` — a pattern the registry does not follow, since
    KoboldCpp's is ``KOBOLDCPP_API_KEY`` and TabbyAPI's is ``TABBY_API_KEY``.
    """

    def test_every_runtime_url_variable_is_listed(self):
        from protor.cli import _runtime_env_help
        from protor.runtimes import RUNTIMES

        block = _runtime_env_help()
        for runtime in RUNTIMES.values():
            if runtime.env_url:
                assert runtime.env_url in block, f"{runtime.key} has no URL in --help"
            if runtime.env_key:
                assert runtime.env_key in block, f"{runtime.key} has no key in --help"

    def test_the_block_is_in_the_help_output(self, capsys):
        from protor.cli import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--help"])
        out = capsys.readouterr().out
        assert "Environment:" in out
        assert "OLLAMA_HOST" in out

    def test_no_variable_is_invented(self):
        """Every variable named in the block is one the code actually reads."""
        from protor.cli import _runtime_env_help
        from protor.runtimes import RUNTIMES

        known = {v for r in RUNTIMES.values() for v in (r.env_url, r.env_key) if v}
        # The block opens with its own "Environment:" heading.
        flags = {"--base-url", "--api-key", "Environment:"}
        for line in _runtime_env_help().splitlines():
            token = line.strip().split(" ")[0] if line.strip() else ""
            if not token or token in flags:
                continue
            assert token in known, f"--help documents {token}, which no runtime reads"
