"""
protor.cli
~~~~~~~~~~
Command-line interface entry point.

Commands
--------
    protor scrape   <urls>...
    protor analyze
    protor run      <urls>...
    protor crawl    <url>
    protor extract  <url> <schema.json>
    protor models
    protor version
    protor update
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from .analyzer import (
    FOCUS_CHOICES,
    analyze_with_runtime,
    list_runtime_models,
    list_runtimes,
)
from .crawler import Crawler
from .exceptions import (
    ConfigurationError,
    DataFileNotFoundError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
    OutputPathError,
    ProtorError,
    URLValidationError,
)
from .extractor import ExtractionSchema
from .formatters import FORMAT_CHOICES
from .llm_backends import BACKEND_CHOICES
from .runtimes import get_runtime, runtime_names
from .scraper import scrape_multiple
from .theme import ERR, console, err, info, safe
from .updater import check_for_update, perform_update
from .utils import get_default_output_dir, load_json, validate_url

if TYPE_CHECKING:
    from .models import SiteManifest

# ── helpers ───────────────────────────────────────────────────────────────────


def _abort(msg: str, hint: str = "") -> NoReturn:
    console.print(f"\n  {err(msg)}")
    if hint:
        console.print(f"  {info(hint)}")
    console.print()
    sys.exit(1)


def _load_index(path: str) -> list[dict[str, Any] | SiteManifest]:
    """
    Load a sites index written by `scrape`.

    Reports the unusable case as a sentence naming the path. Pointing ``--index``
    at a directory used to escape as a bare ``IsADirectoryError`` traceback from
    ``read_text``, which said nothing about which of the paths on the command
    line was the wrong one.
    """
    p = Path(path)
    if p.is_dir():
        raise DataFileNotFoundError(path, "it is a directory")
    try:
        result: list[dict[str, Any] | SiteManifest] = list(load_json(path))
    except FileNotFoundError as exc:
        raise DataFileNotFoundError(path) from exc
    except json.JSONDecodeError as exc:
        raise DataFileNotFoundError(
            path, f"it is not valid JSON ({exc.msg} at line {exc.lineno})"
        ) from exc
    except OSError as exc:
        raise DataFileNotFoundError(path, exc.strerror or str(exc)) from exc
    return result


def _load_schema(path: str | None) -> ExtractionSchema | None:
    """Load an extraction schema from *path*, aborting the CLI on parse errors."""
    if not path:
        return None
    try:
        return ExtractionSchema.from_json(path)
    except Exception as e:
        _abort(f"Failed to load schema: {e}", hint="Schema must be a valid JSON file")


# ── command handlers ──────────────────────────────────────────────────────────


def _run_scrape(args: argparse.Namespace) -> str:
    """Shared scrape step for the ``scrape`` and ``run`` subcommands."""
    for url in args.urls:
        validate_url(url)
    base = Path(args.output) if args.output else get_default_output_dir()

    return scrape_multiple(
        args.urls,
        base,
        download_js=not args.no_js,
        timeout=args.timeout,
        concurrency=args.concurrency,
        extraction_schema=_load_schema(args.schema),
        block_ads=args.block_ads,
        auto_scale=args.auto_scale,
        use_cache=args.cache,
        live=not args.no_live,
    )


def _cmd_scrape(args: argparse.Namespace) -> None:
    _run_scrape(args)


def _resolve_prompt(args: argparse.Namespace) -> str | None:
    """Return the custom prompt from --prompt / --prompt-file, if any."""
    prompt: str | None = getattr(args, "prompt", None)
    if prompt:
        return prompt
    prompt_file: str | None = getattr(args, "prompt_file", None)
    if not prompt_file:
        return None
    path = Path(prompt_file)
    if not path.exists():
        _abort(f"Prompt file not found: {prompt_file}", hint="Check the path to your prompt file")
    return path.read_text(encoding="utf-8")


def _cmd_analyze(args: argparse.Namespace) -> None:
    data = _load_index(args.file)
    out = get_default_output_dir() / "analysis" if args.output == "analysis" else Path(args.output)
    _analyze(args, data, out)


def _cmd_run(args: argparse.Namespace) -> None:
    index = _run_scrape(args)
    data = _load_index(index)
    base = Path(args.output) if args.output else get_default_output_dir()
    _analyze(args, data, base / "analysis")


def _analyze(
    args: argparse.Namespace, data: list[dict[str, Any] | SiteManifest], out: Path
) -> None:
    """Shared analysis step for the `analyze` and `run` subcommands."""
    analyze_with_runtime(
        data,
        args.backend,
        args.model,
        args.focus,
        out,
        base_url=args.base_url,
        api_key=args.api_key,
        prompt=_resolve_prompt(args),
        fmt=args.format,
    )


def _cmd_crawl(args: argparse.Namespace) -> None:
    validate_url(args.url)
    base = Path(args.output) if args.output else get_default_output_dir()
    Crawler(
        args.url,
        args.max_pages,
        base / "crawler",
        resume=args.resume,
        auto_scale=args.auto_scale,
        live=not args.no_live,
    ).crawl()


def _cmd_extract(args: argparse.Namespace) -> None:
    """Extract structured data from a URL or file using a schema."""
    validate_url(args.url)

    schema_path = Path(args.schema)
    if not schema_path.exists():
        _abort(f"Schema file not found: {args.schema}", hint="Create a JSON schema file first")

    try:
        schema = ExtractionSchema.from_json(schema_path)
    except Exception as e:
        _abort(f"Invalid schema: {e}")

    import aiohttp

    from .extractor import Extractor
    from .fetcher import fetch
    from .theme import OK, bright, header_rule, label, muted, warn

    async def _extract_async() -> list[dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            result = await fetch(session, args.url, timeout=args.timeout)
            extractor = Extractor(schema, base_url=args.url)
            return extractor.extract(result.text)

    console.print()
    console.print(header_rule("Protor — Extract"))
    console.print(
        f"  {label('url')} {bright(args.url)}\n"
        f"  {label('schema')} {bright(schema.name)} ({len(schema.fields)} fields)"
    )
    console.print()

    results = asyncio.run(_extract_async())

    # A record whose every field is empty is not data. Counting containers is
    # not enough: a schema whose base_selector matches but whose field
    # selectors match nothing produced a full set of records with every value
    # null, written to disk and reported as a successful extraction. That is the
    # same failure-as-success shape the selector *syntax* check exists to
    # prevent, one level up, and it is what a stale CSS selector looks like
    # after the site redesigns its markup.
    empty = sum(1 for r in results if not any(v not in (None, "") for v in r.values()))

    if not results or empty == len(results):
        console.print(f"  {ERR} No data matched the schema")
        console.print()
        return

    if empty:
        console.print(
            f"  {warn('warn')} {empty} of {len(results)} records matched the container "
            f"but no field selector matched inside it; those records are empty."
        )
        console.print()

    console.print(f"  {OK} Extracted {len(results)} records")
    console.print()

    # Output
    out_dir = Path(args.output) if args.output else get_default_output_dir() / "extractions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{schema.name}_{Path(args.url).stem}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"  {label('saved')} {muted(str(out_file))}")
    console.print()

    # Print preview
    for i, record in enumerate(results[:3], 1):
        console.print(f"  [{i}]")
        for k, v in record.items():
            display = str(v)[:80] + ("..." if len(str(v)) > 80 else "")
            console.print(f"      {k}: {display}")
        console.print()
    if len(results) > 3:
        console.print(f"  ... and {len(results) - 3} more")


def _cmd_models(args: argparse.Namespace) -> None:
    list_runtime_models(args.backend, base_url=args.base_url, api_key=args.api_key)


def _cmd_runtimes(_args: argparse.Namespace) -> None:
    list_runtimes()


def _cmd_version(_args: argparse.Namespace) -> None:
    from protor import __version__

    console.print(f"protor {__version__}")


def _cmd_update(args: argparse.Namespace) -> None:
    from .updater import _is_editable_install

    if _is_editable_install():
        console.print(f"\n  {err('Editable install detected.')}")
        console.print(f"  {info('Update via: git pull && pip install -e .')}\n")
        return

    result = check_for_update()

    if result is None:
        console.print(f"\n  {err('Failed to check for updates.')}")
        console.print(f"  {info('Check your internet connection and try again.')}\n")
        return

    current = result["current"]
    latest = result["latest"]
    update_available = result["update_available"]

    if args.check or not update_available:
        if update_available:
            console.print(f"\n  Current: {err(current)}  |  Latest: {info(latest)}")
            console.print(f"  {info('Update available! Run: protor update')}\n")
        else:
            console.print(f"\n  protor is already up to date (v{current})\n")
        return

    console.print(f"\n  Current: {err(current)}  |  Latest: {info(latest)}")

    if not args.yes:
        try:
            confirm = input("\n  Update protor? [y/N]: ").strip().lower()
            if confirm not in ("y", "yes"):
                console.print(f"  {info('Update cancelled.')}\n")
                return
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n  {info('Update cancelled.')}\n")
            return

    console.print(f"\n  Updating protor to v{latest}...")

    if perform_update():
        console.print(f"  protor updated to v{latest}\n")
    else:
        console.print(f"\n  {err('Update failed.')}")
        console.print(f"  {info('Try: pip install --upgrade protor')}\n")


# ── parser ────────────────────────────────────────────────────────────────────


def _normalize_backend(value: str) -> str:
    """
    Canonicalise a ``--backend`` value before argparse validates it.

    Friendly spellings like ``llama.cpp`` and ``lm-studio`` are accepted by the
    backend factory, but argparse's ``choices`` only knows the canonical keys —
    so resolve aliases first, otherwise the CLI would reject names the API
    happily accepts.
    """
    try:
        return get_runtime(value).key
    except ValueError:
        return value.strip().lower()


class _Parser(argparse.ArgumentParser):
    """
    Argument parser whose output survives a terminal that cannot encode it.

    argparse writes usage, help and error text straight to the file, bypassing
    both the shared console and its glyph handling, so ``protor --help`` died
    with a UnicodeEncodeError on an ASCII terminal — the first command anyone
    runs. The hook belongs on the parser (which owns ``_print_message``), not on
    the help formatter, which argparse never routes output through.
    """

    def _print_message(self, message: str, file: Any = None) -> None:
        super()._print_message(safe(message), file)


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    """Add the progress-display flag shared by the long-running commands."""
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="disable in-place progress rendering (plain output for pipes and CI)",
    )


def _add_analysis_flags(parser: argparse.ArgumentParser) -> None:
    """Add the options shared by `analyze` and `run`."""
    parser.add_argument(
        "--backend",
        "-b",
        choices=BACKEND_CHOICES,
        default="ollama",
        type=_normalize_backend,
        metavar="RUNTIME",
        help=(
            "model runtime: "
            f"{', '.join(runtime_names())}, openai-compatible, openai, anthropic "
            "(default: ollama). Aliases like llama.cpp and lm-studio also work."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        metavar="URL",
        help="override the runtime's URL (e.g. http://localhost:8080)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="TOKEN",
        help="token for runtimes started with authentication enabled",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=None,
        metavar="TEXT",
        help="custom analysis prompt (overrides default)",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        metavar="PATH",
        help="read custom prompt from file",
    )
    parser.add_argument(
        "--format",
        choices=FORMAT_CHOICES,
        default="markdown",
        help="output format (default: markdown)",
    )


def _build_parser() -> argparse.ArgumentParser:
    root = _Parser(
        prog="protor",
        description="AI-powered web scraper and analyzer — works with any local LLM runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  protor scrape https://example.com https://news.ycombinator.com\n"
            "  protor scrape https://example.com --block-ads --auto-scale\n"
            "  protor analyze --model mistral --focus technical\n"
            "  protor run https://example.com --model llama3\n"
            "  protor crawl https://example.com --max-pages 20 --resume\n"
            "  protor extract https://example.com products.json\n"
            "  protor runtimes\n"
            "  protor models --backend lmstudio\n"
            "  protor analyze --backend vllm --model Qwen/Qwen3-8B\n"
            "\n"
            "Environment:\n"
            "  OLLAMA_HOST    Ollama base URL (default: http://localhost:11434)\n"
            "  LLAMA_CPP_URL  llama-server URL (default: http://localhost:8080)\n"
            "  LMSTUDIO_URL   LM Studio URL (default: http://localhost:1234)\n"
            "  VLLM_URL       vLLM URL (default: http://localhost:8000)\n"
            "  LOCALAI_URL    LocalAI URL (default: http://localhost:8081)\n"
            "  JAN_URL        Jan URL (default: http://localhost:1337)\n"
            "  *_API_KEY      token for runtimes started with authentication\n"
        ),
    )
    sub = root.add_subparsers(dest="command", metavar="<command>", parser_class=_Parser)
    root.set_defaults(func=lambda _: root.print_help())

    # ── scrape ──────────────────────────────────────────────────────────────
    sp = sub.add_parser("scrape", help="scrape one or more URLs")
    sp.add_argument("urls", nargs="+", metavar="URL")
    sp.add_argument(
        "--output",
        "-o",
        metavar="DIR",
        default=None,
        help="output directory (default: ~/Downloads/protor)",
    )
    sp.add_argument("--no-js", action="store_true", help="skip JavaScript file downloads")
    sp.add_argument(
        "--timeout",
        type=int,
        default=30,
        metavar="SEC",
        help="per-request timeout in seconds (default: 30)",
    )
    sp.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=6,
        metavar="N",
        help="parallel requests (default: 6)",
    )
    sp.add_argument(
        "--schema",
        "-s",
        metavar="SCHEMA.json",
        help="JSON schema file for structured data extraction",
    )
    sp.add_argument(
        "--block-ads",
        action="store_true",
        help="block requests to known ad/tracker domains",
    )
    sp.add_argument(
        "--auto-scale",
        action="store_true",
        help="automatically adjust concurrency based on success rates",
    )
    sp.add_argument(
        "--cache",
        action="store_true",
        help="reuse cached responses (ETag/Last-Modified) across runs",
    )
    _add_output_flags(sp)
    sp.set_defaults(func=_cmd_scrape)

    # ── analyze ─────────────────────────────────────────────────────────────
    ap = sub.add_parser("analyze", help="analyze scraped data with a local or hosted LLM")
    ap.add_argument(
        "--file",
        "-f",
        default=str(get_default_output_dir() / "sites_index.json"),
        metavar="PATH",
        help=(f"scraped index JSON (default: {get_default_output_dir() / 'sites_index.json'})"),
    )
    ap.add_argument(
        "--model",
        "-m",
        default="llama3",
        metavar="MODEL",
        help="Ollama model name (default: llama3)",
    )
    ap.add_argument(
        "--focus",
        choices=FOCUS_CHOICES,
        default="general",
        help="analysis focus (default: general)",
    )
    ap.add_argument(
        "--output",
        "-o",
        default="analysis",
        metavar="DIR",
        help="output directory (default: ~/Downloads/protor/analysis)",
    )
    _add_analysis_flags(ap)
    ap.set_defaults(func=_cmd_analyze)

    # ── run (scrape + analyze) ───────────────────────────────────────────────
    rp = sub.add_parser("run", help="scrape then analyze in one step")
    rp.add_argument("urls", nargs="+", metavar="URL")
    rp.add_argument("--model", "-m", default="llama3", metavar="MODEL")
    rp.add_argument("--focus", choices=FOCUS_CHOICES, default="general")
    rp.add_argument("--output", "-o", metavar="DIR", default=None)
    rp.add_argument("--no-js", action="store_true")
    rp.add_argument("--concurrency", "-c", type=int, default=6, metavar="N")
    rp.add_argument(
        "--timeout",
        type=int,
        default=30,
        metavar="SEC",
        help="per-request timeout in seconds (default: 30)",
    )
    rp.add_argument(
        "--schema",
        "-s",
        metavar="SCHEMA.json",
        help="JSON schema file for structured data extraction",
    )
    rp.add_argument("--block-ads", action="store_true")
    rp.add_argument("--auto-scale", action="store_true")
    rp.add_argument("--cache", action="store_true", help="reuse cached responses across runs")
    _add_analysis_flags(rp)
    _add_output_flags(rp)
    rp.set_defaults(func=_cmd_run)

    # ── crawl ────────────────────────────────────────────────────────────────
    cp = sub.add_parser("crawl", help="recursively crawl a site")
    cp.add_argument("url", metavar="URL")
    cp.add_argument(
        "--max-pages",
        type=int,
        default=10,
        metavar="N",
        help="page limit (default: 10)",
    )
    cp.add_argument("--output", "-o", metavar="DIR", default=None)
    cp.add_argument(
        "--resume",
        action="store_true",
        help="resume from a previous checkpoint if available",
    )
    cp.add_argument(
        "--auto-scale",
        action="store_true",
        help="automatically adjust concurrency based on success rates",
    )
    _add_output_flags(cp)
    cp.set_defaults(func=_cmd_crawl)

    # ── extract ──────────────────────────────────────────────────────────────
    ep = sub.add_parser("extract", help="extract structured data using a schema")
    ep.add_argument("url", metavar="URL")
    ep.add_argument("schema", metavar="SCHEMA.json", help="JSON schema file")
    ep.add_argument("--output", "-o", metavar="DIR", default=None)
    ep.add_argument("--timeout", type=int, default=30, metavar="SEC")
    ep.set_defaults(func=_cmd_extract)

    # ── models ───────────────────────────────────────────────────────────────
    mp = sub.add_parser("models", help="list models available from a runtime")
    mp.add_argument(
        "--backend",
        "-b",
        choices=BACKEND_CHOICES,
        default="ollama",
        type=_normalize_backend,
        metavar="RUNTIME",
        help=f"runtime to query (default: ollama). Try: {', '.join(runtime_names())}",
    )
    mp.add_argument("--base-url", default=None, metavar="URL", help="override the runtime's URL")
    mp.add_argument("--api-key", default=None, metavar="TOKEN", help="runtime API token")
    mp.set_defaults(func=_cmd_models)

    # ── runtimes ──────────────────────────────────────────────────────────────
    rt = sub.add_parser("runtimes", help="show which local model runtimes are running")
    rt.set_defaults(func=_cmd_runtimes)

    # ── version ──────────────────────────────────────────────────────────────
    vp = sub.add_parser("version", help="print version and exit")
    vp.set_defaults(func=_cmd_version)

    # ── update ───────────────────────────────────────────────────────────────
    up = sub.add_parser("update", help="check for updates and upgrade protor")
    up.add_argument("--check", action="store_true", help="only check for updates, don't install")
    up.add_argument("--yes", "-y", action="store_true", help="skip confirmation prompt")
    up.set_defaults(func=_cmd_update)

    return root


# ── entry point ───────────────────────────────────────────────────────────────


def cli() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print(f"\n  {ERR} interrupted\n")
        sys.exit(130)
    except OllamaUnavailableError as exc:
        _abort(str(exc), hint="Start with: ollama serve")
    except OllamaModelNotFoundError as exc:
        # The message already names the exact pull command.
        _abort(str(exc))
    except DataFileNotFoundError as exc:
        _abort(str(exc), hint="Run: protor scrape <urls>")
    except OutputPathError as exc:
        _abort(str(exc), hint="--output/-o takes a directory, not a file")
    except ConfigurationError as exc:
        _abort(str(exc), hint="Check the environment variables listed in protor --help")
    except URLValidationError as exc:
        _abort(str(exc), hint="URLs must include a scheme, e.g. https://example.com")
    except ProtorError as exc:
        _abort(str(exc))
    except ValueError as exc:
        # A stray ValueError from library code: report it without guessing.
        _abort(str(exc))
