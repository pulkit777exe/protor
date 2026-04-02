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
    protor models
    protor version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer import FOCUS_CHOICES, analyze_with_ollama, list_ollama_models
from .crawler import Crawler
from .exceptions import (
    DataFileNotFoundError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
    ProtorError,
)
from .scraper import scrape_multiple
from .theme import ERR, console, err, info
from .utils import get_default_output_dir, load_json


# ── helpers ───────────────────────────────────────────────────────────────────

def _abort(msg: str, hint: str = "") -> None:
    console.print(f"\n  {err(msg)}")
    if hint:
        console.print(f"  {info(hint)}")
    console.print()
    sys.exit(1)


def _load_index(path: str) -> list[dict]:
    try:
        return load_json(path)
    except FileNotFoundError:
        raise DataFileNotFoundError(path)


# ── command handlers ──────────────────────────────────────────────────────────

def _cmd_scrape(args: argparse.Namespace) -> None:
    base = Path(args.output) if args.output else get_default_output_dir()
    scrape_multiple(
        args.urls,
        base,
        download_js=not args.no_js,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )


def _cmd_analyze(args: argparse.Namespace) -> None:
    data = _load_index(args.file)
    out  = (
        get_default_output_dir() / "analysis"
        if args.output == "analysis"
        else Path(args.output)
    )
    analyze_with_ollama(data, args.model, args.focus, out)


def _cmd_run(args: argparse.Namespace) -> None:
    base = Path(args.output) if args.output else get_default_output_dir()
    index = scrape_multiple(
        args.urls,
        base,
        download_js=not args.no_js,
        concurrency=args.concurrency,
    )
    data = _load_index(index)
    analyze_with_ollama(data, args.model, args.focus, base / "analysis")


def _cmd_crawl(args: argparse.Namespace) -> None:
    base = Path(args.output) if args.output else get_default_output_dir()
    Crawler(args.url, args.max_pages, base / "crawler").crawl()


def _cmd_models(_args: argparse.Namespace) -> None:
    list_ollama_models()


def _cmd_version(_args: argparse.Namespace) -> None:
    from protor import __version__
    console.print(f"protor {__version__}")


# ── parser ────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="protor",
        description="AI-powered web scraper and analyzer — powered by Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  protor scrape https://example.com https://news.ycombinator.com\n"
            "  protor analyze --model mistral --focus technical\n"
            "  protor run https://example.com --model llama3\n"
            "  protor crawl https://example.com --max-pages 20\n"
            "  protor models\n"
            "\n"
            "Environment:\n"
            "  OLLAMA_HOST   Ollama base URL (default: http://localhost:11434)\n"
        ),
    )
    sub = root.add_subparsers(dest="command", metavar="<command>")
    root.set_defaults(func=lambda _: root.print_help())

    # ── scrape ──────────────────────────────────────────────────────────────
    sp = sub.add_parser("scrape", help="scrape one or more URLs")
    sp.add_argument("urls", nargs="+", metavar="URL")
    sp.add_argument("--output", "-o",     metavar="DIR",  default=None,
                    help="output directory (default: ~/Downloads/protor)")
    sp.add_argument("--no-js",            action="store_true",
                    help="skip JavaScript file downloads")
    sp.add_argument("--timeout",          type=int, default=30, metavar="SEC",
                    help="per-request timeout in seconds (default: 30)")
    sp.add_argument("--concurrency", "-c", type=int, default=6, metavar="N",
                    help="parallel requests (default: 6)")
    sp.set_defaults(func=_cmd_scrape)

    # ── analyze ─────────────────────────────────────────────────────────────
    ap = sub.add_parser("analyze", help="analyze scraped data with Ollama")
    ap.add_argument("--file", "-f",  default="data/sites_index.json", metavar="PATH",
                    help="scraped index JSON (default: data/sites_index.json)")
    ap.add_argument("--model", "-m", default="llama3", metavar="MODEL",
                    help="Ollama model name (default: llama3)")
    ap.add_argument("--focus",       choices=FOCUS_CHOICES, default="general",
                    help="analysis focus (default: general)")
    ap.add_argument("--output", "-o", default="analysis", metavar="DIR",
                    help="output directory (default: ~/Downloads/protor/analysis)")
    ap.set_defaults(func=_cmd_analyze)

    # ── run (scrape + analyze) ───────────────────────────────────────────────
    rp = sub.add_parser("run", help="scrape then analyze in one step")
    rp.add_argument("urls", nargs="+", metavar="URL")
    rp.add_argument("--model", "-m",      default="llama3",  metavar="MODEL")
    rp.add_argument("--focus",            choices=FOCUS_CHOICES, default="general")
    rp.add_argument("--output", "-o",     metavar="DIR", default=None)
    rp.add_argument("--no-js",            action="store_true")
    rp.add_argument("--concurrency", "-c", type=int, default=6, metavar="N")
    rp.set_defaults(func=_cmd_run)

    # ── crawl ────────────────────────────────────────────────────────────────
    cp = sub.add_parser("crawl", help="recursively crawl a site")
    cp.add_argument("url", metavar="URL")
    cp.add_argument("--max-pages", type=int, default=10, metavar="N",
                    help="page limit (default: 10)")
    cp.add_argument("--output", "-o", metavar="DIR", default=None)
    cp.set_defaults(func=_cmd_crawl)

    # ── models ───────────────────────────────────────────────────────────────
    mp = sub.add_parser("models", help="list available Ollama models")
    mp.set_defaults(func=_cmd_models)

    # ── version ──────────────────────────────────────────────────────────────
    vp = sub.add_parser("version", help="print version and exit")
    vp.set_defaults(func=_cmd_version)

    return root


# ── entry point ───────────────────────────────────────────────────────────────

def cli() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print(f"\n  {ERR} interrupted\n")
        sys.exit(130)
    except OllamaUnavailableError as exc:
        _abort(str(exc), hint="Start with: ollama serve")
    except OllamaModelNotFoundError as exc:
        _abort(str(exc), hint=f"Pull with: ollama pull {exc.model}")
    except DataFileNotFoundError as exc:
        _abort(str(exc), hint="Run: protor scrape <urls>")
    except ProtorError as exc:
        _abort(str(exc))