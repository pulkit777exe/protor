import argparse
import os
import sys
import json
from rich.console import Console
from rich.panel import Panel
from rich import box
from protor.scraper import scrape_multiple
from protor.analyzer import analyze_with_ollama
from protor.utils import get_default_output_dir
from protor.crawler import Crawler

console = Console()

def cli():
    parser = argparse.ArgumentParser(
        prog="protor",
        description="⸸ AI-powered website scraping and analysis CLI using curl and Ollama ⸸"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    scrape_parser = subparsers.add_parser("scrape", help="Scrape websites (HTML + JS)")
    scrape_parser.add_argument("urls", nargs="+", help="One or more URLs to scrape")
    scrape_parser.add_argument("--output", "-o", default=None, help="Output folder (default: Downloads)")
    scrape_parser.add_argument("--no-js", action="store_true", help="Skip JavaScript file downloads")
    scrape_parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze scraped data with local Ollama")
    analyze_parser.add_argument("--file", "-f", default="data/sites_index.json", help="Scraped JSON file")
    analyze_parser.add_argument("--model", "-m", default="llama3", help="Ollama model (e.g., llama3, mistral, codellama)")
    analyze_parser.add_argument("--focus", choices=["general", "technical", "content", "seo"], 
                              default="general", help="Analysis focus area")
    analyze_parser.add_argument("--output", "-o", default="analysis", help="Output folder for analysis")

    run_parser = subparsers.add_parser("run", help="Scrape and analyze in one go")
    run_parser.add_argument("urls", nargs="+", help="One or more URLs to scrape")
    run_parser.add_argument("--model", "-m", default="llama3", help="Ollama model name")
    run_parser.add_argument("--focus", choices=["general", "technical", "content", "seo"],
                          default="general", help="Analysis focus area")
    run_parser.add_argument("--output", "-o", default=None, help="Output folder (default: Downloads)")
    run_parser.add_argument("--no-js", action="store_true", help="Skip JavaScript file downloads")

    crawl_parser = subparsers.add_parser("crawl", help="Recursively crawl a website")
    crawl_parser.add_argument("url", help="Starting URL")
    crawl_parser.add_argument("--max-pages", type=int, default=10, help="Maximum pages to scrape")
    crawl_parser.add_argument("--output", "-o", default=None, help="Output folder (default: Downloads/protor)")
    crawl_parser.add_argument("--analyze", action="store_true", help="Enable concurrent RAG analysis")
    crawl_parser.add_argument("--model", "-m", default="llama3", help="Ollama model for analysis")
    crawl_parser.add_argument("--focus", choices=["general", "technical", "content", "seo"],
                            default="general", help="Analysis focus area")


    list_parser = subparsers.add_parser("models", help="List available Ollama models")

    args = parser.parse_args()

    if args.command == "scrape":
        base_dir = args.output if args.output else get_default_output_dir()
        
        try:
            json_file = scrape_multiple(
                args.urls, 
                base_dir,
                download_js=not args.no_js,
                timeout=args.timeout
            )
        except KeyboardInterrupt:
            console.print("\n\n[bold grey93]⸸ The harvest was interrupted ⸸[/bold grey93]\n")
            sys.exit(0)

    elif args.command == "analyze":
        if not os.path.exists(args.file):
            console.print()
            console.print(Panel(
                f"[bold grey93]⚠ The Tome is Missing ⚠[/bold grey93]\n\n"
                f"[grey74]Path sought:[/grey74] [grey50]{args.file}[/grey50]\n\n"
                f"[grey74]Summon data first with:[/grey74]\n"
                f"[grey50]protor scrape <urls>[/grey50]",
                box=box.DOUBLE_EDGE,
                border_style="grey35",
                style="on grey7"
            ))
            console.print()
            return
        
        console.print()
        console.print(Panel(
            f"[bold grey93]⸸ Commencing Analysis ⸸[/bold grey93]\n"
            f"[grey74]Oracle:[/grey74] [bright_white]{args.model}[/bright_white]\n"
            f"[grey74]Focus:[/grey74] [grey74]{args.focus}[/grey74]\n"
            f"[grey74]Tome:[/grey74] [grey50]{args.file}[/grey50]",
            box=box.DOUBLE_EDGE,
            border_style="grey35",
            style="on grey7"
        ))
        console.print()
        
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Analysis goes into 'analysis' subdirectory
        if args.output == "analysis":
            # Default case - put it in base_dir/analysis
            base_dir = get_default_output_dir()
            analysis_output = os.path.join(base_dir, "analysis")
        else:
            analysis_output = args.output
        
        try:
            analyze_with_ollama(data, args.model, args.focus, analysis_output)
        except KeyboardInterrupt:
            console.print("\n\n[bold grey93]⸸ The divination was interrupted ⸸[/bold grey93]\n")
            sys.exit(0)
        
        console.print()
        console.print(Panel(
            f"[bold grey93]⸸ Analysis Complete ⸸[/bold grey93]\n"
            f"[grey74]Prophecy inscribed in:[/grey74] [grey50]{analysis_output}/[/grey50]",
            box=box.DOUBLE_EDGE,
            border_style="grey35",
            style="on grey7"
        ))
        console.print()

    elif args.command == "run":
        base_dir = args.output if args.output else get_default_output_dir()
        
        console.print()
        console.print(Panel(
            f"[bold grey93]⸸ The Complete Ritual ⸸[/bold grey93]\n"
            f"[grey74]Phase I:[/grey74] [grey50]Harvest[/grey50]\n"
            f"[grey74]Phase II:[/grey74] [grey50]Divine[/grey50]\n"
            f"[grey74]Targets:[/grey74] [bright_white]{len(args.urls)}[/bright_white]\n"
            f"[grey74]Oracle:[/grey74] [bright_white]{args.model}[/bright_white]",
            box=box.DOUBLE_EDGE,
            border_style="grey35",
            style="on grey7"
        ))
        console.print()
        
        try:
            json_file = scrape_multiple(
                args.urls,
                base_dir,
                download_js=not args.no_js
            )
        except KeyboardInterrupt:
            console.print("\n\n[bold grey93]⸸ The harvest was interrupted ⸸[/bold grey93]\n")
            sys.exit(0)
        
        console.print()
        console.print(Panel(
            f"[bold grey93]⸸ Phase II: Divination ⸸[/bold grey93]\n"
            f"[grey74]Consulting the oracle...[/grey74]",
            box=box.DOUBLE_EDGE,
            border_style="grey35",
            style="on grey7"
        ))
        console.print()
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        analysis_dir = os.path.join(base_dir, "analysis")
        
        try:
            analyze_with_ollama(data, args.model, args.focus, analysis_dir)
        except KeyboardInterrupt:
            console.print("\n\n[bold grey93]⸸ The divination was interrupted ⸸[/bold grey93]\n")
            sys.exit(0)
        
        console.print()
        console.print(Panel(
            f"[bold grey93]⸸ The Ritual is Complete ⸸[/bold grey93]\n"
            f"[grey74]All secrets revealed in:[/grey74]\n"
            f"[grey50]{analysis_dir}/[/grey50]",
            box=box.DOUBLE_EDGE,
            border_style="grey35",
            style="on grey7"
        ))
        console.print()

    elif args.command == "models":
        console.print()
        console.print(Panel(
            f"[bold grey93]⸸ Available Oracles ⸸[/bold grey93]\n"
            f"[grey74]Consulting the spirits...[/grey74]",
            box=box.DOUBLE_EDGE,
            border_style="grey35",
            style="on grey7"
        ))
        console.print()
        
        from protor.analyzer import list_ollama_models
        list_ollama_models()

    elif args.command == "crawl":
        base_dir = args.output if args.output else get_default_output_dir()
        crawler_dir = os.path.join(base_dir, "crawler")
        crawler = Crawler(
            args.url, 
            args.max_pages, 
            crawler_dir,
            analyze=args.analyze,
            model=args.model,
            focus=args.focus
        )
        try:
            crawler.crawl()
        except KeyboardInterrupt:
            console.print("\n\n[bold grey93]⸸ The crawl was interrupted ⸸[/bold grey93]\n")
            sys.exit(0)

    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[bold grey93]⸸ The ritual was aborted ⸸[/bold grey93]\n")
        sys.exit(0)