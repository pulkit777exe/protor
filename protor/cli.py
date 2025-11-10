import argparse
import os
import json
from protor.scraper import scrape_multiple
from protor.analyzer import analyze_with_ollama

def cli():
    parser = argparse.ArgumentParser(
        prog="protor",
        description="AI-powered website scraping and analysis CLI using curl and Ollama"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    scrape_parser = subparsers.add_parser("scrape", help="Scrape websites (HTML + JS)")
    scrape_parser.add_argument("urls", nargs="+", help="One or more URLs to scrape")
    scrape_parser.add_argument("--output", "-o", default="data", help="Output folder (default: data)")
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
    run_parser.add_argument("--output", "-o", default="data", help="Output folder")
    run_parser.add_argument("--no-js", action="store_true", help="Skip JavaScript file downloads")

    list_parser = subparsers.add_parser("models", help="List available Ollama models")

    args = parser.parse_args()

    if args.command == "scrape":
        print(f"Scraping {len(args.urls)} URL(s)...")
        json_file = scrape_multiple(
            args.urls, 
            args.output,
            download_js=not args.no_js,
            timeout=args.timeout
        )
        print(f"Scraping complete! Data saved to: {json_file}")

    elif args.command == "analyze":
        if not os.path.exists(args.file):
            print(f"Scraped data file not found: {args.file}")
            print("Run 'protor scrape <urls>' first to gather data")
            return
        
        print(f"Analyzing with {args.model}...")
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        analyze_with_ollama(data, args.model, args.focus, args.output)
        print(f"Analysis complete! Report saved to: {args.output}/")

    elif args.command == "run":
        print(f"Starting scrape and analysis pipeline...")
        print(f"Scraping {len(args.urls)} URL(s)...")
        
        json_file = scrape_multiple(
            args.urls,
            args.output,
            download_js=not args.no_js
        )
        
        print(f"Scraping complete!")
        print(f"Analyzing with {args.model}...")
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        analyze_with_ollama(data, args.model, args.focus, "analysis")
        print(f"All done! Check the analysis/ folder for results")

    elif args.command == "models":
        from protor.analyzer import list_ollama_models
        list_ollama_models()

    else:
        parser.print_help()

if __name__ == "__main__":
    cli()
