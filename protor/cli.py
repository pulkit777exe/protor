import argparse
import os
import json
from protor.scraper import scrape_multiple
from protor.analyzer import analyze_with_ollama
from protor.utils import get_default_output_dir
from protor.crawler import Crawler



def cli():
    parser = argparse.ArgumentParser(
        prog="protor",
        description="AI-powered website scraping and analysis CLI using curl and Ollama"
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
    crawl_parser.add_argument("--output", "-o", default=None, help="Output folder (default: Downloads)")


    list_parser = subparsers.add_parser("models", help="List available Ollama models")

    args = parser.parse_args()

    if args.command == "scrape":
        output_dir = args.output if args.output else get_default_output_dir()
        print(f"Scraping {len(args.urls)} URL(s)...")
        json_file = scrape_multiple(
            args.urls, 
            output_dir,
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
        output_dir = args.output if args.output else get_default_output_dir()
        print(f"Starting scrape and analysis pipeline...")
        print(f"Scraping {len(args.urls)} URL(s)...")
        
        json_file = scrape_multiple(
            args.urls,
            output_dir,
            download_js=not args.no_js
        )
        
        print(f"Scraping complete!")
        print(f"Analyzing with {args.model}...")
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        analysis_dir = os.path.join(output_dir, "analysis")
        analyze_with_ollama(data, args.model, args.focus, analysis_dir)
        print(f"All done! Check the {analysis_dir}/ folder for results")

    elif args.command == "models":
        from protor.analyzer import list_ollama_models
        list_ollama_models()

    elif args.command == "crawl":
        crawler = Crawler(args.url, args.max_pages, args.output)
        crawler.crawl()


    else:
        parser.print_help()

if __name__ == "__main__":
    cli()
