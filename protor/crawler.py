import time
from collections import deque
from urllib.parse import urlparse
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich import box

from protor.scraper import scrape_website, extract_links, fetch_with_curl
from protor.utils import get_default_output_dir

class Crawler:
    def __init__(self, start_url: str, max_pages: int = 10, output_dir: str = None):
        self.start_url = start_url
        self.max_pages = max_pages
        self.output_dir = output_dir or get_default_output_dir()
        self.visited = set()
        self.queue = deque([start_url])
        self.scraped_count = 0
        self.current_url = ""
        self.console = Console()

    def generate_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        layout["body"].split_row(
            Layout(name="current", ratio=2),
            Layout(name="queue", ratio=1)
        )
        return layout

    def get_queue_table(self) -> Panel:
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Queue (Next 10)", style="cyan")
        
        for url in list(self.queue)[:10]:
            table.add_row(url)
            
        if len(self.queue) > 10:
            table.add_row(f"... and {len(self.queue) - 10} more")
            
        return Panel(table, title=f"Queue ({len(self.queue)})", border_style="blue")

    def get_status_panel(self) -> Panel:
        content = f"""
[bold green]Scraping:[/bold green] {self.current_url}
[bold]Progress:[/bold] {self.scraped_count}/{self.max_pages}
[bold]Visited:[/bold] {len(self.visited)}
[bold]Output:[/bold] {self.output_dir}
"""
        return Panel(content, title="Current Job", border_style="green")

    def crawl(self):
        layout = self.generate_layout()
        layout["header"].update(Panel("🕷️  Protor Crawler", style="bold white on blue"))
        layout["footer"].update(Panel("Press Ctrl+C to stop", style="italic grey50"))

        with Live(layout, refresh_per_second=4, console=self.console) as live:
            while self.queue and self.scraped_count < self.max_pages:
                self.current_url = self.queue.popleft()
                
                if self.current_url in self.visited:
                    continue
                
                # Update UI
                layout["current"].update(self.get_status_panel())
                layout["queue"].update(self.get_queue_table())
                
                self.visited.add(self.current_url)
                
                # Scrape
                # We silence stdout from scrape_website to not mess up the TUI
                # In a real scenario we might want to redirect stdout, but for now 
                # we rely on scrape_website not printing too much or just accepting it might flicker slightly
                # Actually, scrape_website prints. Let's just run it. 
                # Ideally we would modify scrape_website to be silent, but for this MVP 
                # we will just let it run. The TUI might be a bit messy if scrape_website prints.
                # Let's try to fetch first to get links, then scrape.
                
                # Fetch content to get links
                html, success = fetch_with_curl(self.current_url)
                
                if success:
                    # Extract links
                    new_links = extract_links(html, self.current_url)
                    for link in new_links:
                        if link not in self.visited and link not in self.queue:
                            self.queue.append(link)
                    
                    # Actually save the data
                    scrape_website(self.current_url, self.output_dir, download_js=False)
                    self.scraped_count += 1
                
                layout["current"].update(self.get_status_panel())
                layout["queue"].update(self.get_queue_table())
                
                time.sleep(0.5) # Be nice

        self.console.print(f"[bold green]Crawl complete![/bold green] Scraped {self.scraped_count} pages.")
