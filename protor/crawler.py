import time
from collections import deque
from urllib.parse import urlparse
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich import box
from protor.scraper import scrape_website, extract_links, fetch_with_curl
from protor.utils import get_default_output_dir

class Crawler:
    def __init__(self, start_url: str, max_pages: int = 100, output_dir: str = None):
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
            Layout(name="progress", size=5),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        layout["body"].split_row(
            Layout(name="current", ratio=2),
            Layout(name="queue", ratio=1)
        )
        return layout

    def get_queue_table(self) -> Panel:
        table = Table(
            show_header=True, 
            header_style="bold bright_white on grey11",
            box=box.DOUBLE_EDGE,
            border_style="grey35"
        )
        table.add_column("⚰ The Queue ⚰", style="grey74", no_wrap=False)
        
        for url in list(self.queue)[:10]:
            table.add_row(f"☩ {url}")
        
        if len(self.queue) > 10:
            table.add_row(f"[dim]...{len(self.queue) - 10} souls await...[/dim]")
        
        return Panel(
            table, 
            title=f"[bold grey93]⟪ Pending Souls: {len(self.queue)} ⟫[/bold grey93]",
            border_style="grey35",
            box=box.DOUBLE_EDGE
        )

    def get_progress_bar(self) -> Panel:
        """Generate a gothic progress bar panel"""
        progress = Progress(
            TextColumn("[bold grey93]{task.description}"),
            BarColumn(
                complete_style="grey74 on grey11", 
                finished_style="bright_white on grey23",
                bar_width=None
            ),
            TextColumn("[grey93]{task.percentage:>3.0f}%"),
            TextColumn("[grey50]⚔[/grey50]"),
            TextColumn("[grey74]{task.completed}/{task.total} souls harvested[/grey74]"),
            expand=True
        )
        task = progress.add_task(
            "⸸ Reaping Progress ⸸", 
            total=self.max_pages, 
            completed=self.scraped_count
        )
        
        return Panel(
            progress, 
            border_style="grey35",
            box=box.DOUBLE_EDGE,
            style="on grey7"
        )

    def get_status_panel(self) -> Panel:
        content = f"""
[bold grey93]⟪ Current Target ⟫[/bold grey93]
[grey74]☩ {self.current_url}[/grey74]

[bold grey93]⸸ Statistics ⸸[/bold grey93]
[grey74]├─ Progress:[/grey74] [bright_white]{self.scraped_count}[/bright_white][grey50]/[/grey50][grey74]{self.max_pages}[/grey74]
[grey74]├─ Souls Claimed:[/grey74] [bright_white]{len(self.visited)}[/bright_white]
[grey74]└─ Crypt Path:[/grey74] [grey50]{self.output_dir}[/grey50]
"""
        return Panel(
            content, 
            title="[bold grey93]⚰ The Reaper's Chronicle ⚰[/bold grey93]",
            border_style="grey35",
            box=box.DOUBLE_EDGE,
            style="on grey7"
        )

    def crawl(self):
        layout = self.generate_layout()
        
        layout["header"].update(
            Panel(
                "[bold bright_white]⸸ PROTOR CRAWLER ⸸[/bold bright_white]\n[grey50]Harvesting the Digital Abyss[/grey50]",
                style="bright_white on grey11",
                box=box.DOUBLE_EDGE,
                border_style="grey50"
            )
        )
        
        layout["footer"].update(
            Panel(
                "[italic grey50]⚔ Press Ctrl+C to escape this realm ⚔[/italic grey50]",
                style="grey50 on grey7",
                box=box.DOUBLE_EDGE,
                border_style="grey23"
            )
        )
        
        with Live(layout, refresh_per_second=4, console=self.console) as live:
            while self.queue and self.scraped_count < self.max_pages:
                self.current_url = self.queue.popleft()
                
                if self.current_url in self.visited:
                    continue

                # UI
                layout["progress"].update(self.get_progress_bar())
                layout["current"].update(self.get_status_panel())
                layout["queue"].update(self.get_queue_table())

                self.visited.add(self.current_url)
                
                html, success = fetch_with_curl(self.current_url)
                
                if success:
                    new_links = extract_links(html, self.current_url)
                    for link in new_links:
                        if link not in self.visited and link not in self.queue:
                            self.queue.append(link)
                    
                    scrape_website(self.current_url, self.output_dir, download_js=False)
                    self.scraped_count += 1
                
                layout["progress"].update(self.get_progress_bar())
                layout["current"].update(self.get_status_panel())
                layout["queue"].update(self.get_queue_table())
                
                time.sleep(0.5)
        
        self.console.print(f"\n[bold grey93]⸸ The harvest is complete ⸸[/bold grey93] [grey74]{self.scraped_count} souls claimed.[/grey74]\n")