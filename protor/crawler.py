import time
import os
import threading
from queue import Queue
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
from protor.utils import get_default_output_dir, safe_filename
from protor.rag_analyzer import RAGAnalyzer


class Crawler:
    def __init__(self, start_url: str, max_pages: int = 100, output_dir: str = None, 
                 analyze: bool = False, model: str = "llama3", focus: str = "general"):
        self.start_url = start_url
        self.max_pages = max_pages
        self.output_dir = output_dir or get_default_output_dir()
        self.analyze = analyze
        self.model = model
        self.focus = focus
        self.visited = set()
        self.queue = deque([start_url])
        self.scraped_count = 0
        self.analyzed_count = 0
        self._analyzing = False
        self.current_url = ""
        self.console = Console()
        self.crawled_pages = []
        
        if self.analyze:
            self.analysis_queue = Queue()
            self.analysis_results = {}  # {url: status} - 'queued', 'analyzing', 'complete'
            self.analysis_thread = None
            self.stop_analysis = threading.Event()
        
        if self.analyze:
            self.rag_analyzer = RAGAnalyzer(model=model, focus=focus)


    def generate_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=5),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        if self.analyze:
            layout["body"].split_row(
                Layout(name="current", ratio=2),
                Layout(name="queue", ratio=1),
                Layout(name="analysis", ratio=1)
            )
        else:
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
    
    def get_analysis_table(self) -> Panel:
        """Generate analysis status table"""
        if not self.analyze:
            return Panel("")
            
        table = Table(
            show_header=True,
            header_style="bold grey93 on grey11",
            box=box.SIMPLE,
            border_style="grey35",
            padding=(0, 1)
        )
        table.add_column("URL", style="grey74", no_wrap=True, max_width=25)
        table.add_column("Status", style="yellow", justify="center", width=8)
        
        items = list(self.analysis_results.items())[-10:]
        for url, status in items:
            url_short = url.split('/')[-1][:22] if '/' in url else url[:22]
            if status == 'analyzing':
                table.add_row(url_short, "[yellow]⟡[/yellow]")
            elif status == 'complete':
                table.add_row(url_short, "[green]✓[/green]")
            else:  # queued
                table.add_row(url_short, "[grey50]⋯[/grey50]")
        
        complete_count = sum(1 for s in self.analysis_results.values() if s == 'complete')
        total_count = len(self.analysis_results)
        
        return Panel(
            table,
            title=f"[bold grey93]⟪ Analysis: {complete_count}/{total_count} ⟫[/bold grey93]",
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
        analysis_status = ""
        if self.analyze:
            analysis_status = f"\n[grey74]├─ Queued for Analysis:[/grey74] [bright_white]{len(self.crawled_pages)}[/bright_white]"
        
        content = f"""
[bold grey93]⟪ Current Target ⟫[/bold grey93]
[grey74]☩ {self.current_url}[/grey74]

[bold grey93]⸸ Statistics ⸸[/bold grey93]
[grey74]├─ Progress:[/grey74] [bright_white]{self.scraped_count}[/bright_white][grey50]/[/grey50][grey74]{self.max_pages}[/grey74]
[grey74]├─ Souls Claimed:[/grey74] [bright_white]{len(self.visited)}[/bright_white]{analysis_status}
[grey74]└─ Crypt Path:[/grey74] [grey50]{self.output_dir}[/grey50]
"""
        return Panel(
            content, 
            title="[bold grey93]⚰ The Reaper's Chronicle ⚰[/bold grey93]",
            border_style="grey35",
            box=box.DOUBLE_EDGE,
            style="on grey7"
        )
    
    def analysis_worker(self):
        """Worker thread for processing analysis queue"""
        import sys
        from io import StringIO
        
        while not self.stop_analysis.is_set():
            try:
                page = self.analysis_queue.get(timeout=0.5)
                
                self.analysis_results[page['url']] = 'analyzing'
                
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    analysis = self.rag_analyzer.analyze_page(
                        page['url'],
                        page['html'],
                        page['manifest']
                    )
                    
                    domain = safe_filename(urlparse(page['url']).netloc)
                    analysis_filename = f"{domain}_{self.analyzed_count}.md"
                    analysis_path = os.path.join(self.output_dir, analysis_filename)
                    self.rag_analyzer.save_analysis(analysis_path, page['url'], analysis)
                    self.analyzed_count += 1
                    
                    self.analysis_results[page['url']] = 'complete'
                finally:
                    sys.stdout = old_stdout
                    self.analysis_queue.task_done()
                    
            except:
                # Queue empty or timeout, continue
                continue

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
        layout["progress"].update(self.get_progress_bar())
        layout["current"].update(self.get_status_panel())
        layout["queue"].update(self.get_queue_table())
        if self.analyze:
            layout["analysis"].update(self.get_analysis_table())
            self.analysis_thread = threading.Thread(target=self.analysis_worker, daemon=True)
            self.analysis_thread.start()
        
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
                
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    html, success = fetch_with_curl(self.current_url)
                    
                    if success:
                        new_links = extract_links(html, self.current_url)
                        for link in new_links:
                            if link not in self.visited and link not in self.queue:
                                self.queue.append(link)
                        
                        manifest = scrape_website(self.current_url, self.output_dir, download_js=False)
                        self.scraped_count += 1
                        
                        if self.analyze and manifest:
                            page_data = {
                                'url': self.current_url,
                                'html': html,
                                'manifest': manifest
                            }
                            self.analysis_results[self.current_url] = 'queued'
                            self.analysis_queue.put(page_data)
                finally:
                    sys.stdout = old_stdout
                
                # UI
                layout["progress"].update(self.get_progress_bar())
                layout["current"].update(self.get_status_panel())
                layout["queue"].update(self.get_queue_table())
                if self.analyze:
                    layout["analysis"].update(self.get_analysis_table())
                
                time.sleep(0.5)
        
        self.console.print(f"\n[bold grey93]⸸ The harvest is complete ⸸[/bold grey93] [grey74]{self.scraped_count} souls claimed.[/grey74]\n")
        
        # wait for analysis to complete if enabled
        if self.analyze:
            self.console.print(f"[bold grey93]⸸ Waiting for analysis to complete ⸸[/bold grey93]\n")
            self.analysis_queue.join()  # wait for all items to be processed
            self.stop_analysis.set()
            if self.analysis_thread:
                self.analysis_thread.join(timeout=2)  # Wait for thread to finish
            
            complete_count = sum(1 for s in self.analysis_results.values() if s == 'complete')
            self.console.print(f"\n[bold grey93]⸸ Divination complete ⸸[/bold grey93] [grey74]{complete_count} souls analyzed.[/grey74]\n")
    
