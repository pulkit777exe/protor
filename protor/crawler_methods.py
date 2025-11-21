    def get_analysis_table(self) -> Panel:
        """Generate analysis status table"""
        table = Table(
            show_header=True,
            header_style="bold grey93 on grey11",
            box=box.SIMPLE,
            border_style="grey35",
            padding=(0, 1)
        )
        table.add_column("URL", style="grey74", no_wrap=True, max_width=30)
        table.add_column("Status", style="yellow", justify="center", width=8)
        
        # Show last 10 analysis items
        items = list(self.analysis_results.items())[-10:]
        for url, status in items:
            url_short = url.split('/')[-1][:25] if '/' in url else url[:25]
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
    
    def analysis_worker(self):
        """Worker thread for processing analysis queue"""
        import sys
        from io import StringIO
        
        while not self.stop_analysis.is_set():
            try:
                # Get item from queue with timeout
                page = self.analysis_queue.get(timeout=0.5)
                
                # Mark as analyzing
                self.analysis_results[page['url']] = 'analyzing'
                
                # Suppress output
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    # Analyze the page
                    analysis = self.rag_analyzer.analyze_page(
                        page['url'],
                        page['html'],
                        page['manifest']
                    )
                    
                    # Save analysis to file
                    domain = safe_filename(urlparse(page['url']).netloc)
                    analysis_filename = f"{domain}_{self.analyzed_count}.md"
                    analysis_path = os.path.join(self.output_dir, analysis_filename)
                    self.rag_analyzer.save_analysis(analysis_path, page['url'], analysis)
                    self.analyzed_count += 1
                    
                    # Mark as complete
                    self.analysis_results[page['url']] = 'complete'
                finally:
                    sys.stdout = old_stdout
                    self.analysis_queue.task_done()
                    
            except:
                # Queue empty or timeout, continue
                continue
