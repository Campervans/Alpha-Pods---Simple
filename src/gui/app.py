"""Main terminal GUI application."""

import sys
from pathlib import Path
from typing import Optional
import os
import pandas as pd
from pyfiglet import Figlet
from rich.style import Style

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.gui.components import (
    console, create_header, create_menu, show_error, 
    show_success, show_info, create_data_table,
    confirm_action, get_text_input, clear_screen,
    create_progress_spinner, create_file_tree, create_interactive_file_tree
)
from src.gui.controllers import DataController, OptimizationController, ResultsController
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text


class CVaRGUI:
    """Main GUI application class."""
    
    def __init__(self):
        self.data_controller = DataController()
        self.optimization_controller = OptimizationController()
        self.results_controller = ResultsController()
        self.running = True
    
    def run(self):
        """Main application loop."""
        while self.running:
            clear_screen()
            self.show_main_menu()
    
    def show_main_menu(self):
        """Display main menu."""
        # Create ASCII banner
        figlet = Figlet(font='standard', justify='center')
        ascii_banner = figlet.renderText('eToro Alpha Pod')
        
        # eToro green color style
        etoro_green = Style(color="#6ebe44")  # eToro brand green
        
        # Print the banner in eToro green
        console.print(ascii_banner, style=etoro_green)
        
        console.print(create_header(
            "CVaR/CLEIR Portfolio Optimization",
            "Terminal GUI v0.1.0"
        ))
        
        options = [
            "TASK A - Run CLEIR Optimization",
            "TASK A - Run CVaR Optimization",
            "TASK A - View Results",
            "Data Management",
            "About",
            "Exit"
        ]
        
        choice = create_menu("Main Menu", options)
        
        if choice == "1":
            self.run_cleir_optimization()
        elif choice == "2":
            self.run_cvar_optimization()
        elif choice == "3":
            self.view_results()
        elif choice == "4":
            self.data_management_menu()
        elif choice == "5":
            self.show_about()
        elif choice == "6":
            self.running = False
            show_info("Goodbye!")
    
    def data_management_menu(self):
        """Data management submenu."""
        clear_screen()
        console.print(create_header("Data Management"))
        
        # Show cached tickers
        cached = self.data_controller.get_cached_tickers()
        if cached:
            show_info(f"Cached tickers: {', '.join(cached[:10])}{'...' if len(cached) > 10 else ''}")
        else:
            show_info("No cached data found")
        
        options = [
            "Download S&P 100 data",
            "Download custom tickers",
            "View cached data",
            "Clear cache",
            "Back to main menu"
        ]
        
        choice = create_menu("Data Options", options)
        
        if choice == "1":
            self.download_sp100_data()
        elif choice == "2":
            self.download_custom_data()
        elif choice == "3":
            self.view_cached_data()
        elif choice == "4":
            self.clear_cache()
        elif choice == "5":
            return
    
    def download_sp100_data(self):
        """Download S&P 100 data."""
        clear_screen()
        console.print(create_header("Download S&P 100 Data"))
        
        # Get date range
        start_date = get_text_input("Start date (YYYY-MM-DD)", default="2009-07-01")
        end_date = get_text_input("End date (YYYY-MM-DD)", default="2024-12-31")
        
        # Get S&P 100 tickers
        tickers = self.data_controller.get_universe_list()
        show_info(f"Will download {len(tickers)} tickers")
        
        if confirm_action("Proceed with download?"):
            with create_progress_spinner("Downloading data...") as progress:
                task = progress.add_task("Downloading...", total=None)
                result = self.data_controller.download_data(tickers, start_date, end_date)
                progress.remove_task(task)
            
            if result['success']:
                show_success(f"Downloaded {result['n_assets']} assets from {result['start_date']} to {result['end_date']}")
                if 'warning' in result:
                    show_info(f"⚠️  {result['warning']}")
                if 'failed_tickers' in result and result['failed_tickers']:
                    show_info(f"Note: Portfolio optimization will continue with {result['n_assets']} available assets")
            else:
                show_error(f"Download failed: {result['error']}")
        
        console.input("\nPress Enter to continue...")
    
    def run_cvar_optimization(self):
        """Run CVaR optimization."""
        clear_screen()
        console.print(create_header("CVaR Optimization"))
        
        # Get parameters
        config = {
            'n_stocks': int(get_text_input("Number of stocks", default="60")),
            'confidence_level': float(get_text_input("Confidence level", default="0.95")),
            'max_weight': float(get_text_input("Max weight per stock", default="0.05")),
            'start_date': get_text_input("Start date", default="2010-01-01"),
            'end_date': get_text_input("End date", default="2024-12-31"),
            'rebalance_freq': get_text_input("Rebalance frequency", default="quarterly"),
            'transaction_cost': float(get_text_input("Transaction cost (bps)", default="10"))
        }
        
        # Show config summary
        console.print("\n")
        console.print(create_data_table("Configuration", config))
        
        if confirm_action("\nProceed with optimization?"):
            with create_progress_spinner("Running CVaR optimization...") as progress:
                task = progress.add_task("Optimizing...", total=None)
                result = self.optimization_controller.run_cvar_optimization(config)
                progress.remove_task(task)
            
            if result['success']:
                show_success("Optimization completed!")
                # Results will be available in the results section
            else:
                show_error(f"Optimization failed: {result['error']}")
        
        console.input("\nPress Enter to continue...")
    
    def show_about(self):
        """Show about screen."""
        clear_screen()
        console.print(create_header("About"))
        
        about_text = Text()
        about_text.append("CVaR/CLEIR Portfolio Optimization System\n\n", style="bold")
        about_text.append("This system implements CVaR-based portfolio optimization\n")
        about_text.append("with optional CLEIR (CVaR-LASSO Enhanced Index Replication).\n\n")
        
        about_text.append("Created by: ", style="dim")
        about_text.append("James Campion\n", style="bold cyan")
        about_text.append("eToro: ", style="dim")
        about_text.append("etoro.com/campervans\n", style="cyan underline")
        about_text.append("Email: ", style="dim")
        about_text.append("james@oureasystems.com\n", style="cyan")
        about_text.append("Phone: ", style="dim")
        about_text.append("+971561621929\n\n", style="cyan")
        
        about_text.append("GitHub: ", style="dim")
        about_text.append("https://github.com/Campervans/Alpha-Pods---Simple\n", style="cyan underline")
        about_text.append("\nTask Requirements:\n", style="bold")
        about_text.append("• Universe: 60 liquid stocks from S&P 100\n")
        about_text.append("• Optimization: 95% daily CVaR\n")
        about_text.append("• Constraints: Long-only, max 5% per stock\n")
        about_text.append("• Rebalancing: Quarterly with 10bps costs\n")
        
        console.print(Panel(about_text))
        console.input("\nPress Enter to continue...")
    
    def view_results(self):
        """View optimization results with submenu."""
        while True:
            clear_screen()
            console.print(create_header("View Results & Deliverables"))
            
            # Show deliverables status
            status = self.results_controller.generate_deliverables()
            
            status_table = Table(title="Deliverables Status", show_header=True)
            status_table.add_column("Deliverable", style="cyan")
            status_table.add_column("Status", style="green")
            status_table.add_column("File", style="dim")
            
            status_table.add_row(
                "Daily Index Values", 
                "✓" if status['daily_values'] else "✗",
                "daily_index_values.csv"
            )
            status_table.add_row(
                "Performance Metrics", 
                "✓" if status['metrics_table'] else "✗",
                "performance_summary.csv"
            )
            status_table.add_row(
                "Comparison Plot", 
                "✓" if status['comparison_plot'] else "✗",
                "index_performance_analysis.png"
            )
            
            console.print(status_table)
            console.print()
            
            options = [
                "View Performance Summary",
                "View Daily Index Values",
                "Browse Files (Interactive)",
                "Generate Missing Deliverables",
                "Back to Main Menu"
            ]
            
            choice = create_menu("Results Options", options)
            
            if choice == "1":
                self.view_performance_summary()
            elif choice == "2":
                self.view_daily_index_values()
            elif choice == "3":
                self.view_results_file_tree()
            elif choice == "4":
                self.generate_missing_deliverables()
            elif choice == "5":
                return
    
    def view_performance_summary(self):
        """View performance summary table."""
        clear_screen()
        console.print(create_header("Performance Summary"))
        
        df = self.results_controller.load_performance_summary()
        if df is not None:
            table = Table(title="Performance Metrics", show_header=True)
            
            # Add columns
            for col in df.columns:
                table.add_column(col)
            
            # Add rows
            for _, row in df.iterrows():
                table.add_row(*[str(val) for val in row.values])
            
            console.print(table)
        else:
            show_info("No performance summary found. Run optimization first.")
        
        console.input("\nPress Enter to continue...")
    
    def view_daily_index_values(self):
        """View daily index values."""
        clear_screen()
        console.print(create_header("Daily Index Values"))
        
        path = os.path.join(self.results_controller.results_dir, 'daily_index_values.csv')
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                
                # Show first and last 10 rows
                table = Table(title="Daily Index Values (First 10 and Last 10 rows)", show_header=True)
                
                # Add columns
                for col in df.columns:
                    table.add_column(col)
                
                # Add first 10 rows
                for _, row in df.head(10).iterrows():
                    table.add_row(*[str(val) for val in row.values])
                
                # Add separator
                if len(df) > 20:
                    table.add_row(*["..." for _ in df.columns])
                
                # Add last 10 rows
                if len(df) > 10:
                    for _, row in df.tail(10).iterrows():
                        table.add_row(*[str(val) for val in row.values])
                
                console.print(table)
                show_info(f"Total rows: {len(df)}")
            except Exception as e:
                show_error(f"Error reading file: {str(e)}")
        else:
            show_info("Daily index values file not found. Run optimization first.")
        
        console.input("\nPress Enter to continue...")
    
    def view_results_file_tree(self):
        """View results directory file tree with interactive file viewing."""
        while True:
            clear_screen()
            console.print(create_header("Results File Browser"))
            
            if os.path.exists(self.results_controller.results_dir):
                selected_file = create_interactive_file_tree(self.results_controller.results_dir, "Results")
                
                if selected_file is None:
                    # User chose to go back
                    return
                
                # View the selected file
                self.view_file(selected_file)
            else:
                show_info("Results directory not found.")
                console.input("\nPress Enter to continue...")
                return
    
    def view_file(self, file_path: str):
        """View a file based on its type."""
        clear_screen()
        path = Path(file_path)
        console.print(create_header(f"Viewing: {path.name}"))
        
        try:
            if path.suffix == '.csv':
                # View CSV file
                df = pd.read_csv(file_path)
                
                # Show info about the file
                show_info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Create table for first/last rows
                table = Table(title="Data Preview (First 10 and Last 10 rows)", show_header=True)
                
                # Add columns
                for col in df.columns:
                    table.add_column(col, style="cyan" if col == df.columns[0] else "white")
                
                # Add first 10 rows
                for _, row in df.head(10).iterrows():
                    table.add_row(*[str(val) for val in row.values])
                
                # Add separator if needed
                if len(df) > 20:
                    table.add_row(*["..." for _ in df.columns])
                
                # Add last 10 rows if different from first
                if len(df) > 10:
                    for _, row in df.tail(10).iterrows():
                        table.add_row(*[str(val) for val in row.values])
                
                console.print(table)
                
                # Show column info
                console.print("\n[bold]Column Information:[/bold]")
                col_table = Table(show_header=True)
                col_table.add_column("Column", style="cyan")
                col_table.add_column("Type", style="yellow")
                col_table.add_column("Non-Null Count", style="green")
                
                for col in df.columns:
                    col_table.add_row(
                        col,
                        str(df[col].dtype),
                        str(df[col].count())
                    )
                
                console.print(col_table)
                
            elif path.suffix == '.png':
                # For PNG files, show file info
                size = path.stat().st_size / (1024 * 1024)
                show_info(f"Image file: {path.name}")
                show_info(f"Size: {size:.2f} MB")
                show_info("Note: Cannot display images in terminal. Please open the file directly to view.")
                
            elif path.suffix in ['.txt', '.md']:
                # View text files
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Create a panel with the content
                from rich.panel import Panel
                from rich.syntax import Syntax
                
                if path.suffix == '.md':
                    # Use markdown syntax highlighting
                    syntax = Syntax(content, "markdown", theme="monokai", line_numbers=True)
                else:
                    syntax = Syntax(content, "text", theme="monokai", line_numbers=True)
                
                console.print(Panel(syntax, title=path.name, border_style="blue"))
                
            else:
                show_info(f"File type '{path.suffix}' cannot be displayed in terminal.")
                
        except Exception as e:
            show_error(f"Error reading file: {str(e)}")
        
        console.input("\nPress Enter to continue...")
    
    def generate_missing_deliverables(self):
        """Generate missing deliverables."""
        clear_screen()
        console.print(create_header("Generate Missing Deliverables"))
        
        # Check what's missing
        status = self.results_controller.generate_deliverables()
        missing = [k for k, v in status.items() if not v]
        
        if not missing:
            show_success("All deliverables are already generated!")
        else:
            show_info(f"Missing deliverables: {', '.join(missing)}")
            
            if confirm_action("Generate missing deliverables?"):
                # Check if we have the necessary optimization results
                gui_results = ['cvar_index_gui.csv', 'cleir_index_gui.csv']
                existing_results = []
                
                for result_file in gui_results:
                    if os.path.exists(os.path.join(self.results_controller.results_dir, result_file)):
                        existing_results.append(result_file)
                
                if not existing_results:
                    show_error("No optimization results found. Please run CVaR or CLEIR optimization first.")
                else:
                    show_info(f"Found results from: {', '.join(existing_results)}")
                    
                    # Run the generate_final_results script
                    with create_progress_spinner("Generating deliverables...") as progress:
                        task = progress.add_task("Processing...", total=None)
                        
                        try:
                            # Import and run the generation script
                            sys.path.append(str(Path(__file__).parent.parent.parent / 'scripts'))
                            from generate_final_results import main as generate_results
                            
                            # Temporarily redirect stdout to capture output
                            import io
                            from contextlib import redirect_stdout
                            
                            f = io.StringIO()
                            with redirect_stdout(f):
                                generate_results()
                            
                            progress.remove_task(task)
                            show_success("Deliverables generated successfully!")
                            
                            # Show what was generated
                            new_status = self.results_controller.generate_deliverables()
                            newly_generated = [k for k, v in new_status.items() if v and k in missing]
                            if newly_generated:
                                show_info(f"Generated: {', '.join(newly_generated)}")
                        
                        except Exception as e:
                            progress.remove_task(task)
                            show_error(f"Error generating deliverables: {str(e)}")
        
        console.input("\nPress Enter to continue...")
    
    def download_custom_data(self):
        """Download custom ticker data."""
        clear_screen()
        console.print(create_header("Download Custom Data"))
        
        tickers_input = get_text_input("Enter tickers (comma-separated)")
        tickers = [t.strip().upper() for t in tickers_input.split(',')]
        
        start_date = get_text_input("Start date (YYYY-MM-DD)", default="2009-07-01")
        end_date = get_text_input("End date (YYYY-MM-DD)", default="2024-12-31")
        
        if confirm_action(f"Download {len(tickers)} tickers?"):
            with create_progress_spinner("Downloading data...") as progress:
                task = progress.add_task("Downloading...", total=None)
                result = self.data_controller.download_data(tickers, start_date, end_date)
                progress.remove_task(task)
            
            if result['success']:
                show_success(f"Downloaded {result['n_assets']} assets from {result['start_date']} to {result['end_date']}")
                if 'warning' in result:
                    show_info(f"⚠️  {result['warning']}")
                if 'failed_tickers' in result and result['failed_tickers']:
                    show_info(f"Note: Portfolio optimization will continue with {result['n_assets']} available assets")
            else:
                show_error(f"Download failed: {result['error']}")
        
        console.input("\nPress Enter to continue...")
    
    def view_cached_data(self):
        """View cached data details."""
        clear_screen()
        console.print(create_header("Cached Data"))
        
        cached = self.data_controller.get_cached_tickers()
        if cached:
            # Create a more detailed table
            table = Table(title=f"Cached Tickers ({len(cached)} total)", show_header=True)
            table.add_column("Ticker", style="cyan", width=10)
            table.add_column("File Size", style="yellow", width=12)
            table.add_column("Last Modified", style="green", width=20)
            
            # Get file details
            import os
            from datetime import datetime
            
            # Show first 50 tickers with details
            for ticker in cached[:50]:
                file_path = os.path.join(self.data_controller.cache_dir, f"{ticker}.pkl")
                if os.path.exists(file_path):
                    file_stat = os.stat(file_path)
                    file_size = file_stat.st_size / 1024  # Convert to KB
                    mod_time = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    table.add_row(ticker, f"{file_size:.1f} KB", mod_time)
                else:
                    table.add_row(ticker, "N/A", "N/A")
            
            if len(cached) > 50:
                table.add_row("...", "...", "...")
                table.add_row(f"({len(cached) - 50} more)", "", "")
            
            console.print(table)
            
            # Show summary statistics
            console.print("\n[bold]Summary:[/bold]")
            show_info(f"Total cached tickers: {len(cached)}")
            show_info(f"Cache directory: {os.path.abspath(self.data_controller.cache_dir)}")
            
            # Calculate total cache size
            total_size = 0
            for ticker in cached:
                file_path = os.path.join(self.data_controller.cache_dir, f"{ticker}.pkl")
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
                # Also count meta files
                meta_path = os.path.join(self.data_controller.cache_dir, f"{ticker}_meta.pkl")
                if os.path.exists(meta_path):
                    total_size += os.path.getsize(meta_path)
            
            show_info(f"Total cache size: {total_size / (1024 * 1024):.1f} MB")
        else:
            show_info("No cached data found")
            show_info(f"Cache directory: {os.path.abspath(self.data_controller.cache_dir)}")
        
        console.input("\nPress Enter to continue...")
    
    def clear_cache(self):
        """Clear all cached data."""
        if confirm_action("Clear all cached data? This cannot be undone."):
            if self.data_controller.clear_cache():
                show_success("Cache cleared successfully")
            else:
                show_error("Failed to clear cache")
        
        console.input("\nPress Enter to continue...")
    
    def run_cleir_optimization(self):
        """Run CLEIR optimization."""
        clear_screen()
        console.print(create_header("CLEIR Optimization"))
        
        # Similar to CVaR but with additional parameters
        config = {
            'n_stocks': int(get_text_input("Number of stocks", default="60")),
            'confidence_level': float(get_text_input("Confidence level", default="0.95")),
            'max_weight': float(get_text_input("Max weight per stock", default="0.05")),
            'sparsity_bound': float(get_text_input("Sparsity bound", default="1.2")),
            'benchmark_ticker': get_text_input("Benchmark ticker", default="SPY"),
            'start_date': get_text_input("Start date", default="2010-01-01"),
            'end_date': get_text_input("End date", default="2024-12-31"),
            'rebalance_freq': get_text_input("Rebalance frequency", default="quarterly"),
            'transaction_cost': float(get_text_input("Transaction cost (bps)", default="10"))
        }
        
        console.print("\n")
        console.print(create_data_table("Configuration", config))
        
        if confirm_action("\nProceed with CLEIR optimization?"):
            with create_progress_spinner("Running CLEIR optimization...") as progress:
                task = progress.add_task("Optimizing...", total=None)
                result = self.optimization_controller.run_cleir_optimization(config)
                progress.remove_task(task)
            
            if result['success']:
                show_success("Optimization completed!")
                # Results are already shown in the visualization - no need for duplicate table
            else:
                show_error(f"Optimization failed: {result['error']}")
        
        console.input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    app = CVaRGUI()
    app.run()


if __name__ == "__main__":
    main() 