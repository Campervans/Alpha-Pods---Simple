"""Main terminal GUI application."""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.gui.components import (
    console, create_header, create_menu, show_error, 
    show_success, show_info, create_data_table,
    confirm_action, get_text_input, clear_screen,
    create_progress_spinner
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
        console.print(create_header(
            "CVaR/CLEIR Portfolio Optimization",
            "Terminal GUI v0.1.0"
        ))
        
        options = [
            "Data Management",
            "Run CVaR Optimization", 
            "Run CLEIR Optimization",
            "View Results",
            "Generate Deliverables",
            "About",
            "Exit"
        ]
        
        choice = create_menu("Main Menu", options)
        
        if choice == "1":
            self.data_management_menu()
        elif choice == "2":
            self.run_cvar_optimization()
        elif choice == "3":
            self.run_cleir_optimization()
        elif choice == "4":
            self.view_results()
        elif choice == "5":
            self.generate_deliverables()
        elif choice == "6":
            self.show_about()
        elif choice == "7":
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
        tickers = self.data_controller.get_sp100_tickers()
        show_info(f"Will download {len(tickers)} tickers")
        
        if confirm_action("Proceed with download?"):
            with create_progress_spinner("Downloading data...") as progress:
                task = progress.add_task("Downloading...", total=None)
                result = self.data_controller.download_data(tickers, start_date, end_date)
                progress.remove_task(task)
            
            if result['success']:
                show_success(f"Downloaded {result['n_assets']} assets from {result['start_date']} to {result['end_date']}")
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
                console.print(create_data_table("Results", {
                    'Annual Return': f"{result['annual_return']:.2%}",
                    'Sharpe Ratio': f"{result['sharpe_ratio']:.3f}",
                    'Max Drawdown': f"{result['max_drawdown']:.2%}"
                }))
            else:
                show_error(f"Optimization failed: {result['error']}")
        
        console.input("\nPress Enter to continue...")
    
    def generate_deliverables(self):
        """Generate all Task A deliverables."""
        clear_screen()
        console.print(create_header("Generate Deliverables"))
        
        # Check status
        status = self.results_controller.generate_deliverables()
        
        table = Table(title="Deliverables Status", show_header=True)
        table.add_column("Deliverable", style="cyan")
        table.add_column("Status", style="green")
        
        table.add_row("Daily Index Values CSV", "✓" if status['daily_values'] else "✗")
        table.add_row("Performance Metrics Table", "✓" if status['metrics_table'] else "✗")
        table.add_row("Comparison Plot", "✓" if status['comparison_plot'] else "✗")
        
        console.print(table)
        
        if all(status.values()):
            show_success("All deliverables are ready!")
        else:
            show_info("Some deliverables are missing. Run optimization first.")
        
        console.input("\nPress Enter to continue...")
    
    def show_about(self):
        """Show about screen."""
        clear_screen()
        console.print(create_header("About"))
        
        about_text = Text()
        about_text.append("CVaR/CLEIR Portfolio Optimization System\n\n", style="bold")
        about_text.append("This system implements CVaR-based portfolio optimization\n")
        about_text.append("with optional CLEIR (CVaR-LASSO Enhanced Index Replication).\n\n")
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
        """View optimization results."""
        clear_screen()
        console.print(create_header("View Results"))
        
        # Load performance summary
        df = self.results_controller.load_performance_summary()
        if df is not None:
            table = Table(title="Performance Summary", show_header=True)
            
            # Add columns
            for col in df.columns:
                table.add_column(col)
            
            # Add rows
            for _, row in df.iterrows():
                table.add_row(*[str(val) for val in row.values])
            
            console.print(table)
        else:
            show_info("No results found. Run optimization first.")
        
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
                show_success(f"Downloaded {result['n_assets']} assets")
            else:
                show_error(f"Download failed: {result['error']}")
        
        console.input("\nPress Enter to continue...")
    
    def view_cached_data(self):
        """View cached data details."""
        clear_screen()
        console.print(create_header("Cached Data"))
        
        cached = self.data_controller.get_cached_tickers()
        if cached:
            table = Table(title=f"Cached Tickers ({len(cached)} total)", show_header=False)
            table.add_column("Ticker", style="cyan")
            
            # Show in columns
            for ticker in cached[:50]:  # Limit display
                table.add_row(ticker)
            
            if len(cached) > 50:
                table.add_row("...")
            
            console.print(table)
        else:
            show_info("No cached data found")
        
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
                console.print(create_data_table("Results", {
                    'Annual Return': f"{result['annual_return']:.2%}",
                    'Sharpe Ratio': f"{result['sharpe_ratio']:.3f}",
                    'Max Drawdown': f"{result['max_drawdown']:.2%}"
                }))
            else:
                show_error(f"Optimization failed: {result['error']}")
        
        console.input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    app = CVaRGUI()
    app.run()


if __name__ == "__main__":
    main() 