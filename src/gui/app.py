"""Main terminal GUI application."""

import sys
from pathlib import Path
from typing import Optional
import os
import pandas as pd
import numpy as np
from pyfiglet import Figlet
from rich.style import Style
from rich.console import Console
from rich.table import Table
from rich import box

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
            "TASK B - ML-Enhanced CLEIR (60 stocks, 2014-2019 training)",
            "TASK A&B - View Results",
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
            self.run_ml_enhancement()
        elif choice == "4":
            self.view_results()
        elif choice == "5":
            self.data_management_menu()
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
                
                # Show results summary
                console.print("\n[bold cyan]CVaR Optimization Results:[/bold cyan]")
                results_table = Table(show_header=False, box=box.SIMPLE)
                results_table.add_column("Metric", style="dim")
                results_table.add_column("Value", style="bold green")
                
                results_table.add_row("Annual Return", f"{result['annual_return']:.2%}")
                results_table.add_row("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")
                results_table.add_row("Max Drawdown", f"{result['max_drawdown']:.2%}")
                results_table.add_row("Final Index Value", f"{result['final_value']:.2f}")
                results_table.add_row("Total Return", f"{result['total_return']:.2%}")
                
                console.print(results_table)
                
                # Check if performance analysis graph exists
                graph_path = "/Users/james/Alpha-Pods---Simple/results/cvar_index_performance_analysis.png"
                if os.path.exists(graph_path):
                    console.print("\n[bold]📊 Performance Analysis Graph:[/bold]")
                    console.print(f"[link=file://{graph_path}]{graph_path}[/link]")
                    console.print("[dim]Click the link above to view the CVaR index vs benchmark comparison graph[/dim]")
                else:
                    console.print("\n[yellow]Note: Run 'Generate Missing Deliverables' in the Results menu to create performance graphs[/yellow]")
            else:
                show_error(f"Optimization failed: {result['error']}")
        
        console.input("\nPress Enter to continue...")
    
    def run_ml_enhancement(self):
        """Run ML-enhanced CLEIR optimization."""
        clear_screen()
        console.print(create_header("ML-Enhanced CLEIR (Alpha Overlay)"))
        
        # Show information about the ML enhancement
        info_text = Text()
        info_text.append("This feature enhances the CLEIR optimization with machine learning:\n\n", style="bold")
        info_text.append("• Uses Ridge regression to predict 3-month returns\n")
        info_text.append("• Features: momentum, volatility, volume, RSI, and risk-adjusted momentum\n")
        info_text.append("• Selects top 60 stocks based on alpha predictions (expanded from 30)\n")
        info_text.append("• Training: 2014-2019 (fixed window)\n")
        info_text.append("• Testing: 2020-2024 (out-of-sample)\n")
        info_text.append("• Applies CLEIR optimization to the selected universe\n")
        info_text.append("• Walk-forward training prevents look-ahead bias\n")
        
        console.print(Panel(info_text, title="ML Enhancement Overview", border_style="cyan"))
        console.print()
        
        # Get parameters
        config = {
            'start_date': get_text_input("Start date", default="2020-01-01"),
            'end_date': get_text_input("End date", default="2024-12-31"),
            'top_k': int(get_text_input("Number of stocks to select", default="30")),
            'train_years': int(get_text_input("Training window (years)", default="3")),
            'rebalance_freq': get_text_input("Rebalance frequency", default="quarterly")
        }
        
        # Show config summary
        console.print("\n")
        console.print(create_data_table("ML Configuration", config))
        
        if confirm_action("\nProceed with ML-enhanced optimization?"):
            show_info("Running ML-enhanced backtest... This may take several minutes.")
            
            with create_progress_spinner("Running ML enhancement...") as progress:
                task = progress.add_task("Processing...", total=None)
                
                try:
                    # Run the ML backtest script
                    import subprocess
                    import json
                    
                    # Prepare command
                    script_path = Path(__file__).parent.parent.parent / 'scripts' / 'run_simple_ml_backtest.py'
                    
                    # Run the script and capture output
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        capture_output=True,
                        text=True,
                        cwd=str(Path(__file__).parent.parent.parent)
                    )
                    
                    progress.remove_task(task)
                    
                    if result.returncode == 0:
                        show_success("ML enhancement completed!")
                        
                        # Import display utilities
                        from src.gui.results_display import (
                            create_performance_comparison_table,
                            calculate_spy_metrics
                        )
                        
                        # Try to load ML metrics from JSON first
                        ml_metrics = {}
                        json_path = Path(__file__).parent.parent.parent / 'results' / 'ml_metrics.json'
                        
                        if json_path.exists():
                            try:
                                with open(json_path, 'r') as f:
                                    metrics_data = json.load(f)
                                    ml_metrics = metrics_data.get('ml_metrics', {})
                            except Exception as e:
                                print(f"Error loading JSON metrics: {e}")
                        
                        # Fallback to parsing stdout if JSON not available
                        if not ml_metrics:
                            output_lines = result.stdout.split('\n')
                            
                            # Look for ML-Enhanced Performance Summary section
                            in_ml_section = False
                            for i, line in enumerate(output_lines):
                                if "ML-Enhanced Performance Summary" in line:
                                    in_ml_section = True
                                    continue
                                elif "Baseline Performance Summary" in line or "Performance Improvement" in line:
                                    in_ml_section = False
                                    
                                if in_ml_section:
                                    if "Total Return:" in line and 'total_return' not in ml_metrics:
                                        try:
                                            ml_metrics['total_return'] = float(line.split(':')[1].strip().rstrip('%')) / 100
                                        except:
                                            pass
                                    elif "Annual Return:" in line and 'annual_return' not in ml_metrics:
                                        try:
                                            ml_metrics['annual_return'] = float(line.split(':')[1].strip().rstrip('%')) / 100
                                        except:
                                            pass
                                    elif "Volatility:" in line and 'volatility' not in ml_metrics:
                                        try:
                                            ml_metrics['volatility'] = float(line.split(':')[1].strip().rstrip('%')) / 100
                                        except:
                                            pass
                                    elif "Sharpe Ratio:" in line and 'sharpe_ratio' not in ml_metrics:
                                        try:
                                            ml_metrics['sharpe_ratio'] = float(line.split(':')[1].strip())
                                        except:
                                            pass
                                    elif "Max Drawdown:" in line and 'max_drawdown' not in ml_metrics:
                                        try:
                                            ml_metrics['max_drawdown'] = float(line.split(':')[1].strip().rstrip('%')) / 100
                                        except:
                                            pass
                        
                        # Don't add default values - let table display "—" for missing metrics
                        
                        # Calculate SPY metrics
                        spy_metrics = calculate_spy_metrics(config['start_date'], config['end_date'])
                        
                        # Try to load baseline CLEIR metrics
                        cleir_metrics = None
                        try:
                            cleir_path = Path(__file__).parent.parent.parent / 'results' / 'cleir_index_gui.csv'
                            if cleir_path.exists():
                                cleir_df = pd.read_csv(cleir_path, index_col=0, parse_dates=True)
                                # Filter to same date range
                                cleir_df = cleir_df.loc[config['start_date']:config['end_date']]
                                if len(cleir_df) > 0:
                                    cleir_returns = cleir_df.squeeze().pct_change().dropna()
                                    cleir_metrics = {
                                        'total_return': (cleir_df.iloc[-1] / cleir_df.iloc[0]).squeeze() - 1,
                                        'annual_return': ((cleir_df.iloc[-1] / cleir_df.iloc[0]).squeeze() ** (252 / len(cleir_df))) - 1,
                                        'volatility': cleir_returns.std() * np.sqrt(252),
                                        'sharpe_ratio': (cleir_returns.mean() * 252) / (cleir_returns.std() * np.sqrt(252)),
                                        'max_drawdown': (cleir_df.squeeze() / cleir_df.squeeze().expanding().max() - 1).min()
                                    }
                        except Exception as e:
                            print(f"Could not load baseline CLEIR: {e}")
                        
                        # Create and display comparison table
                        if ml_metrics:
                            comparison_table = create_performance_comparison_table(
                                ml_metrics,
                                cleir_metrics,
                                spy_metrics
                            )
                            
                            console.print("\n")
                            console.print(comparison_table)
                        
                        # List generated files
                        console.print("\n[bold]Generated files:[/bold]")
                        files = [
                            "ml_enhanced_index.csv",
                            "ml_feature_importance.png",
                            "ml_shap_analysis.png",
                            "ml_performance_comparison.png",
                            "ml_predictions_analysis.png",
                            "ml_performance_report.md"
                        ]
                        for file in files:
                            file_path = Path(__file__).parent.parent.parent / 'results' / file
                            if file_path.exists():
                                console.print(f"  ✓ {file}")
                        
                        # Show abbreviated output from script
                        if result.stdout:
                            # Extract key information from output
                            output_lines = result.stdout.split('\n')
                            key_lines = []
                            for line in output_lines:
                                if any(keyword in line for keyword in ['IC:', 'Rank stability:', 'Prediction Diagnostics']):
                                    key_lines.append(line)
                            
                            if key_lines:
                                console.print("\n[dim]Key insights:[/dim]")
                                for line in key_lines[-5:]:  # Last 5 key insights
                                    console.print(f"  {line.strip()}")
                    else:
                        show_error("ML enhancement failed!")
                        if result.stderr:
                            console.print("\n[red]Error output:[/red]")
                            console.print(result.stderr)
                        
                except Exception as e:
                    try:
                        progress.remove_task(task)
                    except:
                        pass  # Task might already be removed
                    show_error(f"Error running ML enhancement: {str(e)}")
        
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
        about_text.append("\nTask A Requirements:\n", style="bold")
        about_text.append("• Universe: 60 liquid stocks from S&P 100\n")
        about_text.append("• Optimization: 95% daily CVaR\n")
        about_text.append("• Constraints: Long-only, max 5% per stock\n")
        about_text.append("• Rebalancing: Quarterly with 10bps costs\n")
        
        about_text.append("\nTask B Requirements:\n", style="bold")
        about_text.append("• ML Enhancement: Ridge regression with technical features\n")
        about_text.append("• Alpha Overlay: Select top 30 stocks based on predictions\n")
        about_text.append("• Walk-Forward: 3-year training window, no look-ahead bias\n")
        about_text.append("• Interpretability: Feature importance visualization\n")
        
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
            
            # Task A deliverables
            status_table.add_row(
                "[bold]Task A:[/bold] Daily Index Values", 
                "✓" if status['daily_values'] else "✗",
                "daily_index_values.csv"
            )
            status_table.add_row(
                "[bold]Task A:[/bold] Performance Metrics", 
                "✓" if status['metrics_table'] else "✗",
                "performance_summary.csv"
            )
            status_table.add_row(
                "[bold]Task A:[/bold] Comparison Plot", 
                "✓" if status['comparison_plot'] else "✗",
                "index_performance_analysis.png"
            )
            
            # Task B deliverables
            ml_results_path = Path(self.results_controller.results_dir) / 'ml_enhanced_index.csv'
            ml_feature_path = Path(self.results_controller.results_dir) / 'ml_feature_importance.png'
            ml_method_path = Path(self.results_controller.results_dir) / 'ml_method_note.md'
            
            status_table.add_row(
                "[bold]Task B:[/bold] ML Enhanced Index", 
                "✓" if ml_results_path.exists() else "✗",
                "ml_enhanced_index.csv"
            )
            status_table.add_row(
                "[bold]Task B:[/bold] Feature Importance", 
                "✓" if ml_feature_path.exists() else "✗",
                "ml_feature_importance.png"
            )
            status_table.add_row(
                "[bold]Task B:[/bold] Method Note", 
                "✓" if ml_method_path.exists() else "✗",
                "ml_method_note.md"
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
                
                # Show results summary
                console.print("\n[bold cyan]CLEIR Optimization Results:[/bold cyan]")
                results_table = Table(show_header=False, box=box.SIMPLE)
                results_table.add_column("Metric", style="dim")
                results_table.add_column("Value", style="bold green")
                
                results_table.add_row("Annual Return", f"{result['annual_return']:.2%}")
                results_table.add_row("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")
                results_table.add_row("Max Drawdown", f"{result['max_drawdown']:.2%}")
                results_table.add_row("Final Index Value", f"{result['final_value']:.2f}")
                results_table.add_row("Total Return", f"{result['total_return']:.2%}")
                
                console.print(results_table)
                
                # Check if performance analysis graph exists
                graph_path = "/Users/james/Alpha-Pods---Simple/results/cleir_index_performance_analysis.png"
                if os.path.exists(graph_path):
                    console.print("\n[bold]📊 Performance Analysis Graph:[/bold]")
                    console.print(f"[link=file://{graph_path}]{graph_path}[/link]")
                    console.print("[dim]Click the link above to view the CLEIR index vs benchmark comparison graph[/dim]")
                else:
                    console.print("\n[yellow]Note: Run 'Generate Missing Deliverables' in the Results menu to create performance graphs[/yellow]")
            else:
                show_error(f"Optimization failed: {result['error']}")
        
        console.input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    app = CVaRGUI()
    app.run()


if __name__ == "__main__":
    main() 