"""Terminal visualization for index performance."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box

try:
    import plotext as plt
except ImportError:
    print("Installing plotext...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotext"])
    import plotext as plt

from ..market_data.downloader import download_benchmark_data
from ..utils.core import annualize_return, calculate_sharpe_ratio, calculate_max_drawdown


def plot_index_comparison(cleir_csv_path: str = "results/cleir_index_gui.csv",
                         benchmark_ticker: str = "SPY",
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Plot CLEIR index vs SPY benchmark in terminal and calculate comparison stats.
    
    Returns dict with stats for both indices.
    """
    console = Console()
    
    # Load CLEIR results
    cleir_df = pd.read_csv(cleir_csv_path)
    cleir_df['Date'] = pd.to_datetime(cleir_df['Date'])
    cleir_df.set_index('Date', inplace=True)
    
    # Determine date range
    if start_date is None:
        start_date = cleir_df.index[0].strftime('%Y-%m-%d')
    if end_date is None:
        end_date = cleir_df.index[-1].strftime('%Y-%m-%d')
    
    # Download SPY data
    console.print(f"[yellow]Downloading {benchmark_ticker} data...[/yellow]")
    spy_data = download_benchmark_data([benchmark_ticker], start_date, end_date)
    
    if benchmark_ticker not in spy_data:
        raise ValueError(f"Failed to download {benchmark_ticker} data")
    
    spy_prices = spy_data[benchmark_ticker]
    
    # Align dates
    common_dates = cleir_df.index.intersection(spy_prices.index)
    cleir_aligned = cleir_df.loc[common_dates, 'Index_Value']
    spy_aligned = spy_prices.loc[common_dates]
    
    # Normalize SPY to start at 100
    spy_normalized = (spy_aligned / spy_aligned.iloc[0]) * 100.0
    
    # Calculate returns
    cleir_returns = cleir_aligned.pct_change().dropna()
    spy_returns = spy_normalized.pct_change().dropna()
    
    # Calculate stats for both
    stats = {}
    
    # CLEIR stats
    cleir_total_return = (cleir_aligned.iloc[-1] / cleir_aligned.iloc[0]) - 1.0
    cleir_annual_return = annualize_return(cleir_total_return, len(cleir_returns), 252)
    cleir_sharpe = calculate_sharpe_ratio(cleir_returns, 0.0, 252)
    cleir_max_dd = calculate_max_drawdown(cleir_returns)
    
    stats['cleir'] = {
        'total_return': cleir_total_return,
        'annual_return': cleir_annual_return,
        'sharpe_ratio': cleir_sharpe,
        'max_drawdown': cleir_max_dd,
        'final_value': cleir_aligned.iloc[-1]
    }
    
    # SPY stats
    spy_total_return = (spy_normalized.iloc[-1] / spy_normalized.iloc[0]) - 1.0
    spy_annual_return = annualize_return(spy_total_return, len(spy_returns), 252)
    spy_sharpe = calculate_sharpe_ratio(spy_returns, 0.0, 252)
    spy_max_dd = calculate_max_drawdown(spy_returns)
    
    stats['spy'] = {
        'total_return': spy_total_return,
        'annual_return': spy_annual_return,
        'sharpe_ratio': spy_sharpe,
        'max_drawdown': spy_max_dd,
        'final_value': spy_normalized.iloc[-1]
    }
    
    # Create terminal plot
    plt.clf()  # Clear any existing plot
    plt.theme('dark')  # Use dark theme for better terminal visibility
    
    # Convert dates to numeric for plotting (days since start)
    days_since_start = (common_dates - common_dates[0]).days
    
    # Plot both lines (reverted to original style)
    plt.plot(days_since_start, cleir_aligned.values, label='CLEIR Index', color='green')
    plt.plot(days_since_start, spy_normalized.values, label=f'{benchmark_ticker} (Normalized)', color='cyan')
    
    # Formatting
    plt.title(f'CLEIR Index vs {benchmark_ticker} Performance')
    plt.xlabel('Days Since Start')
    plt.ylabel('Index Value')
    
    # Add grid for better readability
    plt.grid(True, True)
    
    # Set size for terminal display (reverted to original size)
    plt.plotsize(100, 30)
    
    # Show the plot
    console.print("\n[bold cyan]Performance Comparison Chart[/bold cyan]")
    plt.show()
    
    # Create comparison table
    table = Table(title="Performance Comparison", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("CLEIR Index", style="green", justify="right")
    table.add_column(f"{benchmark_ticker}", style="yellow", justify="right")
    table.add_column("Difference", style="magenta", justify="right")
    
    # Add rows with formatted values
    table.add_row(
        "Annual Return",
        f"{stats['cleir']['annual_return']:.2%}",
        f"{stats['spy']['annual_return']:.2%}",
        f"{stats['cleir']['annual_return'] - stats['spy']['annual_return']:.2%}"
    )
    
    table.add_row(
        "Total Return",
        f"{stats['cleir']['total_return']:.2%}",
        f"{stats['spy']['total_return']:.2%}",
        f"{stats['cleir']['total_return'] - stats['spy']['total_return']:.2%}"
    )
    
    table.add_row(
        "Sharpe Ratio",
        f"{stats['cleir']['sharpe_ratio']:.3f}",
        f"{stats['spy']['sharpe_ratio']:.3f}",
        f"{stats['cleir']['sharpe_ratio'] - stats['spy']['sharpe_ratio']:.3f}"
    )
    
    table.add_row(
        "Max Drawdown",
        f"{stats['cleir']['max_drawdown']:.2%}",
        f"{stats['spy']['max_drawdown']:.2%}",
        f"{stats['cleir']['max_drawdown'] - stats['spy']['max_drawdown']:.2%}"
    )
    
    table.add_row(
        "Final Index Value",
        f"{stats['cleir']['final_value']:.2f}",
        f"{stats['spy']['final_value']:.2f}",
        f"{stats['cleir']['final_value'] - stats['spy']['final_value']:.2f}"
    )
    
    console.print("\n")
    console.print(table)
    
    return stats


def create_mini_sparkline(values: list, width: int = 20, height: int = 5) -> str:
    """Create a mini sparkline for inline display."""
    plt.clf()
    plt.plotsize(width, height)
    plt.plot(values)
    plt.theme('dark')
    return plt.build()  # Returns string representation