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
                         end_date: Optional[str] = None,
                         include_equal_weight: bool = True) -> Dict[str, Any]:
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
    
    # Initialize equal weight data
    equal_weight_index = None
    equal_weight_returns = None
    
    if include_equal_weight:
        # Get the universe of stocks used in the backtest
        from ..market_data.downloader import create_sp100_since_2010, download_universe
        from ..market_data.universe import select_liquid_universe, apply_universe_filters, calculate_liquidity_scores
        from ..utils.schemas import UniverseConfig
        
        # Use same universe config as the optimization
        universe_config = UniverseConfig(
            n_stocks=60,
            lookback_days=252,
            min_trading_days=200,
            min_price=5.0,
            metric="dollar_volume"
        )
        
        # Get S&P 100 tickers
        sp100_tickers = create_sp100_since_2010()
        
        # Download price data for the full period
        console.print("[yellow]Downloading data for equal-weight calculation...[/yellow]")
        price_data = download_universe(
            sp100_tickers, 
            start_date, 
            end_date,
            min_data_points=252,
            use_cache=True,
            cache_dir="data/raw"
        )
        
        # Apply universe filters
        valid_tickers, filter_results = apply_universe_filters(price_data, universe_config)
        
        if len(valid_tickers) >= universe_config.n_stocks:
            # Calculate liquidity scores
            valid_price_data = pd.DataFrame(index=price_data.dates)
            for ticker in valid_tickers:
                valid_price_data[ticker] = price_data.prices[ticker]
            
            # Calculate liquidity scores
            from ..utils.schemas import PriceData as PriceDataSchema
            valid_price_data_obj = PriceDataSchema(
                tickers=valid_tickers,
                dates=price_data.dates,
                prices=price_data.prices[valid_tickers],
                volumes=price_data.volumes[valid_tickers] if price_data.volumes is not None else None
            )
            
            liquidity_scores = calculate_liquidity_scores(valid_price_data_obj, universe_config)
            
            # Select top N most liquid tickers
            selected_tickers = liquidity_scores.nlargest(universe_config.n_stocks).index.tolist()
            
            # Calculate equal-weighted returns
            selected_prices = price_data.prices[selected_tickers]
            
            # Calculate daily returns for each stock
            stock_returns = selected_prices.pct_change()
            
            # Calculate equal-weighted portfolio returns
            equal_weight_portfolio_returns = stock_returns.mean(axis=1)
            
            # Build equal-weighted index
            equal_weight_index = pd.Series(index=selected_prices.index, dtype=float)
            equal_weight_index.iloc[0] = 100.0
            
            for i in range(1, len(equal_weight_index)):
                equal_weight_index.iloc[i] = equal_weight_index.iloc[i-1] * (1 + equal_weight_portfolio_returns.iloc[i])
            
            console.print(f"[green]Equal-weight index calculated using {len(selected_tickers)} stocks[/green]")
    
    # Align dates
    common_dates = cleir_df.index.intersection(spy_prices.index)
    if equal_weight_index is not None:
        common_dates = common_dates.intersection(equal_weight_index.index)
    
    cleir_aligned = cleir_df.loc[common_dates, 'Index_Value']
    spy_aligned = spy_prices.loc[common_dates]
    
    # Normalize SPY to start at 100
    spy_normalized = (spy_aligned / spy_aligned.iloc[0]) * 100.0
    
    # Align equal weight index if available
    if equal_weight_index is not None:
        equal_weight_aligned = equal_weight_index.loc[common_dates]
        # Normalize to start at 100
        equal_weight_aligned = (equal_weight_aligned / equal_weight_aligned.iloc[0]) * 100.0
        equal_weight_returns = equal_weight_aligned.pct_change().dropna()
    
    # Calculate returns
    cleir_returns = cleir_aligned.pct_change().dropna()
    spy_returns = spy_normalized.pct_change().dropna()
    
    # Calculate stats for all indices
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
    
    # Equal weight stats if available
    if equal_weight_returns is not None:
        ew_total_return = (equal_weight_aligned.iloc[-1] / equal_weight_aligned.iloc[0]) - 1.0
        ew_annual_return = annualize_return(ew_total_return, len(equal_weight_returns), 252)
        ew_sharpe = calculate_sharpe_ratio(equal_weight_returns, 0.0, 252)
        ew_max_dd = calculate_max_drawdown(equal_weight_returns)
        
        stats['equal_weight'] = {
            'total_return': ew_total_return,
            'annual_return': ew_annual_return,
            'sharpe_ratio': ew_sharpe,
            'max_drawdown': ew_max_dd,
            'final_value': equal_weight_aligned.iloc[-1]
        }
    
    # Create terminal plot
    plt.clf()  # Clear any existing plot
    plt.theme('dark')  # Use dark theme for better terminal visibility
    
    # Convert dates to numeric for plotting (days since start)
    days_since_start = (common_dates - common_dates[0]).days
    
    # Plot lines
    plt.plot(days_since_start, cleir_aligned.values, label='CLEIR Index', color='green')
    plt.plot(days_since_start, spy_normalized.values, label=f'{benchmark_ticker} (Cap-Weighted)', color='cyan')
    
    if equal_weight_aligned is not None:
        plt.plot(days_since_start, equal_weight_aligned.values, label='Equal-Weighted', color='yellow')
    
    # Formatting
    plt.title(f'Performance Comparison: CLEIR vs Benchmarks')
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
    
    if 'equal_weight' in stats:
        table.add_column("Equal-Weight", style="yellow", justify="right")
    
    table.add_column(f"{benchmark_ticker} (SPY)", style="blue", justify="right")
    table.add_column("Difference", style="magenta", justify="right")
    
    # Add rows with formatted values
    table.add_row(
        "Annual Return",
        f"{stats['cleir']['annual_return']:.2%}",
        f"{stats['equal_weight']['annual_return']:.2%}" if 'equal_weight' in stats else "N/A",
        f"{stats['spy']['annual_return']:.2%}",
        f"{stats['cleir']['annual_return'] - stats['spy']['annual_return']:.2%}"
    )
    
    table.add_row(
        "Total Return",
        f"{stats['cleir']['total_return']:.2%}",
        f"{stats['equal_weight']['total_return']:.2%}" if 'equal_weight' in stats else "N/A",
        f"{stats['spy']['total_return']:.2%}",
        f"{stats['cleir']['total_return'] - stats['spy']['total_return']:.2%}"
    )
    
    table.add_row(
        "Sharpe Ratio",
        f"{stats['cleir']['sharpe_ratio']:.3f}",
        f"{stats['equal_weight']['sharpe_ratio']:.3f}" if 'equal_weight' in stats else "N/A",
        f"{stats['spy']['sharpe_ratio']:.3f}",
        f"{stats['cleir']['sharpe_ratio'] - stats['spy']['sharpe_ratio']:.3f}"
    )
    
    table.add_row(
        "Max Drawdown",
        f"{stats['cleir']['max_drawdown']:.2%}",
        f"{stats['equal_weight']['max_drawdown']:.2%}" if 'equal_weight' in stats else "N/A",
        f"{stats['spy']['max_drawdown']:.2%}",
        f"{stats['cleir']['max_drawdown'] - stats['spy']['max_drawdown']:.2%}"
    )
    
    table.add_row(
        "Final Index Value",
        f"{stats['cleir']['final_value']:.2f}",
        f"{stats['equal_weight']['final_value']:.2f}" if 'equal_weight' in stats else "N/A",
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


def plot_equity_curves(curves: Dict[str, pd.Series], 
                      title: str = "Strategy Comparison",
                      width: int = 100,
                      height: int = 30) -> None:
    """Plot multiple equity curves on single chart.
    
    Args:
        curves: Dictionary mapping strategy names to pandas Series of daily values
        title: Chart title
        width: Plot width in characters
        height: Plot height in characters
    """
    console = Console()
    
    if not curves:
        console.print("[yellow]No data to plot[/yellow]")
        return
    
    # Clear any existing plot
    plt.clf()
    plt.theme('dark')
    
    # Find common date range
    all_dates = None
    for name, series in curves.items():
        if all_dates is None:
            all_dates = series.index
        else:
            all_dates = all_dates.intersection(series.index)
    
    if len(all_dates) == 0:
        console.print("[red]No overlapping dates between strategies[/red]")
        return
    
    # Convert to days since start for plotting
    days_since_start = (all_dates - all_dates[0]).days
    
    # Color mapping for strategies
    colors = {
        'ML-Enhanced CLEIR': 'green',
        'Baseline CLEIR': 'yellow', 
        'SPY Benchmark': 'cyan',
        'Equal-Weight': 'magenta'
    }
    
    # Plot each curve
    for i, (name, series) in enumerate(curves.items()):
        # Align to common dates
        aligned_values = series.loc[all_dates]
        
        # Normalize to start at 100
        normalized_values = (aligned_values / aligned_values.iloc[0]) * 100
        
        # Get color
        color = colors.get(name, ['red', 'blue', 'white'][i % 3])
        
        # Plot
        plt.plot(days_since_start, normalized_values.values, label=name, color=color)
    
    # Formatting
    plt.title(title)
    plt.xlabel('Days Since Start')
    plt.ylabel('Index Value (Base = 100)')
    plt.grid(True, True)
    plt.plotsize(width, height)
    
    # Show the plot
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    plt.show()


def render_metrics_table(metrics: Dict[str, Dict[str, float]], 
                        title: str = "Performance Metrics Comparison") -> None:
    """Display performance metrics in Rich table format.
    
    Args:
        metrics: Dictionary mapping strategy names to metric dictionaries
        title: Table title
    """
    console = Console()
    
    if not metrics:
        console.print("[yellow]No metrics to display[/yellow]")
        return
    
    # Create table
    table = Table(title=title, box=box.ROUNDED, show_header=True)
    
    # Add metric name column
    table.add_column("Metric", style="cyan", no_wrap=True)
    
    # Add column for each strategy
    strategy_names = list(metrics.keys())
    colors = ['green', 'yellow', 'blue', 'magenta', 'red']
    
    for i, strategy in enumerate(strategy_names):
        color = colors[i % len(colors)]
        table.add_column(strategy, style=color, justify="right")
    
    # Add best performer column
    table.add_column("Best", style="bold white", justify="center")
    
    # Define metrics to show and their formatting
    metric_info = {
        'total_return': ('Total Return', lambda x: f"{x:.2%}", True),  # Higher is better
        'annual_return': ('Annual Return', lambda x: f"{x:.2%}", True),
        'volatility': ('Volatility', lambda x: f"{x:.2%}", False),  # Lower is better
        'sharpe_ratio': ('Sharpe Ratio', lambda x: f"{x:.3f}", True),
        'max_drawdown': ('Max Drawdown', lambda x: f"{x:.2%}", False),
        'avg_turnover': ('Avg Turnover', lambda x: f"{x:.1%}", False),
        'transaction_costs': ('Transaction Costs', lambda x: f"{x:.2%}", False)
    }
    
    # Add rows for each metric
    for metric_key, (display_name, formatter, higher_is_better) in metric_info.items():
        row_values = []
        
        # Collect values for this metric
        metric_values = {}
        for strategy in strategy_names:
            if metric_key in metrics[strategy]:
                value = metrics[strategy][metric_key]
                metric_values[strategy] = value
                row_values.append(formatter(value))
            else:
                row_values.append("—")
        
        # Find best performer
        if metric_values:
            if higher_is_better:
                best_strategy = max(metric_values, key=metric_values.get)
            else:
                best_strategy = min(metric_values, key=metric_values.get)
            
            # Get strategy index for star
            best_idx = strategy_names.index(best_strategy)
            row_values.append("⭐" if len(metric_values) > 1 else "")
            
            # Highlight best value
            row_values[best_idx] = f"[bold]{row_values[best_idx]}[/bold]"
        else:
            row_values.append("")
        
        # Add row
        table.add_row(display_name, *row_values)
    
    # Add separator
    table.add_row("", *[""] * (len(strategy_names) + 1))
    
    # Add relative performance section if we have SPY
    if 'SPY Benchmark' in metrics and len(metrics) > 1:
        spy_metrics = metrics['SPY Benchmark']
        
        for strategy in strategy_names:
            if strategy == 'SPY Benchmark':
                continue
                
            strat_metrics = metrics[strategy]
            
            # Calculate excess return
            if 'annual_return' in strat_metrics and 'annual_return' in spy_metrics:
                excess = strat_metrics['annual_return'] - spy_metrics['annual_return']
                
                row_values = [""] * len(strategy_names)
                row_values[strategy_names.index(strategy)] = f"{excess:+.2%}"
                
                table.add_row(
                    f"Excess vs SPY",
                    *row_values,
                    "📈" if excess > 0 else "📉"
                )
    
    # Display table
    console.print("\n")
    console.print(table)
    
    # Add summary text
    if len(metrics) > 1:
        console.print("\n[dim]⭐ = Best performer | 📈 = Outperforms SPY | 📉 = Underperforms SPY[/dim]")