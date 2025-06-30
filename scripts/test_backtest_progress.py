"""Test script to demonstrate rich progress displays during full backtest."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtesting.engine import CVaRIndexBacktest
from src.utils.schemas import OptimizationConfig, BacktestConfig, PriceData
from rich.console import Console

console = Console()

def create_mock_price_data(n_assets=20, n_days=500):
    """Create mock price data for testing."""
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2022, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate tickers
    tickers = [f"STOCK{i}" for i in range(n_assets)]
    
    # Generate prices with realistic properties
    prices = {}
    for ticker in tickers:
        # Random walk with drift
        returns = np.random.randn(n_days) * 0.02 + 0.0002
        price_series = 100 * np.exp(np.cumsum(returns))
        prices[ticker] = pd.Series(price_series, index=dates)
    
    prices_df = pd.DataFrame(prices)
    
    # Generate volumes
    volumes = {}
    for ticker in tickers:
        volume_series = np.random.lognormal(15, 1, n_days)
        volumes[ticker] = pd.Series(volume_series, index=dates)
    
    volumes_df = pd.DataFrame(volumes)
    
    return PriceData(
        tickers=tickers,
        dates=dates,
        prices=prices_df,
        volumes=volumes_df
    )

def test_backtest_with_progress():
    """Test full backtest with rich progress displays."""
    
    console.print("[bold magenta]Testing Full Backtest with Rich Progress[/bold magenta]")
    console.print("=" * 60)
    
    # Create mock data
    price_data = create_mock_price_data(n_assets=30, n_days=750)
    
    # Create optimization config for CVaR (no sparsity_bound)
    optimization_config = OptimizationConfig(
        confidence_level=0.95,
        lookback_days=252,
        max_weight=0.10,
        min_weight=0.0,
        solver="CLARABEL"
    )
    
    # Create backtest config
    backtest_config = BacktestConfig(
        start_date='2022-06-01',
        end_date='2024-06-01',
        rebalance_frequency='monthly',
        transaction_cost_bps=10.0,
        initial_capital=100.0
    )
    
    console.print("\n[yellow]Running backtest WITHOUT optimization progress bars[/yellow]")
    console.print("-" * 40)
    
    # Create and run backtest without optimization progress
    backtest = CVaRIndexBacktest(
        price_data=price_data,
        optimization_config=optimization_config,
        show_optimization_progress=False  # Clean output
    )
    
    results = backtest.run_backtest(backtest_config)
    
    console.print("\n[bold green]Backtest completed successfully![/bold green]")
    console.print(f"\nNumber of rebalances: {len(results.rebalance_events)}")
    
    # Show turnover summary
    turnovers = [event.turnover for event in results.rebalance_events]
    console.print(f"\n[cyan]Turnover Statistics:[/cyan]")
    console.print(f"  Mean: {np.mean(turnovers):.2%}")
    console.print(f"  Max: {np.max(turnovers):.2%}")
    console.print(f"  Min: {np.min(turnovers):.2%}")

if __name__ == "__main__":
    test_backtest_with_progress() 