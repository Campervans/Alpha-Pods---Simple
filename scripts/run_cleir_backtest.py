"""
Run CLEIR (CVaR-LASSO Enhanced Index Replication) backtest aligned with paper's formulation.

This script implements the CLEIR model with the following variables matching the paper:
- Y_t: Benchmark index return at time t
- R_it: Return of i-th candidate stock at time t  
- w_i: Weight of i-th stock (allows negative for short selling)
- p: Number of candidate stocks
- CVaR_α: Conditional Value-at-Risk at confidence level α
- s: LASSO penalty constant (set to 1.5 as in paper)
- ζ (zeta): VaR threshold
- z_t: Slack variables for tracking error exceeding VaR
- u_i: Dummy variables for LASSO linearization
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.schemas import UniverseConfig, OptimizationConfig, BacktestConfig
from src.market_data.downloader import create_sp100_list, download_universe, download_benchmark_data
from src.market_data.universe import select_liquid_universe
from src.backtesting.engine import CVaRIndexBacktest
from src.backtesting.metrics import calculate_cvar, calculate_max_drawdown, calculate_turnover_stats

# Suppress warnings
warnings.filterwarnings('ignore')


def main():
    """Main execution function for CLEIR backtest."""
    print("=" * 60)
    print("RUNNING CLEIR BACKTEST (Paper-Aligned)")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Configuration aligned with paper
    universe_config = UniverseConfig(
        n_stocks=100,  # Paper uses larger universe for CLEIR
        lookback_days=126,
        metric="dollar_volume",
        min_price=5.0,
        min_trading_days=100
    )
    
    # CLEIR optimization config with paper's parameters
    optimization_config = OptimizationConfig(
        confidence_level=0.95,  # α = 0.95 for CVaR
        lookback_days=252,      # T = 252 trading days
        max_weight=1.0,         # No upper bound per asset (paper allows any weight)
        min_weight=-1.0,        # Allow short selling as in paper
        solver="ECOS",
        sparsity_bound=1.5,     # s = 1.5 as specified in paper
        benchmark_ticker="SPY",  # Using SPY as benchmark index
        solver_options={'max_iters': 10000}
    )
    
    # For practical implementation, we'll use long-only constraint
    # Comment out the line below to allow short selling as in paper
    optimization_config.min_weight = 0.0  # Long-only for practical implementation
    
    backtest_config = BacktestConfig(
        start_date="2010-01-01",
        end_date="2024-12-31",
        rebalance_frequency="quarterly",
        transaction_cost_bps=10.0,
        initial_capital=100.0,
        benchmark_tickers=["SPY"]
    )
    
    # Step 1: Universe Selection (p candidate stocks)
    print("\nStep 1: Selecting candidate universe...")
    
    # Get S&P 100 candidates
    sp100_tickers = create_sp100_list()
    
    # Select most liquid stocks (this gives us our p stocks)
    selected_universe = select_liquid_universe(
        sp100_tickers,
        universe_config,
        start_date="2023-01-01",
        end_date="2024-12-31"
    )
    
    print(f"Selected p = {len(selected_universe)} candidate stocks")
    
    # Step 2: Download Data
    print("\nStep 2: Downloading price data...")
    
    # Download universe data (R_it: returns of candidate stocks)
    price_data = download_universe(
        selected_universe,
        start="2009-07-01",
        end="2024-12-31",
        min_data_points=100
    )
    
    # Download benchmark data (Y_t: benchmark returns)
    benchmark_data = download_benchmark_data(
        ["SPY"],
        "2009-07-01", 
        "2024-12-31"
    )
    
    # Merge benchmark with asset data for CLEIR
    # This ensures we have aligned dates
    all_tickers = selected_universe + ["SPY"]
    all_prices = pd.DataFrame(index=price_data.dates)
    
    # Add asset prices
    for i, ticker in enumerate(selected_universe):
        all_prices[ticker] = price_data.prices[:, i]
    
    # Add benchmark prices
    all_prices["SPY"] = benchmark_data["SPY"].reindex(price_data.dates)
    
    # Create combined PriceData object
    from src.utils.schemas import PriceData
    combined_data = PriceData(
        tickers=all_tickers,
        dates=price_data.dates,
        prices=all_prices.values,
        start_date=price_data.start_date,
        end_date=price_data.end_date,
        n_assets=len(all_tickers)
    )
    
    print(f"Data prepared: {len(selected_universe)} assets + benchmark")
    print(f"Date range: {combined_data.start_date.date()} to {combined_data.end_date.date()}")
    
    # Step 3: Run CLEIR Backtest
    print("\nStep 3: Running CLEIR backtest...")
    print(f"CLEIR parameters:")
    print(f"  - Sparsity bound (s): {optimization_config.sparsity_bound}")
    print(f"  - Confidence level (α): {optimization_config.confidence_level}")
    print(f"  - Min weight: {optimization_config.min_weight} (long-only)" if optimization_config.min_weight >= 0 else f"  - Min weight: {optimization_config.min_weight} (allows short selling)")
    
    # Initialize backtester with CLEIR mode
    backtester = CVaRIndexBacktest(
        combined_data, 
        optimization_config,
        asset_tickers=selected_universe  # Specify which tickers are assets vs benchmark
    )
    
    # Run backtest
    cleir_results = backtester.run_backtest(backtest_config)
    
    print("\nCLEIR Performance:")
    print(f"Total Return: {cleir_results.total_return:.2%}")
    print(f"Annual Return: {cleir_results.annual_return:.2%}")
    print(f"Sharpe Ratio: {cleir_results.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {cleir_results.max_drawdown:.2%}")
    
    # Analyze sparsity
    avg_nonzero = 0
    for event in cleir_results.rebalance_events:
        weights = cleir_results.weights_history.loc[event.date, selected_universe].values
        nonzero = np.sum(np.abs(weights) > 1e-6)
        avg_nonzero += nonzero
    avg_nonzero /= len(cleir_results.rebalance_events)
    
    print(f"\nSparsity Analysis:")
    print(f"Average active positions: {avg_nonzero:.0f} / {len(selected_universe)}")
    print(f"Average sparsity: {(1 - avg_nonzero/len(selected_universe)):.1%}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Save CLEIR index values
    cleir_results.index_values.to_csv('results/cleir_index_values.csv')
    
    # Save summary
    summary = {
        'Model': 'CLEIR',
        'Sparsity_Bound_s': optimization_config.sparsity_bound,
        'Confidence_Level_alpha': optimization_config.confidence_level,
        'Num_Candidates_p': len(selected_universe),
        'Total_Return_Pct': cleir_results.total_return * 100,
        'Annual_Return_Pct': cleir_results.annual_return * 100,
        'Sharpe_Ratio': cleir_results.sharpe_ratio,
        'Max_Drawdown_Pct': cleir_results.max_drawdown * 100,
        'Avg_Active_Positions': avg_nonzero,
        'Avg_Sparsity_Pct': (1 - avg_nonzero/len(selected_universe)) * 100
    }
    
    pd.DataFrame([summary]).to_csv('results/cleir_summary.csv', index=False)
    
    print("\n" + "=" * 60)
    print("CLEIR BACKTEST COMPLETE")
    print("=" * 60)
    print("\nResults saved to:")
    print("- results/cleir_index_values.csv")
    print("- results/cleir_summary.csv")
    
    print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main() 