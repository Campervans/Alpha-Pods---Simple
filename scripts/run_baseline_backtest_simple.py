"""
Simplified baseline CVaR index backtest to test CLEIR implementation.
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
from src.market_data.downloader import download_universe, download_benchmark_data
from src.backtesting.engine import CVaRIndexBacktest

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def main():
    """Main execution function."""
    print("=" * 60)
    print("SIMPLIFIED CVaR INDEX BASELINE BACKTEST")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Configuration
    optimization_config = OptimizationConfig(
        confidence_level=0.95,
        lookback_days=252,  # 1 year
        max_weight=0.05,    # 5% max per stock
        min_weight=0.0,     # Long-only
        solver="SCS",       # Use SCS which we know works
        solver_options={'max_iters': 10000},  # More iterations for convergence
        # CLEIR parameters
        sparsity_bound=1.2,  # L1 norm constraint (allows ~80% of assets)
        benchmark_ticker="SPY"  # Track S&P 500
    )
    
    backtest_config = BacktestConfig(
        start_date="2010-01-01",
        end_date="2024-12-31",
        rebalance_frequency="quarterly",
        transaction_cost_bps=10.0,  # 10 bps per side
        initial_capital=100.0,   # Start at 100 for index
        benchmark_tickers=["SPY"]
    )
    
    # Use a small universe for testing
    test_universe = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'BRK-B',
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE',
        'BAC', 'ABBV', 'KO'
    ]
    
    print(f"\nUsing test universe of {len(test_universe)} stocks")
    
    # Step 1: Download Data
    print("\n" + "="*50)
    print("STEP 1: DATA DOWNLOAD")
    print("="*50)
    
    try:
        # Download main universe data
        print("Downloading price data for test universe...")
        price_data = download_universe(
            test_universe,
            start="2009-07-01",  # Extra buffer for optimization
            end="2024-12-31",
            min_data_points=100
        )
        
        print(f"Successfully downloaded data for {price_data.n_assets} assets")
        print(f"Date range: {price_data.start_date.date()} to {price_data.end_date.date()}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return
    
    # Step 2: Download Benchmark Data
    print("\n" + "="*50)
    print("STEP 2: BENCHMARK DATA")
    print("="*50)
    
    try:
        # Download benchmark
        benchmark_data = download_benchmark_data(
            ["SPY"],
            "2009-07-01",
            "2024-12-31"
        )
        print(f"Downloaded benchmark data for SPY")
        
        # Add benchmark to price data for CLEIR
        if 'SPY' in benchmark_data:
            benchmark_prices = benchmark_data['SPY']
            price_data.prices['SPY'] = benchmark_prices.reindex(price_data.dates)
            price_data.volumes['SPY'] = pd.Series(0, index=price_data.dates)
            price_data.tickers = price_data.tickers + ['SPY']
            print(f"Added SPY to price data for CLEIR tracking")
        
    except Exception as e:
        print(f"Error downloading benchmark data: {e}")
        return
    
    # Step 3: Run CVaR Backtest
    print("\n" + "="*50)
    print("STEP 3: CVAR INDEX BACKTEST")
    print("="*50)
    
    try:
        # Initialize backtester
        backtester = CVaRIndexBacktest(
            price_data, 
            optimization_config,
            asset_tickers=test_universe  # Don't include benchmark in assets
        )
        print("Running CLEIR (CVaR-LASSO Enhanced Index Replication)")
        
        # Run backtest
        cvar_results = backtester.run_backtest(backtest_config)
        
        print("\nCVaR Index Performance Summary:")
        print("-" * 40)
        performance_summary = cvar_results.get_performance_summary()
        for metric, value in performance_summary.items():
            if 'return' in metric or 'volatility' in metric or 'drawdown' in metric or 'cost' in metric:
                print(f"{metric:25}: {value:7.2%}")
            else:
                print(f"{metric:25}: {value:7.3f}")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        
        # Save index values
        cvar_results.index_values.to_csv('results/simplified_cvar_index.csv')
        print("\nSaved CVaR index values to results/simplified_cvar_index.csv")
        
    except Exception as e:
        print(f"Error in CVaR backtest: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main() 