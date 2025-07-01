"""
Main script to run baseline CVaR index backtest.

This script orchestrates the entire process from universe selection
to backtest execution and results generation.

Usage: python scripts/run_baseline_backtest.py
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
from src.market_data.downloader import (
    create_sp100_list, 
    create_sp100_since_2010,
    download_universe, 
    save_price_data,
    download_benchmark_data
)
from src.market_data.universe import select_liquid_universe, create_equal_weight_universe
from src.backtesting.engine import CVaRIndexBacktest
from src.backtesting.metrics import create_performance_report, create_monthly_returns_table, save_performance_report, plot_performance_comparison
from src.optimization.risk_models import calculate_portfolio_returns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def main():
    """Main execution function."""
    print("=" * 60)
    print("CVaR INDEX BASELINE BACKTEST")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Configuration
    universe_config = UniverseConfig(
        n_stocks=60,
        lookback_days=126,  # 6 months
        metric="dollar_volume",
        min_price=5.0,
        min_trading_days=100
    )
    
    optimization_config = OptimizationConfig(
        confidence_level=0.95,
        lookback_days=252,  # 1 year
        max_weight=0.05,    # 5% max per stock
        min_weight=0.0,     # Long-only
        solver="ECOS",
        # CLEIR parameters
        sparsity_bound=1.2,  # L1 norm constraint (allows ~80% of assets)
        benchmark_ticker="SPY"  # Track S&P 500
    )
    
    backtest_config = BacktestConfig(
        start_date="2010-01-01",
        end_date="2024-12-31",
        rebalance_frequency="quarterly",
        transaction_cost_bps=10.0,  # 10 bps per side
        initial_capital=1000000.0,   # $1M
        benchmark_tickers=["SPY", "IWV"]  # S&P 500 and Russell 3000
    )
    
    # Step 1: Universe Selection
    print("\n" + "="*50)
    print("STEP 1: UNIVERSE SELECTION")
    print("="*50)
    
    try:
        # Get S&P 100 tickers
        sp100_tickers = create_sp100_since_2010()
        print(f"Loaded {len(sp100_tickers)} S&P 100 candidates")
        
        # Select liquid universe
        selected_universe = select_liquid_universe(
            sp100_tickers, 
            universe_config,
            start_date="2023-01-01",  # Recent period for liquidity analysis
            end_date="2024-12-31"
        )
        
        print(f"Selected {len(selected_universe)} stocks for investment universe")
        print("Top 10 selected stocks:", selected_universe[:10])
        
    except Exception as e:
        print(f"Error in universe selection: {e}")
        print("Using fallback universe...")
        selected_universe = sp100_tickers[:60]  # First 60 as fallback
    
    # Step 2: Data Download
    print("\n" + "="*50)
    print("STEP 2: DATA DOWNLOAD")
    print("="*50)
    
    try:
        # Download main universe data
        print("Downloading price data for selected universe...")
        price_data = download_universe(
            selected_universe,
            start="2009-07-01",  # Extra buffer for optimization
            end="2024-12-31",
            min_data_points=100
        )
        
        # Save price data
        os.makedirs('data/processed', exist_ok=True)
        save_price_data(price_data, 'data/processed/price_data.pkl')
        
        print(f"Successfully downloaded data for {price_data.n_assets} assets")
        print(f"Date range: {price_data.start_date.date()} to {price_data.end_date.date()}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return
    
    # Step 3: Download Benchmark Data
    print("\n" + "="*50)
    print("STEP 3: BENCHMARK DATA")
    print("="*50)
    
    try:
        # Download benchmark for tracking (SPY)
        if optimization_config.benchmark_ticker:
            benchmark_tickers = [optimization_config.benchmark_ticker] + backtest_config.benchmark_tickers
        else:
            benchmark_tickers = backtest_config.benchmark_tickers
            
        benchmark_data = download_benchmark_data(
            benchmark_tickers,
            "2009-07-01",  # Extra buffer for optimization
            backtest_config.end_date
        )
        print(f"Downloaded benchmark data for {len(benchmark_data)} benchmarks")
        
        # Merge benchmark data with main price data for CLEIR
        if optimization_config.benchmark_ticker and optimization_config.benchmark_ticker in benchmark_data:
            # Add benchmark to price data
            benchmark_prices = benchmark_data[optimization_config.benchmark_ticker]
            price_data.prices[optimization_config.benchmark_ticker] = benchmark_prices.reindex(price_data.dates)
            # Also add dummy volume data for the benchmark (benchmarks typically don't have volume)
            price_data.volumes[optimization_config.benchmark_ticker] = pd.Series(0, index=price_data.dates)
            price_data.tickers = price_data.tickers + [optimization_config.benchmark_ticker]
            print(f"Added {optimization_config.benchmark_ticker} to price data for CLEIR tracking")
        
    except Exception as e:
        print(f"Error downloading benchmark data: {e}")
        benchmark_data = {}
    
    # Step 4: Run CVaR Backtest
    print("\n" + "="*50)
    print("STEP 4: CVAR INDEX BACKTEST")
    print("="*50)
    
    try:
        # Initialize backtester
        if optimization_config.sparsity_bound is not None:
            # CLEIR mode: pass asset tickers separately
            backtester = CVaRIndexBacktest(
                price_data, 
                optimization_config,
                asset_tickers=selected_universe  # Don't include benchmark in assets
            )
            print("Running CLEIR (CVaR-LASSO Enhanced Index Replication)")
        else:
            # Standard CVaR mode
            backtester = CVaRIndexBacktest(price_data, optimization_config)
            print("Running standard CVaR optimization")
        
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
        
    except Exception as e:
        print(f"Error in CVaR backtest: {e}")
        return
    
    # Step 5: Run Benchmark Backtests
    print("\n" + "="*50)
    print("STEP 5: BENCHMARK BACKTESTS")
    print("="*50)
    
    benchmark_results = {}
    
    # Equal Weight Benchmark
    try:
        print("Running equal-weight benchmark...")
        equal_weights = create_equal_weight_universe(price_data.tickers)
        ew_returns = run_simple_backtest(price_data, equal_weights, backtest_config)
        benchmark_results['Equal Weight'] = ew_returns
        
        ew_metrics = calculate_simple_metrics(ew_returns)
        print(f"Equal Weight - Annual Return: {ew_metrics['annual_return']:.2%}, "
              f"Volatility: {ew_metrics['annual_volatility']:.2%}, "
              f"Sharpe: {ew_metrics['sharpe_ratio']:.3f}")
        
    except Exception as e:
        print(f"Error in equal weight benchmark: {e}")
    
    # Market benchmarks (SPY, etc.)
    for ticker, prices in benchmark_data.items():
        try:
            aligned_prices = prices.reindex(cvar_results.returns.index)
            benchmark_returns = aligned_prices.pct_change().dropna()
            benchmark_results[ticker] = benchmark_returns
            
            bench_metrics = calculate_simple_metrics(benchmark_returns)
            print(f"{ticker} - Annual Return: {bench_metrics['annual_return']:.2%}, "
                  f"Volatility: {bench_metrics['annual_volatility']:.2%}, "
                  f"Sharpe: {bench_metrics['sharpe_ratio']:.3f}")
            
        except Exception as e:
            print(f"Error processing {ticker} benchmark: {e}")
    
    # Step 6: Performance Analysis
    print("\n" + "="*50)
    print("STEP 6: PERFORMANCE ANALYSIS")
    print("="*50)
    
    try:
        # Create comprehensive performance report
        performance_report = create_performance_report(
            cvar_results.returns,
            benchmark_results
        )
        
        # Create monthly returns table
        monthly_table = create_monthly_returns_table(cvar_results.returns)
        
        # Save results
        os.makedirs('results', exist_ok=True)
        
        # Save index values
        cvar_results.index_values.to_csv('results/baseline_cvar_index.csv')
        print("Saved CVaR index values to results/baseline_cvar_index.csv")
        
        # Save performance report
        save_performance_report(
            performance_report,
            monthly_table,
            'results/performance_metrics.xlsx'
        )
        
        # Save rebalancing events
        rebalancing_summary = backtester.get_rebalancing_summary()
        rebalancing_summary.to_csv('results/rebalancing_events.csv')
        print("Saved rebalancing events to results/rebalancing_events.csv")
        
        # Save weights history
        cvar_results.weights_history.to_csv('results/weights_history.csv')
        print("Saved weights history to results/weights_history.csv")
        
    except Exception as e:
        print(f"Error in performance analysis: {e}")
    
    # Step 7: Generate Plots
    print("\n" + "="*50)
    print("STEP 7: GENERATE PLOTS")
    print("="*50)
    
    try:
        # Performance comparison plot
        fig = plot_performance_comparison(
            cvar_results.returns,
            benchmark_results,
            "CVaR Index vs Benchmarks"
        )
        fig.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved performance comparison plot to results/performance_comparison.png")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    # Final Summary
    print("\n" + "="*60)
    print("BACKTEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print(f"Results saved in 'results/' directory:")
    print("- baseline_cvar_index.csv: Index values")
    print("- performance_metrics.xlsx: Comprehensive metrics")
    print("- rebalancing_events.csv: Rebalancing details")
    print("- weights_history.csv: Portfolio weights over time")
    print("- performance_comparison.png: Performance charts")
    
    print(f"\nCompleted at: {datetime.now()}")


def run_simple_backtest(price_data, weights, config):
    """
    Run simple buy-and-hold backtest with fixed weights.
    
    Args:
        price_data: PriceData object
        weights: Portfolio weights (constant)
        config: BacktestConfig
        
    Returns:
        Series of portfolio returns
    """
    # Filter data to backtest period
    backtest_data = price_data.slice_dates(config.start_date, config.end_date)
    returns_df = backtest_data.get_returns(method='simple')
    
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(weights, returns_df.values)
    
    return pd.Series(portfolio_returns, index=returns_df.index)


def calculate_simple_metrics(returns):
    """Calculate basic performance metrics."""
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    annual_return = (1 + total_return) ** (252 / n_periods) - 1
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio
    }


if __name__ == "__main__":
    main()
