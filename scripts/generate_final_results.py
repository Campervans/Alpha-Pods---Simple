"""
Generate final results for Task A - Baseline CVaR Index.

This script produces the required deliverables:
1. Daily index values (CSV)
2. Performance metrics table
3. Performance comparison plot
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.schemas import UniverseConfig, OptimizationConfig, BacktestConfig
from src.market_data.downloader import create_sp100_list, download_universe, download_benchmark_data
from src.market_data.universe import select_liquid_universe
from src.backtesting.engine import CVaRIndexBacktest
from src.backtesting.metrics import calculate_cvar, calculate_max_drawdown, calculate_turnover_stats
from src.optimization.risk_models import calculate_portfolio_returns

# Suppress warnings
warnings.filterwarnings('ignore')


def calculate_performance_metrics(returns, index_values, rebalance_events=None):
    """Calculate all required performance metrics."""
    # Annual return
    total_return = (index_values.iloc[-1] / index_values.iloc[0]) - 1
    n_years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    
    # Annual volatility
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # 95% CVaR
    cvar_95 = calculate_cvar(returns.values, 0.95)
    
    # Maximum drawdown
    max_drawdown = calculate_max_drawdown(index_values.values)
    
    # Turnover (if rebalance events provided)
    avg_turnover = 0
    total_costs = 0
    if rebalance_events:
        turnovers = [event.turnover for event in rebalance_events]
        costs = [event.transaction_cost for event in rebalance_events]
        avg_turnover = np.mean(turnovers) if turnovers else 0
        total_costs = np.sum(costs) if costs else 0
    
    return {
        'Annual_Return_Pct': annual_return * 100,
        'Annual_Volatility_Pct': annual_volatility * 100,
        'Sharpe_Ratio': sharpe_ratio,
        'CVaR_95_Pct': cvar_95 * 100,
        'Max_Drawdown_Pct': max_drawdown * 100,
        'Total_Return_Pct': total_return * 100,
        'Avg_Turnover_Pct': avg_turnover * 100,
        'Total_Transaction_Costs_Pct': total_costs * 100
    }


def main():
    """Main execution function."""
    print("=" * 60)
    print("GENERATING FINAL RESULTS FOR TASK A")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Configuration for standard CVaR (not CLEIR)
    universe_config = UniverseConfig(
        n_stocks=60,
        lookback_days=126,
        metric="dollar_volume",
        min_price=5.0,
        min_trading_days=100
    )
    
    optimization_config = OptimizationConfig(
        confidence_level=0.95,
        lookback_days=252,
        max_weight=0.05,
        min_weight=0.0,
        solver="SCS",
        solver_options={'max_iters': 10000}
    )
    
    backtest_config = BacktestConfig(
        start_date="2010-01-01",
        end_date="2024-12-31",
        rebalance_frequency="quarterly",
        transaction_cost_bps=10.0,
        initial_capital=100.0,
        benchmark_tickers=["SPY"]
    )
    
    # Step 1: Universe Selection
    print("\nStep 1: Selecting investment universe...")
    
    # Get S&P 100 candidates
    sp100_tickers = create_sp100_list()
    
    # Select 60 most liquid stocks
    selected_universe = select_liquid_universe(
        sp100_tickers,
        universe_config,
        start_date="2023-01-01",
        end_date="2024-12-31"
    )
    
    print(f"Selected {len(selected_universe)} stocks from S&P 100")
    
    # Step 2: Download Data
    print("\nStep 2: Downloading price data...")
    
    # Download universe data
    price_data = download_universe(
        selected_universe,
        start="2009-07-01",
        end="2024-12-31",
        min_data_points=100
    )
    
    print(f"Downloaded data for {price_data.n_assets} assets")
    print(f"Date range: {price_data.start_date.date()} to {price_data.end_date.date()}")
    
    # Download benchmark data
    benchmark_data = download_benchmark_data(
        ["SPY"],
        "2009-07-01",
        "2024-12-31"
    )
    
    # Step 3: Run CVaR Backtest
    print("\nStep 3: Running CVaR index backtest...")
    
    # Initialize backtester
    backtester = CVaRIndexBacktest(price_data, optimization_config)
    
    # Run backtest
    cvar_results = backtester.run_backtest(backtest_config)
    
    print("\nCVaR Index Performance:")
    print(f"Total Return: {cvar_results.total_return:.2%}")
    print(f"Annual Return: {cvar_results.annual_return:.2%}")
    print(f"Sharpe Ratio: {cvar_results.sharpe_ratio:.3f}")
    
    # Step 4: Create Benchmarks
    print("\nStep 4: Creating benchmark portfolios...")
    
    # Equal weight portfolio
    n_assets = len(selected_universe)
    equal_weights = np.ones(n_assets) / n_assets
    
    # Filter data to backtest period
    backtest_data = price_data.slice_dates(backtest_config.start_date, backtest_config.end_date)
    returns_df = backtest_data.get_returns(method='simple')
    
    # Calculate equal weight returns
    ew_returns = calculate_portfolio_returns(equal_weights, returns_df.values)
    ew_returns = pd.Series(ew_returns, index=returns_df.index)
    ew_index = (1 + ew_returns).cumprod() * 100
    ew_index = pd.concat([pd.Series([100.0], index=[backtest_data.dates[0]]), ew_index])
    
    # Get SPY returns
    spy_prices = benchmark_data['SPY'].reindex(ew_index.index)
    spy_returns = spy_prices.pct_change().dropna()
    spy_index = spy_prices / spy_prices.iloc[0] * 100
    
    # Step 5: Generate Deliverables
    print("\nStep 5: Generating deliverables...")
    
    os.makedirs('results', exist_ok=True)
    
    # Deliverable 1: Daily Index Values
    daily_values = pd.DataFrame({
        'Date': cvar_results.index_values.index,
        'Index_Value': cvar_results.index_values.values,
        'Daily_Return': cvar_results.returns.reindex(cvar_results.index_values.index).fillna(0).values,
        'Cumulative_Return': (cvar_results.index_values.values / 100 - 1) * 100
    })
    daily_values.to_csv('results/daily_index_values.csv', index=False)
    print("✓ Saved daily_index_values.csv")
    
    # Deliverable 2: Performance Metrics Table
    metrics_data = []
    
    # CVaR Index metrics
    cvar_metrics = calculate_performance_metrics(
        cvar_results.returns,
        cvar_results.index_values,
        cvar_results.rebalance_events
    )
    cvar_metrics['Strategy'] = 'CVaR_Index'
    metrics_data.append(cvar_metrics)
    
    # Equal Weight metrics
    ew_metrics = calculate_performance_metrics(ew_returns, ew_index)
    ew_metrics['Strategy'] = 'Equal_Weight'
    metrics_data.append(ew_metrics)
    
    # SPY metrics
    spy_metrics = calculate_performance_metrics(spy_returns, spy_index)
    spy_metrics['Strategy'] = 'Cap_Weight_SPY'
    metrics_data.append(spy_metrics)
    
    # Create DataFrame and reorder columns
    metrics_df = pd.DataFrame(metrics_data)
    column_order = ['Strategy', 'Annual_Return_Pct', 'Annual_Volatility_Pct', 
                    'Sharpe_Ratio', 'CVaR_95_Pct', 'Max_Drawdown_Pct', 
                    'Total_Return_Pct', 'Avg_Turnover_Pct', 'Total_Transaction_Costs_Pct']
    metrics_df = metrics_df[column_order]
    
    # Round appropriately
    for col in metrics_df.columns:
        if col == 'Strategy':
            continue
        elif col == 'Sharpe_Ratio':
            metrics_df[col] = metrics_df[col].round(2)
        else:
            metrics_df[col] = metrics_df[col].round(2)
    
    metrics_df.to_csv('results/performance_summary.csv', index=False)
    print("✓ Saved performance_summary.csv")
    
    # Deliverable 3: Performance Comparison Plot
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative returns
    plt.plot(cvar_results.index_values.index, cvar_results.index_values.values, 
             label='CVaR Index', linewidth=2, color='blue')
    plt.plot(ew_index.index, ew_index.values, 
             label='Equal Weight', linewidth=2, color='green', linestyle='--')
    plt.plot(spy_index.index, spy_index.values, 
             label='S&P 500 (SPY)', linewidth=2, color='red', linestyle=':')
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Index Value (Base = 100)', fontsize=12)
    plt.title('CVaR Index Performance Comparison\n2010-2024', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better show relative performance
    
    # Add performance annotations
    for strategy, index, y_pos in [
        ('CVaR Index', cvar_results.index_values, 0.85),
        ('Equal Weight', ew_index, 0.75),
        ('S&P 500', spy_index, 0.65)
    ]:
        final_value = index.iloc[-1]
        total_return = (final_value / 100 - 1) * 100
        plt.text(0.02, y_pos, f'{strategy}: {total_return:.0f}% total return',
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/index_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved index_performance_analysis.png")
    
    # Also save the data used for plotting
    plot_data = pd.DataFrame({
        'Date': cvar_results.index_values.index,
        'CVaR_Index': cvar_results.index_values.values,
        'Equal_Weight': ew_index.reindex(cvar_results.index_values.index).values,
        'SPY': spy_index.reindex(cvar_results.index_values.index).values
    })
    plot_data.to_csv('results/performance_comparison_data.csv', index=False)
    
    # Summary Report
    print("\n" + "=" * 60)
    print("TASK A DELIVERABLES COMPLETE")
    print("=" * 60)
    
    print("\nDeliverables generated:")
    print("1. ✓ daily_index_values.csv - Daily index values from 2010-2024")
    print("2. ✓ performance_summary.csv - Performance metrics comparison table")
    print("3. ✓ index_performance_analysis.png - Performance comparison plot")
    
    print("\nKey Results:")
    print(f"- CVaR Index Total Return: {cvar_metrics['Total_Return_Pct']:.1f}%")
    print(f"- CVaR Index Sharpe Ratio: {cvar_metrics['Sharpe_Ratio']:.2f}")
    print(f"- CVaR Index Max Drawdown: {cvar_metrics['Max_Drawdown_Pct']:.1f}%")
    
    print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main()