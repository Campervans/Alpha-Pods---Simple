import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.schemas import OptimizationConfig, BacktestConfig, PriceData
from backtesting.engine import CVaRIndexBacktest


def test_transaction_cost_application():
    """Test that transaction costs are applied correctly to portfolio value."""
    # Create simple 2-asset universe with flat prices
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
    prices_df = pd.DataFrame({
        'A': 100.0 * np.ones(len(dates)),  # Flat price
        'B': 100.0 * np.ones(len(dates))   # Flat price
    }, index=dates)
    
    # Create PriceData object
    price_data = PriceData(
        prices=prices_df,
        tickers=['A', 'B'],
        start_date=dates[0],
        end_date=dates[-1]
    )
    
    # Configure quarterly rebalancing
    opt_config = OptimizationConfig(
        confidence_level=0.95,
        lookback_days=60,
        max_weight=1.0,
        min_weight=0.0
    )
    
    backtest_config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2020-12-31',
        rebalance_frequency='quarterly',
        transaction_cost_bps=10.0,
        initial_capital=100.0
    )
    
    # Run backtest
    backtester = CVaRIndexBacktest(price_data, opt_config)
    results = backtester.run_backtest(backtest_config)
    
    # With flat prices, value should only decrease by transaction costs
    final_value = results.index_values.iloc[-1]
    total_costs = sum(event.transaction_cost for event in results.rebalance_events)
    
    # Verify costs were applied
    assert total_costs > 0, "Should have non-zero transaction costs"
    assert final_value < 100.0, "Final value should be less than initial due to costs"
    
    # Verify the multiplicative application
    expected_value = 100.0
    for event in results.rebalance_events:
        expected_value *= (1 - event.transaction_cost)
    
    # Allow small tolerance for rounding
    assert abs(final_value - expected_value) < 0.01, \
        f"Expected value {expected_value:.4f}, got {final_value:.4f}"
    
    print(f"✓ Transaction costs correctly applied")
    print(f"  Initial value: 100.0")
    print(f"  Final value: {final_value:.4f}")
    print(f"  Total costs: {total_costs:.4%}")
    print(f"  Number of rebalances: {len(results.rebalance_events)}")


if __name__ == "__main__":
    test_transaction_cost_application() 