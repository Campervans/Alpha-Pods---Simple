# Transaction Cost Implementation Remediation

## Audit Summary

### Current Implementation Issues

The current implementation in `src/backtesting/engine.py` (lines 107-113) has several problems:

1. **Incorrect Application Point**: Transaction costs are being applied to the return calculation rather than directly to the portfolio value
2. **Timing Issue**: The cost is effectively being applied to the previous period's value (`index_values[-2]`) rather than the current value
3. **Mathematical Error**: The formula `index_values[-1] = index_values[-2] * (1 + net_return)` where `net_return = current_return - transaction_cost` creates a compound effect that slightly distorts the cost impact

### Required Implementation

Per specification:
- Transaction costs should be **10 basis points per side**
- Applied **once** at each quarterly rebalance
- Formula: `IndexValue_post_TC = IndexValue_pre_TC × (1 - TC)`
- Where `TC = turnover × 0.001` (10 bps converted to decimal)

## Step-by-Step Remediation Plan

### Step 1: Update the Backtesting Engine
**File**: `src/backtesting/engine.py`

1.1. Remove the unused import:
   - Delete `apply_transaction_costs_to_returns` from the import statement on line 14

1.2. Replace the transaction cost application logic (lines 107-113):
   ```python
   # DELETE these lines:
   # if len(index_values) > 1:
   #     current_return = (index_values[-1] / index_values[-2]) - 1
   #     net_return = apply_transaction_costs_to_returns(
   #         current_return, rebalance_event.transaction_cost
   #     )
   #     index_values[-1] = index_values[-2] * (1 + net_return)
   
   # REPLACE with:
   # Apply transaction costs directly to portfolio value
   if len(index_values) > 0:
       index_values[-1] *= (1 - rebalance_event.transaction_cost)
   ```

### Step 2: Handle the Deprecated Function
**File**: `src/backtesting/rebalancer.py`

2.1. Add deprecation notice to `apply_transaction_costs_to_returns` (line 140):
   ```python
   def apply_transaction_costs_to_returns(portfolio_return: float,
                                         transaction_cost: float) -> float:
       """
       [DEPRECATED] Apply transaction costs to portfolio return.
       
       This function is deprecated. Transaction costs should be applied
       directly to portfolio value using: value_post = value_pre * (1 - cost)
       
       Kept for backward compatibility only.
       """
       return portfolio_return - transaction_cost
   ```

### Step 3: Verify Transaction Cost Calculation
**File**: `src/utils/core.py` (No changes needed)

3.1. Confirm `calculate_transaction_costs` function is correct:
   - Formula: `turnover * cost_per_side_bps / 10000.0`
   - This correctly converts basis points to decimal percentage

### Step 4: Add Integration Test
**File**: `tests/integration/test_transaction_costs.py` (New file)

```python
import pytest
import numpy as np
import pandas as pd
from src.utils.schemas import OptimizationConfig, BacktestConfig
from src.backtesting.engine import CVaRIndexBacktest
from src.market_data.universe import create_mock_price_data

def test_transaction_cost_application():
    """Test that transaction costs are applied correctly to portfolio value."""
    # Create simple 2-asset universe with predictable returns
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
    prices = pd.DataFrame({
        'A': 100 * np.ones(len(dates)),  # Flat price
        'B': 100 * np.ones(len(dates))   # Flat price
    }, index=dates)
    
    price_data = create_mock_price_data(prices)
    
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
    
    # With flat prices and rebalancing, value should only decrease by transaction costs
    # Expected: 4 rebalances in 2020, each with some turnover
    final_value = results.index_values.iloc[-1]
    total_costs = sum(event.transaction_cost for event in results.rebalance_events)
    
    # Verify costs were applied
    assert total_costs > 0
    assert final_value < 100.0  # Should be less than initial due to costs
    
    # Verify the multiplicative application
    expected_value = 100.0
    for event in results.rebalance_events:
        expected_value *= (1 - event.transaction_cost)
    
    # Allow small tolerance for rounding
    assert abs(final_value - expected_value) < 0.01
```

### Step 5: Update Unit Tests
**File**: `tests/test_core.py` (No changes needed)

5.1. Existing test `test_calculate_transaction_costs` is still valid
   - It correctly tests the turnover-to-cost conversion

### Step 6: Documentation Update
**File**: `src/backtesting/engine.py`

6.1. Add comment above the transaction cost application:
   ```python
   # Apply transaction costs directly to portfolio value
   # Cost formula: IndexValue_post = IndexValue_pre * (1 - transaction_cost)
   # Where transaction_cost = turnover * 10bps (0.001)
   ```

## Validation Steps

1. **Run existing tests**: `pytest tests/`
2. **Run new integration test**: `pytest tests/integration/test_transaction_costs.py`
3. **Run full backtest**: `python scripts/run_baseline_backtest.py`
4. **Compare results**:
   - Total transaction costs should remain approximately the same (±0.01%)
   - The difference comes from removing the compound effect error
   - Expected total costs over 15 years: ~0.20% (as shown in current results)

## Impact Analysis

- **Minimal Impact**: Since transaction costs are small (0.20% total over 15 years), the correction will have minimal impact on final results
- **More Accurate**: The new implementation correctly models how transaction costs work in practice
- **Cleaner Code**: Removes unnecessary complexity and potential confusion

## Notes

- The "10 bps per side" specification means we charge 10 basis points for the total turnover
- Turnover already accounts for both buying and selling (sum of absolute weight changes)
- The current `calculate_transaction_costs` function correctly implements this 