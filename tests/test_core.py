# tests for core utils

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.core import (
    get_trading_days, get_quarter_end_dates, calculate_log_returns,
    calculate_simple_returns, calculate_turnover, calculate_transaction_costs,
    annualize_return, annualize_volatility, calculate_sharpe_ratio,
    calculate_max_drawdown
)


class TestCoreFunctions:
    # core tests
    
    def test_calculate_simple_returns(self):
        # test simple returns
        prices = pd.Series([100, 105, 102, 108], name='price')
        returns = calculate_simple_returns(prices)
        
        # first should be nan
        assert pd.isna(returns.iloc[0])
        
        # second = 5%
        assert abs(returns.iloc[1] - 0.05) < 1e-6
        
        # third ~= -2.86%
        expected = (102 - 105) / 105
        assert abs(returns.iloc[2] - expected) < 1e-6
    
    def test_calculate_log_returns(self):
        """Test log returns calculation."""
        prices = pd.Series([100, 105, 102, 108])
        returns = calculate_log_returns(prices)
        
        # First return should be NaN
        assert pd.isna(returns.iloc[0])
        
        # Second return should be ln(105/100)
        expected = np.log(105/100)
        assert abs(returns.iloc[1] - expected) < 1e-6
    
    def test_calculate_turnover(self):
        """Test portfolio turnover calculation."""
        old_weights = np.array([0.5, 0.3, 0.2])
        new_weights = np.array([0.4, 0.4, 0.2])
        
        turnover = calculate_turnover(old_weights, new_weights)
        expected = abs(0.5 - 0.4) + abs(0.3 - 0.4) + abs(0.2 - 0.2)
        
        assert abs(turnover - expected) < 1e-6
        assert turnover == 0.2
    
    def test_calculate_transaction_costs(self):
        """Test transaction costs calculation."""
        turnover = 0.5  # 50% turnover
        cost_bps = 10.0  # 10 basis points per side
        
        cost = calculate_transaction_costs(turnover, cost_bps)
        expected = 0.5 * 10.0 / 10000.0  # 0.05%
        
        assert abs(cost - expected) < 1e-6
        assert cost == 0.0005
    
    def test_annualize_return(self):
        """Test return annualization."""
        # 10% return over 6 months (126 days)
        total_return = 0.10
        n_periods = 126
        
        annual_return = annualize_return(total_return, n_periods, 252)
        expected = (1.10) ** (252/126) - 1  # Should be about 21%
        
        assert abs(annual_return - expected) < 1e-6
    
    def test_annualize_volatility(self):
        """Test volatility annualization."""
        # Daily returns with 1% daily volatility
        daily_returns = np.random.normal(0, 0.01, 252)
        
        annual_vol = annualize_volatility(daily_returns, 252)
        # Should be approximately 16% (1% * sqrt(252))
        expected = 0.01 * np.sqrt(252)
        
        # Allow some tolerance due to randomness
        assert abs(annual_vol - expected) < 0.05
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Simple case: 10% annual return, 20% annual volatility
        returns = pd.Series([0.10/252] * 252)  # Daily returns for 10% annual
        returns += np.random.normal(0, 0.20/np.sqrt(252), 252)  # Add volatility
        
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
        
        # Should be approximately (0.10 - 0.02) / 0.20 = 0.4
        # Allow wide tolerance due to randomness
        assert abs(sharpe - 0.4) < 1.0
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create returns with known drawdown
        returns = pd.Series([0.1, -0.2, 0.05, -0.1, 0.15])
        
        max_dd = calculate_max_drawdown(returns)
        
        # Should be positive value
        assert max_dd >= 0
        
        # Calculate expected manually
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        expected_max_dd = abs(drawdown.min())
        
        assert abs(max_dd - expected_max_dd) < 1e-6
    
    def test_quarter_end_dates(self):
        """Test quarter end date calculation."""
        # Create a year of business days
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        quarter_ends = get_quarter_end_dates(dates)
        
        # Should have 4 quarter ends
        assert len(quarter_ends) == 4
        
        # Check that they are indeed quarter ends
        for date in quarter_ends:
            assert date.month in [3, 6, 9, 12]
    
    def test_empty_series_handling(self):
        """Test functions handle empty series gracefully."""
        empty_series = pd.Series([], dtype=float)
        
        # Should not raise errors
        returns = calculate_simple_returns(empty_series)
        assert len(returns) == 0
        
        log_returns = calculate_log_returns(empty_series)
        assert len(log_returns) == 0
        
        max_dd = calculate_max_drawdown(empty_series)
        assert max_dd == 0.0
    
    def test_turnover_edge_cases(self):
        """Test turnover calculation edge cases."""
        # Identical weights
        weights = np.array([0.5, 0.3, 0.2])
        turnover = calculate_turnover(weights, weights)
        assert turnover == 0.0
        
        # Complete rebalancing
        old_weights = np.array([1.0, 0.0, 0.0])
        new_weights = np.array([0.0, 0.0, 1.0])
        turnover = calculate_turnover(old_weights, new_weights)
        assert turnover == 2.0  # Maximum possible turnover
    
    def test_weight_validation(self):
        """Test weight arrays validation."""
        old_weights = np.array([0.5, 0.5])
        new_weights = np.array([0.4, 0.4, 0.2])  # Different length
        
        with pytest.raises(ValueError):
            calculate_turnover(old_weights, new_weights)


if __name__ == "__main__":
    pytest.main([__file__])
