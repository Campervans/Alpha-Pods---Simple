"""
Core utility functions for CVaR index construction.

This module provides foundational building blocks that are used throughout
the CVaR index system. Each function is designed to be simple, testable,
and focused on a single responsibility.
"""

import numpy as np
import pandas as pd
from typing import Union
import pandas_market_calendars as mcal


def get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Get NYSE trading calendar between dates.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DatetimeIndex of trading days
        
    Example:
        >>> days = get_trading_days('2020-01-01', '2020-01-10')
        >>> len(days)  # Should be around 7-8 trading days
    """
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    return schedule.index


def get_quarter_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Filter to last trading day of each quarter.
    
    Args:
        dates: DatetimeIndex of trading days
        
    Returns:
        DatetimeIndex containing only quarter-end dates
        
    Example:
        >>> trading_days = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        >>> quarter_ends = get_quarter_end_dates(trading_days)
        >>> len(quarter_ends)  # Should be 4 for year 2020
    """
    if len(dates) == 0:
        return pd.DatetimeIndex([])
    
    # Convert to DataFrame for easier grouping
    df = pd.DataFrame({'date': dates})
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    # Group by year-quarter and take the last date in each group
    quarter_ends = df.groupby(['year', 'quarter'])['date'].max()
    
    return pd.DatetimeIndex(quarter_ends.values).sort_values()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Convert prices to log returns.
    
    Args:
        prices: Series of asset prices
        
    Returns:
        Series of log returns (first value will be NaN)
        
    Example:
        >>> prices = pd.Series([100, 105, 102, 108])
        >>> returns = calculate_log_returns(prices)
        >>> returns.iloc[1]  # Should be approximately log(105/100) ≈ 0.0488
    """
    if len(prices) < 2:
        return pd.Series(dtype=float, index=prices.index)
    
    return np.log(prices / prices.shift(1))


def calculate_simple_returns(prices: pd.Series) -> pd.Series:
    """
    Convert prices to simple returns.
    
    Args:
        prices: Series of asset prices
        
    Returns:
        Series of simple returns (first value will be NaN)
        
    Example:
        >>> prices = pd.Series([100, 105, 102, 108])
        >>> returns = calculate_simple_returns(prices)
        >>> returns.iloc[1]  # Should be 0.05 (5% return)
    """
    if len(prices) < 2:
        return pd.Series(dtype=float, index=prices.index)
    
    return prices.pct_change()


def calculate_turnover(weights_old: np.ndarray, weights_new: np.ndarray) -> float:
    """
    Calculate portfolio turnover.
    
    Turnover is defined as the sum of absolute differences in weights,
    which represents the total amount of trading required.
    
    Args:
        weights_old: Array of old portfolio weights
        weights_new: Array of new portfolio weights
        
    Returns:
        Turnover as a float between 0 and 2
        
    Example:
        >>> old = np.array([0.5, 0.3, 0.2])
        >>> new = np.array([0.4, 0.4, 0.2])
        >>> turnover = calculate_turnover(old, new)
        >>> turnover  # Should be 0.2 (|0.5-0.4| + |0.3-0.4| + |0.2-0.2|)
    """
    if len(weights_old) != len(weights_new):
        raise ValueError("Weight arrays must have the same length")
    
    return np.abs(weights_new - weights_old).sum()


def calculate_transaction_costs(turnover: float, cost_per_side_bps: float = 10.0) -> float:
    """
    Calculate transaction costs based on turnover.
    
    Args:
        turnover: Portfolio turnover (sum of absolute weight changes)
        cost_per_side_bps: Transaction cost in basis points per side
        
    Returns:
        Total transaction cost as a percentage
        
    Example:
        >>> costs = calculate_transaction_costs(0.5, 10.0)
        >>> costs  # Should be 0.001 (0.1% for 50% turnover at 10bps/side)
    """
    return turnover * cost_per_side_bps / 10000.0


def annualize_return(total_return: float, num_periods: int, periods_per_year: int = 252) -> float:
    """
    Annualize a total return.
    
    Args:
        total_return: Total return over the period
        num_periods: Number of periods
        periods_per_year: Number of periods in a year (default 252 for daily)
        
    Returns:
        Annualized return
        
    Example:
        >>> ann_ret = annualize_return(0.10, 126, 252)  # 10% over 6 months
        >>> ann_ret  # Should be approximately 0.21 (21% annualized)
    """
    if num_periods <= 0:
        raise ValueError("Number of periods must be positive")
    
    return (1 + total_return) ** (periods_per_year / num_periods) - 1


def annualize_volatility(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
    """
    Annualize volatility from periodic returns.
    
    Args:
        returns: Series or array of periodic returns
        periods_per_year: Number of periods in a year (default 252 for daily)
        
    Returns:
        Annualized volatility
        
    Example:
        >>> daily_returns = np.random.normal(0, 0.01, 252)  # 1% daily vol
        >>> ann_vol = annualize_volatility(daily_returns, 252)
        >>> ann_vol  # Should be approximately 0.16 (16% annualized)
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    
    if len(returns) == 0:
        return 0.0
    
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray], 
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio from periodic returns.
    
    Args:
        returns: Series or array of periodic returns
        risk_free_rate: Annualized risk-free rate (default 0.0)
        periods_per_year: Number of periods in a year (default 252 for daily)
        
    Returns:
        Sharpe ratio
        
    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        >>> sharpe = calculate_sharpe_ratio(returns, 0.02, 252)
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns) * periods_per_year
    volatility = annualize_volatility(returns, periods_per_year)
    
    if volatility == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / volatility


def calculate_max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate maximum drawdown from returns.
    
    Args:
        returns: Series or array of periodic returns
        
    Returns:
        Maximum drawdown as a positive percentage
        
    Example:
        >>> returns = pd.Series([0.1, -0.2, 0.05, -0.1])
        >>> max_dd = calculate_max_drawdown(returns)
    """
    if isinstance(returns, pd.Series):
        cumulative = (1 + returns).cumprod()
    else:
        cumulative = np.cumprod(1 + returns)
    
    if len(cumulative) == 0:
        return 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    return abs(np.min(drawdown))


def rebalance_weights_to_target(current_weights: np.ndarray, 
                               target_weights: np.ndarray,
                               max_turnover: float = None) -> np.ndarray:
    """
    Rebalance weights toward target, optionally with turnover constraint.
    
    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        max_turnover: Maximum allowed turnover (None for no constraint)
        
    Returns:
        New weights after rebalancing
        
    Example:
        >>> current = np.array([0.6, 0.4])
        >>> target = np.array([0.5, 0.5])
        >>> new_weights = rebalance_weights_to_target(current, target, 0.1)
    """
    if len(current_weights) != len(target_weights):
        raise ValueError("Weight arrays must have the same length")
    
    if max_turnover is None:
        return target_weights.copy()
    
    # Calculate required turnover for full rebalancing
    full_turnover = calculate_turnover(current_weights, target_weights)
    
    if full_turnover <= max_turnover:
        return target_weights.copy()
    
    # Scale the rebalancing to meet turnover constraint
    scale_factor = max_turnover / full_turnover
    weight_change = (target_weights - current_weights) * scale_factor
    
    return current_weights + weight_change 