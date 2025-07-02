"""
Core utils for CVaR index construction.

Foundational building blocks for the CVaR index system.
Each function is simple, testable, and has a single responsibility.
"""

import numpy as np
import pandas as pd
from typing import Union
import pandas_market_calendars as mcal


def get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Get NYSE trading calendar.
    
    Args:
        start_date: start date 'YYYY-MM-DD'
        end_date: end date 'YYYY-MM-DD'
        
    Returns:
        DatetimeIndex of trading days
        
    Example:
        >>> days = get_trading_days('2020-01-01', '2020-01-10')
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
        DatetimeIndex of quarter-end dates
        
    Example:
        >>> trading_days = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        >>> quarter_ends = get_quarter_end_dates(trading_days)
    """
    if len(dates) == 0:
        return pd.DatetimeIndex([])
    
    # convert to DataFrame for easier grouping
    df = pd.DataFrame({'date': dates})
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    # group by year-quarter and take the last date
    quarter_ends = df.groupby(['year', 'quarter'])['date'].max()
    
    return pd.DatetimeIndex(quarter_ends.values).sort_values()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Convert prices to log returns.
    
    Args:
        prices: series of asset prices
        
    Returns:
        series of log returns (first value is NaN)
        
    Example:
        >>> prices = pd.Series([100, 105, 102, 108])
        >>> returns = calculate_log_returns(prices)
    """
    if len(prices) < 2:
        return pd.Series(dtype=float, index=prices.index)
    
    return np.log(prices / prices.shift(1))


def calculate_simple_returns(prices: pd.Series) -> pd.Series:
    """
    Convert prices to simple returns.
    
    Args:
        prices: series of asset prices
        
    Returns:
        series of simple returns (first value is NaN)
        
    Example:
        >>> prices = pd.Series([100, 105, 102, 108])
        >>> returns = calculate_simple_returns(prices)
    """
    if len(prices) < 2:
        return pd.Series(dtype=float, index=prices.index)
    
    return prices.pct_change()


def calculate_turnover(weights_old: np.ndarray, weights_new: np.ndarray) -> float:
    """
    Calculate portfolio turnover.
    
    Turnover is sum of absolute differences in weights,
    representing total trading required.
    
    Args:
        weights_old: array of old weights
        weights_new: array of new weights
        
    Returns:
        turnover as a float (0 to 2)
        
    Example:
        >>> old = np.array([0.5, 0.3, 0.2])
        >>> new = np.array([0.4, 0.4, 0.2])
        >>> turnover = calculate_turnover(old, new)
    """
    if len(weights_old) != len(weights_new):
        raise ValueError("Weight arrays must have same length")
    
    return np.abs(weights_new - weights_old).sum()


def calculate_transaction_costs(turnover: float, cost_per_side_bps: float = 10.0) -> float:
    """
    Calculate transaction costs from turnover.
    
    Args:
        turnover: portfolio turnover
        cost_per_side_bps: transaction cost in bps per side
        
    Returns:
        total transaction cost as a percentage
        
    Example:
        >>> costs = calculate_transaction_costs(0.5, 10.0)
    """
    return turnover * cost_per_side_bps / 10000.0


def annualize_return(total_return: float, num_periods: int, periods_per_year: int = 252) -> float:
    """
    Annualize a total return.
    
    Args:
        total_return: total return over the period
        num_periods: number of periods
        periods_per_year: number of periods in a year (default 252 for daily)
        
    Returns:
        annualized return
        
    Example:
        >>> ann_ret = annualize_return(0.10, 126, 252)
    """
    if num_periods <= 0:
        raise ValueError("Number of periods must be positive")
    
    return (1 + total_return) ** (periods_per_year / num_periods) - 1


def annualize_volatility(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
    """
    Annualize volatility from periodic returns.
    
    Args:
        returns: series or array of periodic returns
        periods_per_year: number of periods in a year (default 252 for daily)
        
    Returns:
        annualized volatility
        
    Example:
        >>> daily_returns = np.random.normal(0, 0.01, 252)
        >>> ann_vol = annualize_volatility(daily_returns, 252)
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
        returns: series or array of periodic returns
        risk_free_rate: annualized risk-free rate
        periods_per_year: number of periods in a year
        
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
    Calculate max drawdown from returns.
    
    Args:
        returns: series or array of periodic returns
        
    Returns:
        max drawdown as a positive percentage
        
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
    
    # calculate running max
    running_max = np.maximum.accumulate(cumulative)
    
    # calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    return abs(np.min(drawdown))


def rebalance_weights_to_target(current_weights: np.ndarray, 
                               target_weights: np.ndarray,
                               max_turnover: float = None) -> np.ndarray:
    """
    Rebalance weights toward target, with optional turnover constraint.
    
    Args:
        current_weights: current portfolio weights
        target_weights: target portfolio weights
        max_turnover: max allowed turnover (None for no constraint)
        
    Returns:
        new weights after rebalancing
        
    Example:
        >>> current = np.array([0.6, 0.4])
        >>> target = np.array([0.5, 0.5])
        >>> new_weights = rebalance_weights_to_target(current, target, 0.1)
    """
    if len(current_weights) != len(target_weights):
        raise ValueError("Weight arrays must have same length")
    
    if max_turnover is None:
        return target_weights.copy()
    
    # calculate required turnover for full rebalancing
    full_turnover = calculate_turnover(current_weights, target_weights)
    
    if full_turnover <= max_turnover:
        return target_weights.copy()
    
    # scale rebalancing to meet turnover constraint
    scale_factor = max_turnover / full_turnover
    weight_change = (target_weights - current_weights) * scale_factor
    
    return current_weights + weight_change 