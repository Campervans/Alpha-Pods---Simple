"""
Risk models and calculations for CVaR optimization.

This module provides functions for calculating portfolio returns, risk metrics,
and other components needed for CVaR-based portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from scipy import stats


def calculate_portfolio_returns(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """
    Calculate portfolio returns given weights and asset returns.
    
    Args:
        weights: Array of portfolio weights (length n_assets)
        returns: Array of asset returns (shape: n_periods x n_assets)
        
    Returns:
        Array of portfolio returns (length n_periods)
        
    Example:
        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> returns = np.random.normal(0, 0.01, (100, 3))
        >>> port_returns = calculate_portfolio_returns(weights, returns)
    """
    if len(weights.shape) == 1:
        weights = weights.reshape(-1, 1)
    
    if returns.shape[1] != weights.shape[0]:
        raise ValueError(
            f"Dimension mismatch: returns has {returns.shape[1]} assets, "
            f"weights has {weights.shape[0]} assets"
        )
    
    # Calculate portfolio returns: R_p = sum(w_i * R_i)
    portfolio_returns = np.dot(returns, weights).flatten()
    
    return portfolio_returns


def calculate_historical_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate historical CVaR at given confidence level.
    
    CVaR is the expected value of losses beyond the VaR threshold.
    
    Args:
        returns: Array of portfolio or asset returns
        confidence: Confidence level (e.g., 0.95 for 95% CVaR)
        
    Returns:
        CVaR value (positive number representing expected loss)
        
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 1000)
        >>> cvar_95 = calculate_historical_cvar(returns, 0.95)
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate VaR threshold
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    
    # Calculate CVaR as mean of losses beyond VaR
    tail_losses = returns[returns <= var_threshold]
    
    if len(tail_losses) == 0:
        return abs(var_threshold)
    
    cvar = np.mean(tail_losses)
    
    # Return as positive value (loss)
    return abs(cvar)


def calculate_historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate historical Value at Risk (VaR) at given confidence level.
    
    Args:
        returns: Array of portfolio or asset returns
        confidence: Confidence level (e.g., 0.95 for 95% VaR)
        
    Returns:
        VaR value (positive number representing potential loss)
        
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 1000)
        >>> var_95 = calculate_historical_var(returns, 0.95)
    """
    if len(returns) == 0:
        return 0.0
    
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    
    # Return as positive value (loss)
    return abs(var_threshold)


def calculate_parametric_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate parametric CVaR assuming normal distribution.
    
    Args:
        returns: Array of portfolio or asset returns
        confidence: Confidence level (e.g., 0.95 for 95% CVaR)
        
    Returns:
        Parametric CVaR value
        
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 1000)
        >>> cvar_95 = calculate_parametric_cvar(returns, 0.95)
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    # Calculate parametric CVaR for normal distribution
    alpha = 1 - confidence
    z_alpha = stats.norm.ppf(alpha)
    
    # CVaR formula for normal distribution
    cvar = mean_return - std_return * stats.norm.pdf(z_alpha) / alpha
    
    # Return as positive value (loss)
    return abs(cvar)


def calculate_expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate both VaR and Expected Shortfall (CVaR).
    
    Args:
        returns: Array of portfolio or asset returns
        confidence: Confidence level
        
    Returns:
        Tuple of (VaR, CVaR)
        
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 1000)
        >>> var, cvar = calculate_expected_shortfall(returns, 0.95)
    """
    var = calculate_historical_var(returns, confidence)
    cvar = calculate_historical_cvar(returns, confidence)
    
    return var, cvar


def calculate_portfolio_volatility(weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
    """
    Calculate portfolio volatility using covariance matrix.
    
    Args:
        weights: Array of portfolio weights
        covariance_matrix: Asset covariance matrix
        
    Returns:
        Portfolio volatility (standard deviation)
        
    Example:
        >>> weights = np.array([0.6, 0.4])
        >>> cov_matrix = np.array([[0.0004, 0.0002], [0.0002, 0.0009]])
        >>> vol = calculate_portfolio_volatility(weights, cov_matrix)
    """
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    return np.sqrt(portfolio_variance)


def calculate_risk_contribution(weights: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate risk contribution of each asset to portfolio risk.
    
    Args:
        weights: Array of portfolio weights
        covariance_matrix: Asset covariance matrix
        
    Returns:
        Array of risk contributions (sum to 1.0)
        
    Example:
        >>> weights = np.array([0.6, 0.4])
        >>> cov_matrix = np.array([[0.0004, 0.0002], [0.0002, 0.0009]])
        >>> risk_contrib = calculate_risk_contribution(weights, cov_matrix)
    """
    portfolio_vol = calculate_portfolio_volatility(weights, covariance_matrix)
    
    if portfolio_vol == 0:
        return np.zeros_like(weights)
    
    # Marginal risk contribution
    marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
    
    # Risk contribution = weight * marginal contribution / portfolio volatility
    risk_contrib = weights * marginal_contrib / portfolio_vol
    
    return risk_contrib


def estimate_covariance_matrix(returns: pd.DataFrame, method: str = "sample") -> np.ndarray:
    """
    Estimate covariance matrix from historical returns.
    
    Args:
        returns: DataFrame of asset returns
        method: Estimation method ("sample", "shrinkage", "exponential")
        
    Returns:
        Covariance matrix
        
    Example:
        >>> returns_df = pd.DataFrame(np.random.normal(0, 0.01, (100, 3)))
        >>> cov_matrix = estimate_covariance_matrix(returns_df)
    """
    if method == "sample":
        return returns.cov().values
    
    elif method == "shrinkage":
        # Simple shrinkage estimator
        sample_cov = returns.cov().values
        n_assets = sample_cov.shape[0]
        
        # Shrinkage target: diagonal matrix with average variance
        avg_var = np.trace(sample_cov) / n_assets
        target = np.eye(n_assets) * avg_var
        
        # Shrinkage intensity (simple rule)
        shrinkage = 0.2
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target
        
        return shrunk_cov
    
    elif method == "exponential":
        # Exponentially weighted covariance
        ewm_cov = returns.ewm(halflife=60).cov().iloc[-len(returns.columns):]
        return ewm_cov.values
    
    else:
        raise ValueError(f"Unknown covariance estimation method: {method}")


def calculate_tracking_error(portfolio_returns: np.ndarray, 
                           benchmark_returns: np.ndarray) -> float:
    """
    Calculate tracking error between portfolio and benchmark.
    
    Args:
        portfolio_returns: Array of portfolio returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        Annualized tracking error
        
    Example:
        >>> port_ret = np.random.normal(0.001, 0.02, 252)
        >>> bench_ret = np.random.normal(0.0008, 0.015, 252)
        >>> te = calculate_tracking_error(port_ret, bench_ret)
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Portfolio and benchmark returns must have same length")
    
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(252)
    
    return tracking_error


def calculate_information_ratio(portfolio_returns: np.ndarray,
                               benchmark_returns: np.ndarray) -> float:
    """
    Calculate information ratio (excess return / tracking error).
    
    Args:
        portfolio_returns: Array of portfolio returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        Information ratio
        
    Example:
        >>> port_ret = np.random.normal(0.001, 0.02, 252)
        >>> bench_ret = np.random.normal(0.0008, 0.015, 252)
        >>> ir = calculate_information_ratio(port_ret, bench_ret)
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Portfolio and benchmark returns must have same length")
    
    excess_returns = portfolio_returns - benchmark_returns
    avg_excess_return = np.mean(excess_returns) * 252  # Annualized
    
    tracking_error = calculate_tracking_error(portfolio_returns, benchmark_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return avg_excess_return / tracking_error


def calculate_maximum_drawdown_series(cumulative_returns: Union[pd.Series, np.ndarray]) -> Tuple[float, pd.Series]:
    """
    Calculate maximum drawdown and drawdown series.
    
    Args:
        cumulative_returns: Series or array of cumulative returns (or prices)
        
    Returns:
        Tuple of (max_drawdown, drawdown_series)
        
    Example:
        >>> cum_returns = (1 + np.random.normal(0.001, 0.02, 252)).cumprod()
        >>> max_dd, dd_series = calculate_maximum_drawdown_series(cum_returns)
    """
    if isinstance(cumulative_returns, pd.Series):
        cum_ret = cumulative_returns
    else:
        cum_ret = pd.Series(cumulative_returns)
    
    # Calculate running maximum
    running_max = cum_ret.expanding().max()
    
    # Calculate drawdown series
    drawdown_series = (cum_ret - running_max) / running_max
    
    # Maximum drawdown
    max_drawdown = abs(drawdown_series.min())
    
    return max_drawdown, drawdown_series


def calculate_tail_ratio(returns: np.ndarray, percentile: float = 5.0) -> float:
    """
    Calculate tail ratio (average of top percentile / average of bottom percentile).
    
    Args:
        returns: Array of returns
        percentile: Percentile to use for tails (e.g., 5.0 for top/bottom 5%)
        
    Returns:
        Tail ratio
        
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 1000)
        >>> tail_ratio = calculate_tail_ratio(returns, 5.0)
    """
    if len(returns) == 0:
        return 1.0
    
    # Calculate percentile thresholds
    top_threshold = np.percentile(returns, 100 - percentile)
    bottom_threshold = np.percentile(returns, percentile)
    
    # Get tail returns
    top_tail = returns[returns >= top_threshold]
    bottom_tail = returns[returns <= bottom_threshold]
    
    if len(top_tail) == 0 or len(bottom_tail) == 0:
        return 1.0
    
    # Calculate averages
    avg_top = np.mean(top_tail)
    avg_bottom = np.mean(bottom_tail)
    
    if avg_bottom == 0:
        return np.inf if avg_top > 0 else 1.0
    
    return abs(avg_top / avg_bottom)


def calculate_skewness(returns: np.ndarray) -> float:
    """
    Calculate skewness of returns.
    
    Args:
        returns: Array of returns
        
    Returns:
        Skewness value
        
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 1000)
        >>> skew = calculate_skewness(returns)
    """
    if len(returns) < 3:
        return 0.0
    
    return stats.skew(returns)


def calculate_kurtosis(returns: np.ndarray) -> float:
    """
    Calculate excess kurtosis of returns.
    
    Args:
        returns: Array of returns
        
    Returns:
        Excess kurtosis value (normal distribution has kurtosis = 0)
        
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 1000)
        >>> kurt = calculate_kurtosis(returns)
    """
    if len(returns) < 4:
        return 0.0
    
    return stats.kurtosis(returns, fisher=True)  # fisher=True returns excess kurtosis


def calculate_downside_deviation(returns: np.ndarray, mar: float = 0.0) -> float:
    """
    Calculate downside deviation relative to minimum acceptable return.
    
    Args:
        returns: Array of returns
        mar: Minimum acceptable return (default 0.0)
        
    Returns:
        Annualized downside deviation
        
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 252)
        >>> dd = calculate_downside_deviation(returns, 0.0)
    """
    if len(returns) == 0:
        return 0.0
    
    # Only consider returns below MAR
    downside_returns = returns[returns < mar] - mar
    
    if len(downside_returns) == 0:
        return 0.0
    
    # Calculate downside variance
    downside_variance = np.mean(downside_returns ** 2)
    
    # Annualize and return standard deviation
    return np.sqrt(downside_variance * 252)


def calculate_sortino_ratio(returns: np.ndarray, mar: float = 0.0, 
                           periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (excess return / downside deviation).
    
    Args:
        returns: Array of returns
        mar: Minimum acceptable return
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Sortino ratio
        
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 252)
        >>> sortino = calculate_sortino_ratio(returns, 0.0)
    """
    if len(returns) == 0:
        return 0.0
    
    avg_return = np.mean(returns) * periods_per_year
    downside_dev = calculate_downside_deviation(returns, mar)
    
    if downside_dev == 0:
        return np.inf if avg_return > mar else 0.0
    
    return (avg_return - mar) / downside_dev


def calculate_calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annualized return / maximum drawdown).
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Calmar ratio
        
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 252)
        >>> calmar = calculate_calmar_ratio(returns)
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate annualized return
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1 if hasattr(cumulative, 'iloc') else cumulative[-1] - 1
    ann_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    
    # Calculate maximum drawdown
    max_dd, _ = calculate_maximum_drawdown_series(cumulative)
    
    if max_dd == 0:
        return np.inf if ann_return > 0 else 0.0
    
    return ann_return / max_dd 