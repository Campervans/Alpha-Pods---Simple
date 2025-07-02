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

    if len(returns) == 0:
        return 0.0
    
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    
    # Return as positive value (loss)
    return abs(var_threshold)


def calculate_parametric_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:

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

    var = calculate_historical_var(returns, confidence)
    cvar = calculate_historical_cvar(returns, confidence)
    
    return var, cvar


def calculate_portfolio_volatility(weights: np.ndarray, covariance_matrix: np.ndarray) -> float:

    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    return np.sqrt(portfolio_variance)


def calculate_risk_contribution(weights: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:

    portfolio_vol = calculate_portfolio_volatility(weights, covariance_matrix)
    
    if portfolio_vol == 0:
        return np.zeros_like(weights)
    
    # Marginal risk contribution
    marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
    
    # Risk contribution = weight * marginal contribution / portfolio volatility
    risk_contrib = weights * marginal_contrib / portfolio_vol
    
    return risk_contrib


def estimate_covariance_matrix(returns: pd.DataFrame, method: str = "sample") -> np.ndarray:

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

    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Portfolio and benchmark returns must have same length")
    
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(252)
    
    return tracking_error


def calculate_information_ratio(portfolio_returns: np.ndarray,
                               benchmark_returns: np.ndarray) -> float:

    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Portfolio and benchmark returns must have same length")
    
    excess_returns = portfolio_returns - benchmark_returns
    avg_excess_return = np.mean(excess_returns) * 252  # Annualized
    
    tracking_error = calculate_tracking_error(portfolio_returns, benchmark_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return avg_excess_return / tracking_error


def calculate_maximum_drawdown_series(cumulative_returns: Union[pd.Series, np.ndarray]) -> Tuple[float, pd.Series]:

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

    if len(returns) < 3:
        return 0.0
    
    return stats.skew(returns)


def calculate_kurtosis(returns: np.ndarray) -> float:

    if len(returns) < 4:
        return 0.0
    
    return stats.kurtosis(returns, fisher=True)  # fisher=True returns excess kurtosis


def calculate_downside_deviation(returns: np.ndarray, mar: float = 0.0) -> float:

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

    if len(returns) == 0:
        return 0.0
    
    avg_return = np.mean(returns) * periods_per_year
    downside_dev = calculate_downside_deviation(returns, mar)
    
    if downside_dev == 0:
        return np.inf if avg_return > mar else 0.0
    
    return (avg_return - mar) / downside_dev


def calculate_calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:

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