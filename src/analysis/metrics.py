"""Standardized metric extraction for strategy comparison.

This module provides utilities to extract consistent performance metrics
from different strategy result formats.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union


def summarise_results(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract standardized metrics from any strategy results.
    
    Args:
        results: Strategy results dictionary containing at minimum:
            - 'daily_values' or 'index_values': pd.Series of portfolio values
            - Optional: 'returns', 'volatility', 'sharpe_ratio', etc.
    
    Returns:
        Dictionary with standardized metrics:
            - total_return: Total return over period
            - annual_return: Annualized return
            - volatility: Annualized volatility
            - sharpe_ratio: Sharpe ratio (0% risk-free rate)
            - max_drawdown: Maximum drawdown (positive value)
            - avg_turnover: Average portfolio turnover
            - transaction_costs: Total transaction costs
    """
    metrics = {}
    
    # Get daily values series
    if 'daily_values' in results:
        daily_values = results['daily_values']
    elif 'index_values' in results:
        daily_values = results['index_values']
    else:
        raise ValueError("Results must contain 'daily_values' or 'index_values'")
    
    # Ensure it's a pandas Series
    if isinstance(daily_values, list):
        daily_values = pd.Series(daily_values)
    
    # Calculate returns if not provided
    if 'returns' in results:
        returns = results['returns']
    else:
        returns = daily_values.pct_change().dropna()
    
    # Total return
    metrics['total_return'] = (daily_values.iloc[-1] / daily_values.iloc[0]) - 1.0
    
    # Annual return
    n_days = len(returns)
    if n_days > 0:
        metrics['annual_return'] = (1 + metrics['total_return']) ** (252 / n_days) - 1
    else:
        metrics['annual_return'] = 0.0
    
    # Volatility
    if 'volatility' in results:
        metrics['volatility'] = results['volatility']
    elif 'annual_volatility' in results:
        metrics['volatility'] = results['annual_volatility']
    else:
        metrics['volatility'] = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    if 'sharpe_ratio' in results:
        metrics['sharpe_ratio'] = results['sharpe_ratio']
    else:
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0.0
    
    # Max drawdown
    if 'max_drawdown' in results:
        # Ensure positive value
        metrics['max_drawdown'] = abs(results['max_drawdown'])
    else:
        # Calculate from daily values
        cumulative = daily_values / daily_values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = abs(drawdown.min())
    
    # Turnover
    if 'avg_turnover' in results:
        metrics['avg_turnover'] = results['avg_turnover']
    elif 'turnover_history' in results and results['turnover_history']:
        metrics['avg_turnover'] = np.mean(results['turnover_history'])
    else:
        metrics['avg_turnover'] = 0.0  # Default for buy-and-hold
    
    # Transaction costs
    if 'total_transaction_costs' in results:
        metrics['transaction_costs'] = results['total_transaction_costs']
    elif 'transaction_costs_history' in results and results['transaction_costs_history']:
        metrics['transaction_costs'] = sum(results['transaction_costs_history'])
    else:
        metrics['transaction_costs'] = 0.0
    
    # Round all metrics for display
    for key, value in metrics.items():
        if key in ['total_return', 'annual_return', 'volatility', 'max_drawdown', 
                   'avg_turnover', 'transaction_costs']:
            metrics[key] = round(value, 4)
        else:  # Sharpe ratio
            metrics[key] = round(value, 3)
    
    return metrics


def calculate_relative_metrics(strategy_metrics: Dict[str, float], 
                             benchmark_metrics: Dict[str, float]) -> Dict[str, float]:
    """Calculate relative performance metrics vs benchmark.
    
    Args:
        strategy_metrics: Metrics for the strategy
        benchmark_metrics: Metrics for the benchmark
    
    Returns:
        Dictionary with relative metrics:
            - excess_return: Annual return difference
            - tracking_error: Volatility of return differences
            - information_ratio: Excess return / tracking error
            - relative_drawdown: Difference in max drawdowns
    """
    relative = {}
    
    # Excess return
    relative['excess_return'] = (
        strategy_metrics['annual_return'] - benchmark_metrics['annual_return']
    )
    
    # For now, approximate tracking error as volatility difference
    # (proper calculation would need daily return differences)
    relative['tracking_error'] = abs(
        strategy_metrics['volatility'] - benchmark_metrics['volatility']
    )
    
    # Information ratio
    if relative['tracking_error'] > 0:
        relative['information_ratio'] = (
            relative['excess_return'] / relative['tracking_error']
        )
    else:
        relative['information_ratio'] = 0.0
    
    # Relative drawdown (positive means strategy had smaller drawdown)
    relative['relative_drawdown'] = (
        benchmark_metrics['max_drawdown'] - strategy_metrics['max_drawdown']
    )
    
    # Round for display
    for key, value in relative.items():
        relative[key] = round(value, 4)
    
    return relative


def format_metric_for_display(value: float, metric_type: str) -> str:
    """Format metric value for display based on type.
    
    Args:
        value: Metric value
        metric_type: Type of metric (e.g., 'return', 'ratio', 'cost')
    
    Returns:
        Formatted string for display
    """
    if metric_type in ['total_return', 'annual_return', 'volatility', 
                       'max_drawdown', 'avg_turnover', 'transaction_costs',
                       'excess_return', 'tracking_error', 'relative_drawdown']:
        return f"{value:.2%}"
    elif metric_type in ['sharpe_ratio', 'information_ratio']:
        return f"{value:.3f}"
    else:
        return f"{value:.4f}" 