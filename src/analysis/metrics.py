"""get some metrics for comparing strats"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union


def summarise_results(results: Dict[str, Any]) -> Dict[str, float]:

    metrics = {}
    
    # daily values series
    if 'daily_values' in results:
        daily_values = results['daily_values']
    elif 'index_values' in results:
        daily_values = results['index_values']
    else:
        raise ValueError("need 'daily_values' or 'index_values' in results")
    
    # make sure it's a pandas series
    if isinstance(daily_values, list):
        daily_values = pd.Series(daily_values)
    
    # calc returns if not there
    if 'returns' in results:
        returns = results['returns']
    else:
        returns = daily_values.pct_change().dropna()
    
    # total return
    metrics['total_return'] = (daily_values.iloc[-1] / daily_values.iloc[0]) - 1.0
    
    # annual return
    n_days = len(returns)
    if n_days > 0:
        metrics['annual_return'] = (1 + metrics['total_return']) ** (252 / n_days) - 1
    else:
        metrics['annual_return'] = 0.0
    
    # volatility
    if 'volatility' in results:
        metrics['volatility'] = results['volatility']
    elif 'annual_volatility' in results:
        metrics['volatility'] = results['annual_volatility']
    else:
        metrics['volatility'] = returns.std() * np.sqrt(252)
    
    # sharpe
    if 'sharpe_ratio' in results:
        metrics['sharpe_ratio'] = results['sharpe_ratio']
    else:
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0.0
    
    # max drawdown
    if 'max_drawdown' in results:
        # make it positive
        metrics['max_drawdown'] = abs(results['max_drawdown'])
    else:
        # calc from daily values
        cumulative = daily_values / daily_values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = abs(drawdown.min())
    
    # turnover
    if 'avg_turnover' in results:
        metrics['avg_turnover'] = results['avg_turnover']
    elif 'turnover_history' in results and results['turnover_history']:
        metrics['avg_turnover'] = np.mean(results['turnover_history'])
    else:
        metrics['avg_turnover'] = 0.0  # if buy-and-hold
    
    # transaction costs
    if 'total_transaction_costs' in results:
        metrics['transaction_costs'] = results['total_transaction_costs']
    elif 'transaction_costs_history' in results and results['transaction_costs_history']:
        metrics['transaction_costs'] = sum(results['transaction_costs_history'])
    else:
        metrics['transaction_costs'] = 0.0
    
    # round everything for display
    for key, value in metrics.items():
        if key in ['total_return', 'annual_return', 'volatility', 'max_drawdown', 
                   'avg_turnover', 'transaction_costs']:
            metrics[key] = round(value, 4)
        else:  # Sharpe ratio
            metrics[key] = round(value, 3)
    
    return metrics


def calculate_relative_metrics(strategy_metrics: Dict[str, float], 
                             benchmark_metrics: Dict[str, float]) -> Dict[str, float]:

    relative = {}
    
    # excess return
    relative['excess_return'] = (
        strategy_metrics['annual_return'] - benchmark_metrics['annual_return']
    )
    
    # TODO: this is a dumb way to get tracking error, fix it later
    # need daily return differences for a proper calculation
    relative['tracking_error'] = abs(
        strategy_metrics['volatility'] - benchmark_metrics['volatility']
    )
    
    # info ratio
    if relative['tracking_error'] > 0:
        relative['information_ratio'] = (
            relative['excess_return'] / relative['tracking_error']
        )
    else:
        relative['information_ratio'] = 0.0
    
    # relative drawdown (positive means our strat had a smaller one)
    relative['relative_drawdown'] = (
        benchmark_metrics['max_drawdown'] - strategy_metrics['max_drawdown']
    )
    
    # round for display
    for key, value in relative.items():
        relative[key] = round(value, 4)
    
    return relative


def format_metric_for_display(value: float, metric_type: str) -> str:

    if metric_type in ['total_return', 'annual_return', 'volatility', 
                       'max_drawdown', 'avg_turnover', 'transaction_costs',
                       'excess_return', 'tracking_error', 'relative_drawdown']:
        return f"{value:.2%}"
    elif metric_type in ['sharpe_ratio', 'information_ratio']:
        return f"{value:.3f}"
    else:
        return f"{value:.4f}" 