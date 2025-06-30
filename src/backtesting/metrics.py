"""
Performance analytics and metrics calculation.

This module provides comprehensive performance analysis including
risk metrics, return analytics, and benchmark comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from ..utils.core import (
    annualize_return, annualize_volatility, calculate_sharpe_ratio,
    calculate_max_drawdown
)
from ..optimization.risk_models import (
    calculate_historical_cvar, calculate_historical_var,
    calculate_tracking_error, calculate_information_ratio,
    calculate_sortino_ratio, calculate_calmar_ratio
)


def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for a return series.
    
    Args:
        returns: Series of periodic returns
        risk_free_rate: Annualized risk-free rate
        
    Returns:
        Dictionary with performance metrics
        
    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        >>> metrics = calculate_metrics(returns)
    """
    if len(returns) == 0:
        return {}
    
    # Remove any NaN values
    clean_returns = returns.dropna()
    
    if len(clean_returns) == 0:
        return {}
    
    # Basic return metrics
    total_return = (1 + clean_returns).prod() - 1
    n_periods = len(clean_returns)
    annual_return = annualize_return(total_return, n_periods, 252)
    annual_volatility = annualize_volatility(clean_returns, 252)
    
    # Risk-adjusted metrics
    sharpe_ratio = calculate_sharpe_ratio(clean_returns, risk_free_rate, 252)
    sortino_ratio = calculate_sortino_ratio(clean_returns.values, risk_free_rate, 252)
    calmar_ratio = calculate_calmar_ratio(clean_returns.values, 252)
    
    # Drawdown metrics
    max_drawdown = calculate_max_drawdown(clean_returns)
    
    # Risk metrics
    cvar_95 = calculate_historical_cvar(clean_returns.values, 0.95)
    cvar_99 = calculate_historical_cvar(clean_returns.values, 0.99)
    var_95 = calculate_historical_var(clean_returns.values, 0.95)
    var_99 = calculate_historical_var(clean_returns.values, 0.99)
    
    # Distribution metrics
    skewness = clean_returns.skew()
    kurtosis = clean_returns.kurtosis()
    
    # Hit rate (percentage of positive returns)
    hit_rate = (clean_returns > 0).mean()
    
    # Best and worst periods
    best_day = clean_returns.max()
    worst_day = clean_returns.min()
    
    # Win/loss ratio
    positive_returns = clean_returns[clean_returns > 0]
    negative_returns = clean_returns[clean_returns < 0]
    
    avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
    avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'var_95': var_95,
        'var_99': var_99,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'hit_rate': hit_rate,
        'best_day': best_day,
        'worst_day': worst_day,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'n_observations': n_periods
    }


def create_performance_report(index_returns: pd.Series, 
                            benchmark_returns: Dict[str, pd.Series],
                            risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Create comparative performance report between index and benchmarks.
    
    Args:
        index_returns: CVaR index returns
        benchmark_returns: Dictionary of benchmark name to returns series
        risk_free_rate: Annualized risk-free rate
        
    Returns:
        DataFrame with comparative metrics
        
    Example:
        >>> benchmarks = {'SPY': spy_returns, 'Equal Weight': ew_returns}
        >>> report = create_performance_report(cvar_returns, benchmarks)
    """
    # Calculate metrics for the CVaR index
    cvar_metrics = calculate_metrics(index_returns, risk_free_rate)
    
    # Calculate metrics for each benchmark
    all_metrics = {'CVaR Index': cvar_metrics}
    
    for name, returns in benchmark_returns.items():
        # Align returns with index returns
        aligned_returns = returns.reindex(index_returns.index).dropna()
        benchmark_metrics = calculate_metrics(aligned_returns, risk_free_rate)
        all_metrics[name] = benchmark_metrics
    
    # Convert to DataFrame
    report_df = pd.DataFrame(all_metrics).T
    
    # Add relative metrics (vs first benchmark if available)
    if benchmark_returns:
        first_benchmark_name = list(benchmark_returns.keys())[0]
        first_benchmark_returns = benchmark_returns[first_benchmark_name].reindex(index_returns.index).dropna()
        
        # Calculate tracking error and information ratio
        tracking_error = calculate_tracking_error(
            index_returns.values, first_benchmark_returns.values
        )
        information_ratio = calculate_information_ratio(
            index_returns.values, first_benchmark_returns.values
        )
        
        # Add to CVaR index row
        report_df.loc['CVaR Index', 'tracking_error'] = tracking_error
        report_df.loc['CVaR Index', 'information_ratio'] = information_ratio
        
        # Calculate excess return
        excess_return = report_df.loc['CVaR Index', 'annual_return'] - report_df.loc[first_benchmark_name, 'annual_return']
        report_df.loc['CVaR Index', 'excess_return'] = excess_return
    
    return report_df


def calculate_rolling_metrics(returns: pd.Series, 
                            window: int = 252,
                            metrics: List[str] = ['sharpe_ratio', 'max_drawdown', 'cvar_95']) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Args:
        returns: Series of returns
        window: Rolling window size (default 252 for 1 year)
        metrics: List of metrics to calculate
        
    Returns:
        DataFrame with rolling metrics
        
    Example:
        >>> rolling_metrics = calculate_rolling_metrics(returns, 252, ['sharpe_ratio', 'cvar_95'])
    """
    results = {}
    
    for metric in metrics:
        rolling_values = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            
            if metric == 'sharpe_ratio':
                value = calculate_sharpe_ratio(window_returns, 0.0, 252)
            elif metric == 'max_drawdown':
                value = calculate_max_drawdown(window_returns)
            elif metric == 'cvar_95':
                value = calculate_historical_cvar(window_returns.values, 0.95)
            elif metric == 'annual_return':
                total_ret = (1 + window_returns).prod() - 1
                value = annualize_return(total_ret, len(window_returns), 252)
            elif metric == 'annual_volatility':
                value = annualize_volatility(window_returns, 252)
            else:
                value = np.nan
            
            rolling_values.append(value)
        
        # Create series with proper index
        rolling_index = returns.index[window-1:]
        results[metric] = pd.Series(rolling_values, index=rolling_index)
    
    return pd.DataFrame(results)


def create_monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Create monthly returns table for detailed analysis.
    
    Args:
        returns: Series of daily returns
        
    Returns:
        DataFrame with monthly returns by year
        
    Example:
        >>> monthly_table = create_monthly_returns_table(daily_returns)
    """
    # Calculate monthly returns
    monthly_returns = (1 + returns).resample('M').prod() - 1
    
    # Create pivot table
    monthly_table = monthly_returns.to_frame('return')
    monthly_table['year'] = monthly_table.index.year
    monthly_table['month'] = monthly_table.index.month
    
    pivot_table = monthly_table.pivot(index='year', columns='month', values='return')
    
    # Add month names as column headers
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
    
    # Add annual returns
    annual_returns = (1 + returns).resample('Y').prod() - 1
    pivot_table['Annual'] = annual_returns.values
    
    return pivot_table


def plot_performance_comparison(index_returns: pd.Series,
                              benchmark_returns: Dict[str, pd.Series],
                              title: str = "Performance Comparison") -> plt.Figure:
    """
    Create performance comparison plot.
    
    Args:
        index_returns: CVaR index returns
        benchmark_returns: Dictionary of benchmark returns
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Calculate cumulative returns
    cvar_cumulative = (1 + index_returns).cumprod()
    
    # Plot 1: Cumulative returns
    ax1 = axes[0, 0]
    ax1.plot(cvar_cumulative.index, cvar_cumulative.values, label='CVaR Index', linewidth=2)
    
    for name, returns in benchmark_returns.items():
        aligned_returns = returns.reindex(index_returns.index).fillna(0)
        benchmark_cumulative = (1 + aligned_returns).cumprod()
        ax1.plot(benchmark_cumulative.index, benchmark_cumulative.values, label=name, alpha=0.7)
    
    ax1.set_title('Cumulative Returns')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rolling Sharpe Ratio
    ax2 = axes[0, 1]
    rolling_sharpe = calculate_rolling_metrics(index_returns, 252, ['sharpe_ratio'])
    ax2.plot(rolling_sharpe.index, rolling_sharpe['sharpe_ratio'], label='CVaR Index')
    ax2.set_title('Rolling 1-Year Sharpe Ratio')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown
    ax3 = axes[1, 0]
    cumulative = (1 + index_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax3.plot(drawdown.index, drawdown.values, color='red')
    ax3.set_title('Drawdown')
    ax3.set_ylabel('Drawdown')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Return distribution
    ax4 = axes[1, 1]
    ax4.hist(index_returns.values, bins=50, alpha=0.7, density=True, label='CVaR Index')
    
    if benchmark_returns:
        first_benchmark = list(benchmark_returns.values())[0]
        aligned_benchmark = first_benchmark.reindex(index_returns.index).dropna()
        ax4.hist(aligned_benchmark.values, bins=50, alpha=0.5, density=True, 
                label=list(benchmark_returns.keys())[0])
    
    ax4.set_title('Return Distribution')
    ax4.set_xlabel('Daily Return')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_performance_report(report_df: pd.DataFrame, 
                          monthly_table: pd.DataFrame,
                          filepath: str):
    """
    Save performance analysis to Excel file.
    
    Args:
        report_df: Performance metrics comparison DataFrame
        monthly_table: Monthly returns table
        filepath: Path to save Excel file
    """
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        # Performance metrics
        report_df.to_excel(writer, sheet_name='Performance_Metrics')
        
        # Monthly returns
        monthly_table.to_excel(writer, sheet_name='Monthly_Returns')
        
        # Format the sheets
        workbook = writer.book
        
        # Format performance metrics sheet
        metrics_sheet = writer.sheets['Performance_Metrics']
        percent_format = workbook.add_format({'num_format': '0.00%'})
        number_format = workbook.add_format({'num_format': '0.000'})
        
        # Apply formatting based on metric type
        for row in range(1, len(report_df) + 1):
            for col in range(1, len(report_df.columns) + 1):
                col_name = report_df.columns[col-1]
                if 'return' in col_name or 'drawdown' in col_name or 'hit_rate' in col_name:
                    metrics_sheet.write(row, col, report_df.iloc[row-1, col-1], percent_format)
                else:
                    metrics_sheet.write(row, col, report_df.iloc[row-1, col-1], number_format)
        
        # Format monthly returns sheet
        monthly_sheet = writer.sheets['Monthly_Returns']
        for row in range(1, len(monthly_table) + 1):
            for col in range(1, len(monthly_table.columns) + 1):
                monthly_sheet.write(row, col, monthly_table.iloc[row-1, col-1], percent_format)
    
    print(f"Performance report saved to {filepath}")


def calculate_risk_adjusted_metrics(returns: pd.Series, 
                                  benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Calculate additional risk-adjusted performance metrics.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Optional benchmark returns for relative metrics
        
    Returns:
        Dictionary with risk-adjusted metrics
    """
    metrics = {}
    
    # Basic risk metrics
    metrics['var_95_daily'] = calculate_historical_var(returns.values, 0.95)
    metrics['cvar_95_daily'] = calculate_historical_cvar(returns.values, 0.95)
    
    # Annualized versions
    metrics['var_95_annual'] = metrics['var_95_daily'] * np.sqrt(252)
    metrics['cvar_95_annual'] = metrics['cvar_95_daily'] * np.sqrt(252)
    
    # Tail ratio
    tail_returns = returns.quantile([0.05, 0.95])
    metrics['tail_ratio'] = abs(tail_returns.iloc[1] / tail_returns.iloc[0]) if tail_returns.iloc[0] != 0 else np.inf
    
    # Relative metrics if benchmark provided
    if benchmark_returns is not None:
        aligned_benchmark = benchmark_returns.reindex(returns.index).dropna()
        common_dates = returns.index.intersection(aligned_benchmark.index)
        
        if len(common_dates) > 0:
            port_aligned = returns.loc[common_dates]
            bench_aligned = aligned_benchmark.loc[common_dates]
            
            metrics['tracking_error'] = calculate_tracking_error(
                port_aligned.values, bench_aligned.values
            )
            metrics['information_ratio'] = calculate_information_ratio(
                port_aligned.values, bench_aligned.values
            )
            
            # Beta calculation
            excess_returns = port_aligned - bench_aligned
            benchmark_variance = bench_aligned.var()
            metrics['beta'] = excess_returns.cov(bench_aligned) / benchmark_variance if benchmark_variance > 0 else 1
            
            # Alpha calculation (CAPM)
            rf_rate = 0.0  # Assuming zero risk-free rate
            expected_return = rf_rate + metrics['beta'] * (bench_aligned.mean() * 252 - rf_rate)
            actual_return = port_aligned.mean() * 252
            metrics['alpha'] = actual_return - expected_return
    
    return metrics
