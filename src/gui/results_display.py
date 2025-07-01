"""Results display utilities for the GUI."""

import pandas as pd
from rich.table import Table
from rich.console import Console
from typing import Dict, Optional

def create_performance_comparison_table(
    ml_metrics: Dict,
    cleir_metrics: Optional[Dict] = None,
    spy_metrics: Optional[Dict] = None
) -> Table:
    """Create a rich table comparing ML-enhanced, baseline CLEIR, and SPY performance.
    
    Args:
        ml_metrics: ML-enhanced strategy metrics
        cleir_metrics: Baseline CLEIR metrics (optional)
        spy_metrics: SPY benchmark metrics (optional)
        
    Returns:
        Rich Table object for display
    """
    table = Table(title="Performance Comparison (2020-2024)", show_header=True, header_style="bold magenta")
    
    # Add columns
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("ML-Enhanced CLEIR", justify="right", style="green")
    
    if cleir_metrics:
        table.add_column("Baseline CLEIR", justify="right", style="blue")
    
    if spy_metrics:
        table.add_column("SPY Benchmark", justify="right", style="yellow")
    
    # Define metrics to display with formatting
    metrics_config = [
        ("Total Return", "total_return", ".2%"),
        ("Annual Return", "annual_return", ".2%"),
        ("Volatility", "volatility", ".2%"),
        ("Sharpe Ratio", "sharpe_ratio", ".3f"),
        ("Max Drawdown", "max_drawdown", ".2%"),
        ("CVaR 95%", "cvar_95", ".2%"),
        ("Average Turnover", "avg_turnover", ".1%"),
        ("Transaction Costs", "total_transaction_costs", ".2%"),
    ]
    
    # Add rows
    for display_name, key, fmt in metrics_config:
        row = [display_name]
        
        # ML-Enhanced value
        ml_value = ml_metrics.get(key, 0)
        if fmt.endswith('%'):
            row.append(f"{ml_value:{fmt}}")
        else:
            row.append(f"{ml_value:{fmt}}")
        
        # Baseline CLEIR value
        if cleir_metrics:
            cleir_value = cleir_metrics.get(key, 0)
            if fmt.endswith('%'):
                row.append(f"{cleir_value:{fmt}}")
            else:
                row.append(f"{cleir_value:{fmt}}")
        
        # SPY value
        if spy_metrics:
            spy_value = spy_metrics.get(key, 0)
            if fmt.endswith('%'):
                row.append(f"{spy_value:{fmt}}")
            else:
                row.append(f"{spy_value:{fmt}}")
        
        table.add_row(*row)
    
    # Add improvement row if we have baseline
    if cleir_metrics:
        table.add_row("", "", "", "")  # Empty row for spacing
        table.add_row(
            "[bold]Sharpe Improvement[/bold]",
            f"[bold green]+{((ml_metrics['sharpe_ratio'] / cleir_metrics['sharpe_ratio']) - 1) * 100:.1f}%[/bold green]",
            "—",
            "—" if spy_metrics else None
        )
    
    return table


def format_results_summary(results: Dict) -> str:
    """Format results for display in the GUI.
    
    Args:
        results: Dictionary containing backtest results
        
    Returns:
        Formatted string for display
    """
    summary_lines = []
    
    # Key metrics
    summary_lines.append("ML-Enhanced CLEIR Results:")
    summary_lines.append("")
    summary_lines.append(f"  Final Index Value   {results['daily_values'].iloc[-1]:.2f}")
    summary_lines.append(f"  Total Return        {results['total_return']:.2%}")
    summary_lines.append(f"  Annual Return       {results['annual_return']:.2%}")
    summary_lines.append(f"  Sharpe Ratio        {results['sharpe_ratio']:.3f}")
    summary_lines.append(f"  Max Drawdown        {results['max_drawdown']:.2%}")
    
    # File outputs
    summary_lines.append("")
    summary_lines.append("Generated files:")
    summary_lines.append("  ✓ ml_enhanced_index.csv")
    summary_lines.append("  ✓ ml_feature_importance.png")
    summary_lines.append("  ✓ ml_shap_analysis.png")
    summary_lines.append("  ✓ ml_performance_comparison.png")
    summary_lines.append("  ✓ ml_predictions_analysis.png")
    summary_lines.append("  ✓ ml_performance_report.md")
    
    return "\n".join(summary_lines)


def calculate_spy_metrics(start_date: str = '2020-01-01', end_date: str = '2024-12-31') -> Dict:
    """Calculate SPY benchmark metrics for the same period.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        
    Returns:
        Dictionary of SPY performance metrics
    """
    try:
        import yfinance as yf
        
        # Download SPY data
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=True)
        if len(spy) == 0:
            return {}
        
        # Calculate returns
        # Handle different column names
        if 'Adj Close' in spy.columns:
            price_col = 'Adj Close'
        elif 'Close' in spy.columns:
            price_col = 'Close'
        else:
            price_col = spy.columns[0]
        
        spy_returns = spy[price_col].pct_change().dropna()
        
        # Calculate metrics
        total_return = (spy[price_col].iloc[-1] / spy[price_col].iloc[0]) - 1
        n_years = len(spy_returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        volatility = spy_returns.std() * (252 ** 0.5)
        sharpe_ratio = (spy_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Drawdown
        cum_returns = (1 + spy_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # CVaR
        var_95 = spy_returns.quantile(0.05)
        cvar_95 = spy_returns[spy_returns <= var_95].mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cvar_95': cvar_95,
            'avg_turnover': 0.0,  # Buy and hold
            'total_transaction_costs': 0.001  # Initial purchase only
        }
        
    except Exception as e:
        print(f"Error calculating SPY metrics: {e}")
        return {} 