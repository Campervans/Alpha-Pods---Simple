"""Generate final results and deliverables for Task A.

This script:
1. Loads CVaR and CLEIR index results
2. Generates daily_index_values.csv
3. Calculates performance metrics and saves performance_summary.csv
4. Creates index_performance_analysis.png comparison plot
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.core import annualize_return, calculate_sharpe_ratio, calculate_max_drawdown
from scripts.generate_performance_comparison_plots import load_index_data, create_comparison_plot


def main():
    """Generate all Task A deliverables."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    print("Generating Task A deliverables...")
    
    # Load index data
    cvar_df, cleir_df = load_index_data()
    
    if cvar_df is None and cleir_df is None:
        print("Error: No index data found. Please run CVaR or CLEIR optimization first.")
        return
    
    # 1. Generate daily_index_values.csv
    # Use whichever index is available, preferring CLEIR if both exist
    if cleir_df is not None:
        daily_values = cleir_df.copy()
        index_type = "CLEIR"
    else:
        daily_values = cvar_df.copy()
        index_type = "CVaR"
    
    # Save daily index values
    daily_values_path = results_dir / "daily_index_values.csv"
    daily_values.to_csv(daily_values_path, index=False)
    print(f"✓ Generated daily_index_values.csv using {index_type} data")
    
    # 2. Calculate performance metrics
    metrics_data = []
    
    # Process CVaR if available
    if cvar_df is not None:
        cvar_metrics = calculate_metrics(cvar_df, "CVaR Index")
        metrics_data.append(cvar_metrics)
    
    # Process CLEIR if available
    if cleir_df is not None:
        cleir_metrics = calculate_metrics(cleir_df, "CLEIR Index")
        metrics_data.append(cleir_metrics)
    
    # Add S&P 500 benchmark comparison if possible
    try:
        from src.market_data.downloader import download_benchmark_data
        
        # Get date range from available data
        if cleir_df is not None:
            start_date = cleir_df['Date'].min()
            end_date = cleir_df['Date'].max()
        else:
            start_date = cvar_df['Date'].min()
            end_date = cvar_df['Date'].max()
        
        # Download SPY data
        spy_data = download_benchmark_data(['SPY'], 
                                          start_date.strftime('%Y-%m-%d'), 
                                          end_date.strftime('%Y-%m-%d'))
        
        if 'SPY' in spy_data:
            spy_prices = spy_data['SPY']
            # Create DataFrame in same format
            spy_df = pd.DataFrame({
                'Date': spy_prices.index,
                'Index_Value': spy_prices.values
            })
            # Normalize to start at 100
            spy_df['Index_Value'] = (spy_df['Index_Value'] / spy_df['Index_Value'].iloc[0]) * 100
            
            spy_metrics = calculate_metrics(spy_df, "S&P 500 (SPY)")
            metrics_data.append(spy_metrics)
    except Exception as e:
        print(f"Warning: Could not load SPY benchmark data: {e}")
    
    # Save performance summary
    if metrics_data:
        performance_df = pd.DataFrame(metrics_data)
        performance_path = results_dir / "performance_summary.csv"
        performance_df.to_csv(performance_path, index=False)
        print("✓ Generated performance_summary.csv")
        
        # Display the metrics
        print("\nPerformance Summary:")
        print(performance_df.to_string(index=False))
    
    # 3. Generate comparison plot
    try:
        create_comparison_plot(cvar_df, cleir_df)
        print("✓ Generated index_performance_analysis.png")
    except Exception as e:
        print(f"Warning: Could not generate comparison plot: {e}")
    
    print("\nAll deliverables generated successfully!")


def calculate_metrics(df: pd.DataFrame, index_name: str) -> dict:
    """Calculate performance metrics for an index.
    
    Args:
        df: DataFrame with Date and Index_Value columns
        index_name: Name of the index
        
    Returns:
        Dictionary of metrics
    """
    # Calculate returns
    returns = df['Index_Value'].pct_change().dropna()
    
    # Calculate metrics
    total_return = (df['Index_Value'].iloc[-1] / df['Index_Value'].iloc[0]) - 1
    annual_return = annualize_return(total_return, len(returns), 252)
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = calculate_sharpe_ratio(returns, 0.0, 252)
    max_drawdown = calculate_max_drawdown(returns)
    
    # Get date range
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    return {
        'Index': index_name,
        'Start Date': start_date.strftime('%Y-%m-%d'),
        'End Date': end_date.strftime('%Y-%m-%d'),
        'Total Return': f"{total_return:.2%}",
        'Annual Return': f"{annual_return:.2%}",
        'Annual Volatility': f"{annual_volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.3f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Final Value': f"{df['Index_Value'].iloc[-1]:.2f}"
    }


if __name__ == "__main__":
    main() 