"""Generate performance comparison plots for CVaR and CLEIR indices.

This module provides functions to:
1. Load index data from CSV files
2. Create individual performance plots for each index
3. Generate comparison plots with benchmarks
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import Tuple, Optional, Union

# Set matplotlib backend to non-interactive to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

# Define results directory relative to script location
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_index_data(
        cvar_csv: Union[Path, str] = None,
        cleir_csv: Union[Path, str] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load CVaR and CLEIR index data from CSV files.
    
    Args:
        cvar_csv: Path to CVaR index CSV file (default: results/cvar_index_gui.csv)
        cleir_csv: Path to CLEIR index CSV file (default: results/cleir_index_gui.csv)
        
    Returns:
        Tuple of (cvar_df, cleir_df), where each is a DataFrame with Date and Index_Value columns
        or None if the file doesn't exist
    """
    if cvar_csv is None:
        cvar_csv = RESULTS_DIR / "cvar_index_gui.csv"
    if cleir_csv is None:
        cleir_csv = RESULTS_DIR / "cleir_index_gui.csv"
    
    # Load CVaR data
    cvar_df = None
    if Path(cvar_csv).exists():
        cvar_df = pd.read_csv(cvar_csv)
        if 'Date' in cvar_df.columns:
            cvar_df['Date'] = pd.to_datetime(cvar_df['Date'])
    
    # Load CLEIR data
    cleir_df = None
    if Path(cleir_csv).exists():
        cleir_df = pd.read_csv(cleir_csv)
        if 'Date' in cleir_df.columns:
            cleir_df['Date'] = pd.to_datetime(cleir_df['Date'])
    
    return cvar_df, cleir_df


def create_performance_plot(
        df: pd.DataFrame,
        label: str,
        output_file: Union[str, Path],
        benchmark_data: Optional[pd.DataFrame] = None,
        benchmark_label: str = "S&P 500") -> None:
    """Create a performance plot for an index.
    
    Args:
        df: DataFrame with Date and Index_Value columns
        label: Label for the index (e.g., "CVaR", "CLEIR")
        output_file: Path to save the plot
        benchmark_data: Optional benchmark data with Date and Close columns
        benchmark_label: Label for benchmark
    """
    if df is None or df.empty:
        raise ValueError(f"No data available for {label}")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up the plot
    plt.figure(figsize=(12, 7))
    
    # Prepare index data
    df = df.copy()
    if 'Date' in df.columns:
        df = df.set_index('Date')
    
    # Normalize to base 100
    base_value = df['Index_Value'].iloc[0]
    normalized_index = (df['Index_Value'] / base_value) * 100
    
    # Plot the index
    plt.plot(df.index, normalized_index, label=f'{label} Index', linewidth=2, color='blue')
    
    # Plot benchmark if provided
    if benchmark_data is not None and not benchmark_data.empty:
        benchmark_data = benchmark_data.copy()
        if 'Date' in benchmark_data.columns:
            benchmark_data = benchmark_data.set_index('Date')
        
        # Align benchmark dates with index dates
        common_dates = df.index.intersection(benchmark_data.index)
        if len(common_dates) > 0:
            benchmark_aligned = benchmark_data.loc[common_dates]
            
            # Normalize benchmark
            if 'Close' in benchmark_aligned.columns:
                benchmark_values = benchmark_aligned['Close']
            elif 'Index_Value' in benchmark_aligned.columns:
                benchmark_values = benchmark_aligned['Index_Value']
            else:
                benchmark_values = benchmark_aligned.iloc[:, 0]
            
            benchmark_base = benchmark_values.iloc[0]
            normalized_benchmark = (benchmark_values / benchmark_base) * 100
            
            plt.plot(benchmark_aligned.index, normalized_benchmark, 
                    label=benchmark_label, linewidth=2, color='red', alpha=0.7)
    
    # Formatting
    plt.title(f'{label} Index Performance', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Index Value (Base = 100)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gcf().autofmt_xdate()
    
    # Add statistics box
    returns = df['Index_Value'].pct_change().dropna()
    total_return = (df['Index_Value'].iloc[-1] / df['Index_Value'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    stats_text = f'{label} Statistics:\n'
    stats_text += f'Total Return: {total_return:.1%}\n'
    stats_text += f'Annual Return: {annual_return:.1%}\n'
    stats_text += f'Annual Volatility: {volatility:.1%}\n'
    stats_text += f'Sharpe Ratio: {sharpe:.2f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_plot(
        cvar_df: Optional[pd.DataFrame] = None,
        cleir_df: Optional[pd.DataFrame] = None,
        output_file: Union[str, Path] = None) -> None:
    """Create a comparison plot of CVaR and CLEIR indices.
    
    Args:
        cvar_df: CVaR index data
        cleir_df: CLEIR index data
        output_file: Path to save the plot (default: results/index_performance_analysis.png)
    """
    if output_file is None:
        output_file = RESULTS_DIR / "index_performance_analysis.png"
    
    # Ensure at least one index is available
    if (cvar_df is None or cvar_df.empty) and (cleir_df is None or cleir_df.empty):
        raise ValueError("No index data available for comparison")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Plot CVaR if available
    if cvar_df is not None and not cvar_df.empty:
        cvar_df = cvar_df.copy()
        if 'Date' in cvar_df.columns:
            cvar_df = cvar_df.set_index('Date')
        
        # Normalize to base 100
        cvar_base = cvar_df['Index_Value'].iloc[0]
        cvar_normalized = (cvar_df['Index_Value'] / cvar_base) * 100
        
        plt.plot(cvar_df.index, cvar_normalized, label='CVaR Index', 
                linewidth=2, color='blue')
    
    # Plot CLEIR if available
    if cleir_df is not None and not cleir_df.empty:
        cleir_df = cleir_df.copy()
        if 'Date' in cleir_df.columns:
            cleir_df = cleir_df.set_index('Date')
        
        # Normalize to base 100
        cleir_base = cleir_df['Index_Value'].iloc[0]
        cleir_normalized = (cleir_df['Index_Value'] / cleir_base) * 100
        
        plt.plot(cleir_df.index, cleir_normalized, label='CLEIR Index', 
                linewidth=2, color='green')
    
    # Formatting
    plt.title('CVaR vs CLEIR Index Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Index Value (Base = 100)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gcf().autofmt_xdate()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test the functions
    print("Loading index data...")
    cvar_df, cleir_df = load_index_data()
    
    if cvar_df is not None:
        print(f"Loaded CVaR data: {len(cvar_df)} rows")
        create_performance_plot(cvar_df, 'CVaR', RESULTS_DIR / 'cvar_index_performance_analysis.png')
        print("Created CVaR performance plot")
    
    if cleir_df is not None:
        print(f"Loaded CLEIR data: {len(cleir_df)} rows")
        create_performance_plot(cleir_df, 'CLEIR', RESULTS_DIR / 'cleir_index_performance_analysis.png')
        print("Created CLEIR performance plot")
    
    if cvar_df is not None or cleir_df is not None:
        create_comparison_plot(cvar_df, cleir_df)
        print("Created comparison plot") 