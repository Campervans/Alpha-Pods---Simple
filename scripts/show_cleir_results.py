#!/usr/bin/env python3
"""Display CLEIR optimization results with SPY comparison."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.gui.visualization import plot_index_comparison


def main():
    """Show CLEIR results with terminal plot and comparison table."""
    print("\n" + "="*80)
    print("CLEIR Index Performance Analysis")
    print("="*80)
    
    # Plot comparison - this will show the graph and table
    stats = plot_index_comparison(
        cleir_csv_path="results/cleir_index_gui.csv",
        benchmark_ticker="SPY"
    )
    
    # Additional summary
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    
    cleir_stats = stats['cleir']
    spy_stats = stats['spy']
    
    # Calculate outperformance
    annual_outperformance = cleir_stats['annual_return'] - spy_stats['annual_return']
    total_outperformance = cleir_stats['total_return'] - spy_stats['total_return']
    sharpe_improvement = cleir_stats['sharpe_ratio'] - spy_stats['sharpe_ratio']
    
    print(f"CLEIR Annual Return: {cleir_stats['annual_return']:.2%}")
    print(f"SPY Annual Return: {spy_stats['annual_return']:.2%}")
    print(f"Annual Outperformance: {annual_outperformance:.2%}")
    print(f"\nCLEIR Total Return: {cleir_stats['total_return']:.2%}")
    print(f"SPY Total Return: {spy_stats['total_return']:.2%}")
    print(f"Total Outperformance: {total_outperformance:.2%}")
    print(f"\nCLEIR Sharpe Ratio: {cleir_stats['sharpe_ratio']:.3f}")
    print(f"SPY Sharpe Ratio: {spy_stats['sharpe_ratio']:.3f}")
    print(f"Sharpe Improvement: {sharpe_improvement:.3f}")
    
    print("\n" + "="*80)
    print("Note: The backtest period starts from 2015-07-06 due to data availability")
    print("for all selected stocks. The index is normalized to start at 100.00.")
    print("="*80)


if __name__ == "__main__":
    main() 