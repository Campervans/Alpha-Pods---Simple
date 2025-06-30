#!/usr/bin/env python3
"""Test the terminal visualization functionality."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.gui.visualization import plot_index_comparison


def main():
    """Test visualization of CLEIR vs SPY."""
    print("Testing CLEIR Index vs SPY Visualization")
    print("=" * 50)
    
    # Plot comparison
    stats = plot_index_comparison(
        cleir_csv_path="results/cleir_index_gui.csv",
        benchmark_ticker="SPY"
    )
    
    print("\n✅ Visualization complete!")
    print(f"\nCLEIR outperformed SPY by {(stats['cleir']['annual_return'] - stats['spy']['annual_return']) * 100:.1f}% annually")


if __name__ == "__main__":
    main() 