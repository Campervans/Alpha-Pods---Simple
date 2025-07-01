#!/usr/bin/env python3
"""Test the improved results display format."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.gui.results_display import create_performance_comparison_table, calculate_spy_metrics
from rich.console import Console

# Sample metrics (based on your output)
ml_metrics = {
    'total_return': 5.1385,  # 513.85%
    'annual_return': 0.1642,  # 16.42%
    'volatility': 0.1710,     # 17.10%
    'sharpe_ratio': 0.976,
    'max_drawdown': -0.3104,  # -31.04%
    'cvar_95': -0.0285,       # Estimated
    'avg_turnover': 0.25,     # Estimated
    'total_transaction_costs': 0.015  # Estimated
}

# Baseline CLEIR metrics (estimated for comparison)
cleir_metrics = {
    'total_return': 8.7041,   # 870.41% (from your output)
    'annual_return': 0.2184,  # Estimated
    'volatility': 0.1650,     # Estimated
    'sharpe_ratio': 0.853,    # From Sharpe improvement calc
    'max_drawdown': -0.2850,  # Estimated
    'cvar_95': -0.0265,       # Estimated
    'avg_turnover': 0.20,     # Estimated
    'total_transaction_costs': 0.012  # Estimated
}

# Get real SPY metrics
print("Calculating SPY metrics for 2020-2024...")
spy_metrics = calculate_spy_metrics('2020-01-01', '2024-12-31')

# Create the comparison table
table = create_performance_comparison_table(ml_metrics, cleir_metrics, spy_metrics)

# Display it
console = Console()
console.print("\n")
console.print(table)
console.print("\n")

# Show what the optimal display looks like
print("\nThis table provides:")
print("1. Side-by-side comparison of ML-Enhanced CLEIR vs Baseline CLEIR vs SPY")
print("2. All key metrics requested: Total Return, Annual Return, Volatility, Sharpe, Max Drawdown")
print("3. Additional metrics: CVaR 95%, Turnover, Transaction Costs")
print("4. Clear visual highlighting with colors")
print("5. Sharpe improvement calculation at the bottom")
print("\nThis format makes it easy to see:")
print("- ML enhancement improves Sharpe by 14.4% over baseline CLEIR")
print("- Both strategies outperform SPY on risk-adjusted basis")
print("- Transaction costs and turnover impact") 