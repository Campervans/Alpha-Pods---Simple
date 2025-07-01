"""
Test script to verify tracking error formulation in CLEIR.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

# Create synthetic data where we know the answer
np.random.seed(42)
n_periods = 100
n_assets = 3

# Benchmark returns
benchmark_returns = np.array([0.001] * n_periods)  # 0.1% daily return

# Asset returns:
# Asset 0: Exactly matches benchmark
# Asset 1: Outperforms benchmark
# Asset 2: Underperforms benchmark
asset_returns = np.zeros((n_periods, n_assets))
asset_returns[:, 0] = benchmark_returns  # Perfect match
asset_returns[:, 1] = benchmark_returns + 0.0005  # Outperforms by 0.05%
asset_returns[:, 2] = benchmark_returns - 0.0005  # Underperforms by 0.05%

print("Average returns:")
print(f"Benchmark: {np.mean(benchmark_returns):.4f}")
print(f"Asset 0 (match): {np.mean(asset_returns[:, 0]):.4f}")
print(f"Asset 1 (outperform): {np.mean(asset_returns[:, 1]):.4f}")
print(f"Asset 2 (underperform): {np.mean(asset_returns[:, 2]):.4f}")

# Test different portfolio weights
test_weights = [
    [1.0, 0.0, 0.0],  # All in matching asset
    [0.0, 1.0, 0.0],  # All in outperforming asset
    [0.0, 0.0, 1.0],  # All in underperforming asset
    [0.33, 0.33, 0.34],  # Equal weight
]

print("\nTracking error analysis:")
print("(Tracking error = benchmark - portfolio)")

for i, weights in enumerate(test_weights):
    weights = np.array(weights)
    portfolio_returns = asset_returns @ weights
    tracking_error = benchmark_returns - portfolio_returns
    
    print(f"\nPortfolio {i+1}: weights = {weights}")
    print(f"  Mean portfolio return: {np.mean(portfolio_returns):.4f}")
    print(f"  Mean tracking error: {np.mean(tracking_error):.4f}")
    print(f"  Std tracking error: {np.std(tracking_error):.4f}")
    print(f"  95% CVaR of tracking error: {np.percentile(tracking_error, 95):.4f}")
    
    # Sort tracking errors to see the distribution
    sorted_te = np.sort(tracking_error)
    print(f"  Worst 5% tracking errors: {sorted_te[-5:]}")

print("\nInterpretation:")
print("- Positive tracking error means benchmark outperforms (bad for us)")
print("- Negative tracking error means portfolio outperforms (good for us)")
print("- Minimizing CVaR of tracking error should favor portfolios that don't underperform") 