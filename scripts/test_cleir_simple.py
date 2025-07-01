"""
Simple test script to verify CLEIR optimization works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.optimization.cleir_solver import solve_cleir
from src.utils.schemas import OptimizationConfig

# Create synthetic test data
np.random.seed(42)
n_periods = 252  # 1 year
n_assets = 10

# Generate asset returns (with some correlation to benchmark)
benchmark_returns = np.random.normal(0.0005, 0.01, n_periods)
asset_returns = np.zeros((n_periods, n_assets))

for i in range(n_assets):
    # Each asset has some correlation with benchmark + idiosyncratic component
    correlation = 0.7 + 0.2 * np.random.rand()
    asset_returns[:, i] = correlation * benchmark_returns + np.sqrt(1 - correlation**2) * np.random.normal(0.0005, 0.01, n_periods)

# Create config
config = OptimizationConfig(
    confidence_level=0.95,
    lookback_days=252,
    max_weight=0.5,    # Relaxed for testing
    min_weight=0.0,    # Long-only
    solver="ECOS",
    sparsity_bound=1.2,  # L1 norm constraint
    benchmark_ticker="TEST_BENCHMARK"
)

print("Testing CLEIR optimization...")
print(f"Assets: {n_assets}, Periods: {n_periods}")
print(f"Sparsity bound: {config.sparsity_bound}")

# Run optimization
try:
    weights, info = solve_cleir(asset_returns, benchmark_returns, config, verbose=True)
    
    print("\nOptimization Results:")
    print(f"Status: {info['status']}")
    print(f"Solver: {info['solver_used']}")
    print(f"Objective: {info['objective_value']:.6f}")
    print(f"Sparsity: {info['sparsity']}/{n_assets} assets")
    print(f"L1 norm: {info['l1_norm']:.4f}")
    
    print("\nWeights:")
    for i, w in enumerate(weights):
        if w > 1e-6:
            print(f"  Asset {i}: {w:.4f}")
    
    print(f"\nSum of weights: {np.sum(weights):.6f}")
    print(f"Min weight: {np.min(weights):.6f}")
    print(f"Max weight: {np.max(weights):.6f}")
    
    # Verify constraints
    assert np.all(weights >= -1e-6), "Weights should be non-negative"
    assert np.abs(np.sum(weights) - 1.0) < 1e-6, "Weights should sum to 1"
    assert np.sum(np.abs(weights)) <= config.sparsity_bound + 1e-6, "L1 norm constraint violated"
    
    print("\n✅ All constraints satisfied!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc() 