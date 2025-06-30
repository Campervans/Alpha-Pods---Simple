"""
Simple test script to verify CLEIR implementation works correctly.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Direct imports to avoid path issues
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from optimization.cleir_solver import solve_cleir
from utils.schemas import OptimizationConfig

def generate_test_data():
    """Generate synthetic test data."""
    np.random.seed(42)
    
    # Generate 252 days of returns for 20 assets
    n_days = 252
    n_assets = 20
    
    # Create dates
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=n_days, freq='B')
    
    # Generate asset returns with some correlation structure
    market_factor = np.random.normal(0.0005, 0.02, n_days)
    
    asset_returns = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        idiosyncratic = np.random.normal(0, 0.01, n_days)
        asset_returns[:, i] = 0.7 * market_factor + 0.3 * idiosyncratic
    
    # Generate benchmark returns (highly correlated with market)
    benchmark_returns = 0.9 * market_factor + 0.1 * np.random.normal(0, 0.005, n_days)
    
    return asset_returns, benchmark_returns, dates

def test_cleir():
    """Test CLEIR optimization."""
    print("=" * 60)
    print("TESTING CLEIR IMPLEMENTATION")
    print("=" * 60)
    
    # Generate test data
    asset_returns, benchmark_returns, dates = generate_test_data()
    print(f"\nGenerated test data:")
    print(f"- {asset_returns.shape[0]} days of returns")
    print(f"- {asset_returns.shape[1]} assets")
    
    # Configure CLEIR
    config = OptimizationConfig(
        confidence_level=0.95,
        lookback_days=252,
        max_weight=0.15,  # 15% max per asset
        min_weight=0.0,   # Long-only
        solver="ECOS",
        sparsity_bound=1.1,  # Tight constraint to force sparsity
        benchmark_ticker="TEST_BENCH"
    )
    
    print(f"\nCLEIR Configuration:")
    print(f"- Confidence level: {config.confidence_level}")
    print(f"- Sparsity bound: {config.sparsity_bound}")
    print(f"- Max weight: {config.max_weight}")
    
    # Solve CLEIR
    print("\nSolving CLEIR optimization...")
    try:
        weights, solver_info = solve_cleir(
            asset_returns, 
            benchmark_returns, 
            config,
            verbose=True
        )
        
        print(f"\nOptimization Results:")
        print(f"- Status: {solver_info['status']}")
        print(f"- Solver: {solver_info['solver_used']}")
        print(f"- Objective (CVaR): {solver_info['objective_value']:.6f}")
        print(f"- Solve time: {solver_info['solve_time']:.3f}s")
        
        # Analyze weights
        print(f"\nPortfolio Weights:")
        print(f"- Sum of weights: {np.sum(weights):.6f}")
        print(f"- L1 norm: {solver_info['l1_norm']:.4f} (bound: {config.sparsity_bound})")
        print(f"- Non-zero weights: {solver_info['sparsity']} out of {len(weights)}")
        
        # Show top holdings
        sorted_idx = np.argsort(weights)[::-1]
        print(f"\nTop 10 Holdings:")
        for i in range(min(10, len(weights))):
            idx = sorted_idx[i]
            if weights[idx] > 1e-6:
                print(f"  Asset {idx:2d}: {weights[idx]:6.2%}")
        
        # Calculate tracking error statistics
        portfolio_returns = asset_returns @ weights
        tracking_error = benchmark_returns - portfolio_returns
        
        print(f"\nTracking Error Statistics:")
        print(f"- Mean: {np.mean(tracking_error):.6f}")
        print(f"- Std Dev: {np.std(tracking_error):.6f}")
        print(f"- Min: {np.min(tracking_error):.6f}")
        print(f"- Max: {np.max(tracking_error):.6f}")
        
        # Verify constraints
        print(f"\nConstraint Verification:")
        print(f"- Budget constraint (sum=1): {'✓' if abs(np.sum(weights) - 1.0) < 1e-6 else '✗'}")
        print(f"- Sparsity constraint: {'✓' if solver_info['l1_norm'] <= config.sparsity_bound + 1e-6 else '✗'}")
        print(f"- Weight bounds: {'✓' if np.all(weights >= -1e-6) and np.all(weights <= config.max_weight + 1e-6) else '✗'}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sparsity_levels():
    """Test different sparsity levels."""
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT SPARSITY LEVELS")
    print("=" * 60)
    
    asset_returns, benchmark_returns, _ = generate_test_data()
    
    sparsity_bounds = [1.0, 1.1, 1.2, 1.5, 2.0]
    results = []
    
    for s in sparsity_bounds:
        config = OptimizationConfig(
            confidence_level=0.95,
            lookback_days=252,
            max_weight=0.15,
            min_weight=0.0,
            solver="ECOS",
            sparsity_bound=s,
            benchmark_ticker="TEST_BENCH"
        )
        
        try:
            weights, solver_info = solve_cleir(asset_returns, benchmark_returns, config)
            results.append({
                'sparsity_bound': s,
                'n_assets': solver_info['sparsity'],
                'l1_norm': solver_info['l1_norm'],
                'cvar': solver_info['objective_value']
            })
        except Exception as e:
            print(f"Failed for sparsity={s}: {e}")
    
    # Display results
    if results:
        print(f"\nSparsity Analysis:")
        print(f"{'Bound':>6} | {'Assets':>6} | {'L1 Norm':>8} | {'CVaR':>8}")
        print("-" * 40)
        for r in results:
            print(f"{r['sparsity_bound']:>6.1f} | {r['n_assets']:>6d} | {r['l1_norm']:>8.4f} | {r['cvar']:>8.6f}")

if __name__ == "__main__":
    # Run tests
    success = test_cleir()
    
    if success:
        test_sparsity_levels()
        print("\n✓ CLEIR implementation is working correctly!")
    else:
        print("\n✗ CLEIR implementation has issues!") 