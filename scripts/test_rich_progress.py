"""Test script to demonstrate rich progress displays for optimizations."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from src.optimization.cleir_solver import solve_cleir
from src.optimization.cvar_solver import solve_cvar
from src.utils.schemas import OptimizationConfig
from rich.console import Console

console = Console()

def test_rich_progress():
    """Test rich progress displays for both solvers."""
    
    # Create sample data
    np.random.seed(42)
    n_periods = 252
    n_assets = 50
    
    # Generate random returns
    asset_returns = np.random.randn(n_periods, n_assets) * 0.01 + 0.0002
    benchmark_returns = np.random.randn(n_periods) * 0.008 + 0.0003
    
    # Create optimization config
    config = OptimizationConfig(
        confidence_level=0.95,
        lookback_days=252,
        max_weight=0.05,
        min_weight=0.0,
        sparsity_bound=1.2,  # For CLEIR
        solver="CLARABEL"
    )
    
    console.print("\n[bold magenta]Testing CLEIR Optimization with Rich Progress[/bold magenta]")
    console.print("=" * 60)
    
    # Test CLEIR solver
    weights_cleir, info_cleir = solve_cleir(
        asset_returns, 
        benchmark_returns, 
        config, 
        verbose=True
    )
    
    console.print(f"\n[green]CLEIR Results:[/green]")
    console.print(f"  Non-zero weights: {np.sum(weights_cleir > 1e-6)}")
    console.print(f"  Max weight: {np.max(weights_cleir):.4f}")
    console.print(f"  Sum of weights: {np.sum(weights_cleir):.6f}")
    
    console.print("\n[bold magenta]Testing CVaR Optimization with Rich Progress[/bold magenta]")
    console.print("=" * 60)
    
    # Test CVaR solver
    weights_cvar, info_cvar = solve_cvar(
        asset_returns, 
        config, 
        verbose=True
    )
    
    console.print(f"\n[green]CVaR Results:[/green]")
    console.print(f"  Non-zero weights: {np.sum(weights_cvar > 1e-6)}")
    console.print(f"  Max weight: {np.max(weights_cvar):.4f}")
    console.print(f"  Sum of weights: {np.sum(weights_cvar):.6f}")
    
    console.print("\n[bold green]✅ Rich progress displays working correctly![/bold green]")

if __name__ == "__main__":
    test_rich_progress() 