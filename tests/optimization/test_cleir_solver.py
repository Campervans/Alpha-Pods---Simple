"""
Unit tests for CLEIR (CVaR-LASSO Enhanced Index Replication) solver.
"""

import pytest
import numpy as np
import pandas as pd
from src.optimization.cleir_solver import solve_cleir, create_cleir_problem
from src.utils.schemas import OptimizationConfig


class TestCLEIRSolver:
    """Test suite for CLEIR optimization."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_periods = 250  # ~1 year of daily data
        n_assets = 10
        
        # Generate asset returns with some correlation structure
        mean_returns = np.random.uniform(-0.0005, 0.0015, n_assets)
        volatilities = np.random.uniform(0.01, 0.03, n_assets)
        
        # Simple factor model for correlation
        factor_returns = np.random.normal(0, 0.015, n_periods)
        asset_returns = np.zeros((n_periods, n_assets))
        
        for i in range(n_assets):
            idiosyncratic = np.random.normal(0, volatilities[i] * 0.7, n_periods)
            asset_returns[:, i] = mean_returns[i] + 0.5 * factor_returns + idiosyncratic
        
        # Generate benchmark returns (correlated with assets)
        benchmark_returns = 0.0008 + 0.8 * factor_returns + np.random.normal(0, 0.01, n_periods)
        
        return asset_returns, benchmark_returns
    
    def test_cleir_basic_functionality(self, sample_data):
        """Test basic CLEIR optimization runs without errors."""
        asset_returns, benchmark_returns = sample_data
        
        config = OptimizationConfig(
            confidence_level=0.95,
            lookback_days=250,
            max_weight=0.3,
            min_weight=0.0,
            solver="ECOS",
            sparsity_bound=1.2,
            benchmark_ticker="TEST_BENCH"
        )
        
        weights, solver_info = solve_cleir(asset_returns, benchmark_returns, config)
        
        # Basic assertions
        assert weights is not None
        assert len(weights) == asset_returns.shape[1]
        assert solver_info['status'] in ['optimal', 'optimal_inaccurate']
    
    def test_sparsity_constraint(self, sample_data):
        """Test that sparsity constraint is respected."""
        asset_returns, benchmark_returns = sample_data
        
        # Very tight sparsity bound
        config = OptimizationConfig(
            confidence_level=0.95,
            lookback_days=250,
            max_weight=0.5,
            min_weight=0.0,
            solver="ECOS",
            sparsity_bound=1.0,  # Forces exact sparsity
            benchmark_ticker="TEST_BENCH"
        )
        
        weights, solver_info = solve_cleir(asset_returns, benchmark_returns, config)
        
        # Check L1 norm constraint
        l1_norm = np.sum(np.abs(weights))
        assert l1_norm <= config.sparsity_bound + 1e-6, f"L1 norm {l1_norm} exceeds bound {config.sparsity_bound}"
        
        # Check sparsity info
        assert 'l1_norm' in solver_info
        assert solver_info['l1_norm'] <= config.sparsity_bound + 1e-6
    
    def test_budget_constraint(self, sample_data):
        """Test that weights sum to 1."""
        asset_returns, benchmark_returns = sample_data
        
        config = OptimizationConfig(
            confidence_level=0.95,
            lookback_days=250,
            max_weight=0.3,
            min_weight=0.0,
            solver="ECOS",
            sparsity_bound=1.5,
            benchmark_ticker="TEST_BENCH"
        )
        
        weights, _ = solve_cleir(asset_returns, benchmark_returns, config)
        
        # Check budget constraint
        weight_sum = np.sum(weights)
        assert np.isclose(weight_sum, 1.0, atol=1e-6), f"Weights sum to {weight_sum}, not 1.0"
    
    def test_cleir_vs_equal_weight(self, sample_data):
        """Test that CLEIR achieves lower CVaR of tracking error than equal weight."""
        asset_returns, benchmark_returns = sample_data
        n_assets = asset_returns.shape[1]
        
        config = OptimizationConfig(
            confidence_level=0.95,
            lookback_days=250,
            max_weight=0.3,
            min_weight=0.0,
            solver="ECOS",
            sparsity_bound=1.5,
            benchmark_ticker="TEST_BENCH"
        )
        
        # Get CLEIR weights
        cleir_weights, solver_info = solve_cleir(asset_returns, benchmark_returns, config)
        
        # Calculate tracking errors
        cleir_portfolio_returns = asset_returns @ cleir_weights
        cleir_tracking_error = benchmark_returns - cleir_portfolio_returns
        
        equal_weights = np.ones(n_assets) / n_assets
        equal_portfolio_returns = asset_returns @ equal_weights
        equal_tracking_error = benchmark_returns - equal_portfolio_returns
        
        # Calculate CVaR of tracking errors
        from src.optimization.risk_models import calculate_historical_cvar
        cleir_cvar = calculate_historical_cvar(cleir_tracking_error, config.confidence_level)
        equal_cvar = calculate_historical_cvar(equal_tracking_error, config.confidence_level)
        
        # CLEIR should have lower CVaR (or at least not worse)
        assert cleir_cvar <= equal_cvar + 1e-6, \
            f"CLEIR CVaR {cleir_cvar} is worse than equal weight {equal_cvar}"
        
        # The objective value should match our calculated CVaR (approximately)
        if solver_info['status'] == 'optimal':
            assert np.isclose(solver_info['objective_value'], cleir_cvar, rtol=0.1), \
                "Solver objective doesn't match calculated CVaR"
    
    def test_weight_bounds(self, sample_data):
        """Test that weight bounds are respected."""
        asset_returns, benchmark_returns = sample_data
        
        min_weight = 0.02
        max_weight = 0.15
        
        config = OptimizationConfig(
            confidence_level=0.95,
            lookback_days=250,
            max_weight=max_weight,
            min_weight=min_weight,
            solver="ECOS",
            sparsity_bound=1.5,
            benchmark_ticker="TEST_BENCH"
        )
        
        weights, _ = solve_cleir(asset_returns, benchmark_returns, config)
        
        # Check bounds
        assert np.all(weights >= min_weight - 1e-6), f"Some weights below {min_weight}"
        assert np.all(weights <= max_weight + 1e-6), f"Some weights above {max_weight}"
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Only 5 observations
        asset_returns = np.random.normal(0, 0.01, (5, 10))
        benchmark_returns = np.random.normal(0, 0.01, 5)
        
        config = OptimizationConfig(
            confidence_level=0.95,
            lookback_days=250,
            max_weight=0.3,
            min_weight=0.0,
            solver="ECOS",
            sparsity_bound=1.2,
            benchmark_ticker="TEST_BENCH"
        )
        
        with pytest.raises(ValueError, match="at least 10 observations"):
            solve_cleir(asset_returns, benchmark_returns, config)
    
    def test_different_solvers(self, sample_data):
        """Test that different solvers produce similar results."""
        asset_returns, benchmark_returns = sample_data
        
        results = {}
        
        for solver in ['ECOS', 'SCS']:
            config = OptimizationConfig(
                confidence_level=0.95,
                lookback_days=250,
                max_weight=0.3,
                min_weight=0.0,
                solver=solver,
                sparsity_bound=1.2,
                benchmark_ticker="TEST_BENCH"
            )
            
            try:
                weights, solver_info = solve_cleir(asset_returns, benchmark_returns, config)
                if solver_info['status'] in ['optimal', 'optimal_inaccurate']:
                    results[solver] = weights
            except Exception:
                pass  # Solver might not be installed
        
        # If we have multiple results, they should be similar
        if len(results) > 1:
            solver_names = list(results.keys())
            for i in range(1, len(solver_names)):
                weights1 = results[solver_names[0]]
                weights2 = results[solver_names[i]]
                
                # Check that solutions are similar (allowing for solver differences)
                max_diff = np.max(np.abs(weights1 - weights2))
                assert max_diff < 0.05, \
                    f"Large difference between {solver_names[0]} and {solver_names[i]}: {max_diff}" 