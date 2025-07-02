"""Integration tests for the comparison dashboard."""

import pytest
from pathlib import Path
import tempfile
import shutil
import pandas as pd
import numpy as np

from src.analysis.metrics import summarise_results, calculate_relative_metrics


def test_summarise_results_integration():
    """Test summarise_results with various input formats."""
    # Test with full backtest results
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    index_values = pd.Series(range(100, 100 + len(dates)), index=dates, dtype=float)
    
    # Create results dict that matches what summarise_results expects
    results = {
        'index_values': index_values,
        'daily_returns': index_values.pct_change().dropna(),
        # These should be used directly by summarise_results
        'volatility': 0.15,
        'sharpe_ratio': 3.33,
        'max_drawdown': 0.05,
        'avg_turnover': 0.20,
        'total_transaction_costs': 0.01
    }
    
    summary = summarise_results(results)
    
    # Verify all metrics present
    expected_metrics = [
        'total_return', 'annual_return', 'volatility',
        'sharpe_ratio', 'max_drawdown', 'avg_turnover',
        'transaction_costs'
    ]
    
    for metric in expected_metrics:
        assert metric in summary
    
    # Check that calculated metrics are reasonable
    # The index goes from 100 to 464 (365 days), so total return should be 3.64
    assert abs(summary['total_return'] - 3.64) < 0.01
    
    # Check that pre-specified metrics are used when available
    assert summary['volatility'] == 0.15
    assert summary['sharpe_ratio'] == 3.33
    assert summary['max_drawdown'] == 0.05
    assert summary['avg_turnover'] == 0.20
    assert summary['transaction_costs'] == 0.01


def test_calculate_relative_metrics():
    """Test relative metric calculation."""
    strategy_metrics = {
        'annual_return': 0.15,
        'volatility': 0.18,
        'sharpe_ratio': 0.83,
        'max_drawdown': 0.12
    }
    
    benchmark_metrics = {
        'annual_return': 0.10,
        'volatility': 0.20,
        'sharpe_ratio': 0.50,
        'max_drawdown': 0.15
    }
    
    relative = calculate_relative_metrics(strategy_metrics, benchmark_metrics)
    
    # Check calculations
    assert relative['excess_return'] == 0.05  # 15% - 10%
    assert relative['tracking_error'] == 0.02  # abs(18% - 20%)
    assert relative['information_ratio'] == 2.5  # 0.05 / 0.02
    assert relative['relative_drawdown'] == 0.03  # 15% - 12%


def test_results_cache_basic():
    """Test basic results caching functionality."""
    from src.utils.results_cache import save_results, load_results
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        test_data = {
            'index_values': pd.Series([100, 105, 110]),
            'metrics': {'total_return': 0.10}
        }
        
        # Save and load
        save_results('test_strategy', test_data, temp_path)
        loaded = load_results('test_strategy', path=temp_path)
        
        # Verify
        assert loaded['metrics']['total_return'] == 0.10
        assert len(loaded['index_values']) == 3


def test_cleir_runner_imports():
    """Test that cleir_runner can be imported and has required functions."""
    from src.utils.cleir_runner import run_baseline_cleir
    
    # Just check it's callable
    assert callable(run_baseline_cleir)


def test_visualization_imports():
    """Test that visualization functions can be imported."""
    from src.gui.visualization import plot_equity_curves, render_metrics_table
    
    # Just check they're callable
    assert callable(plot_equity_curves)
    assert callable(render_metrics_table) 