"""Test deliverable generation scripts."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def test_import_generate_performance_comparison_plots():
    """Test that we can import the performance plot generation module."""
    try:
        from scripts.generate_performance_comparison_plots import (
            load_index_data, create_performance_plot, create_comparison_plot
        )
        assert callable(load_index_data)
        assert callable(create_performance_plot)
        assert callable(create_comparison_plot)
    except ImportError as e:
        pytest.fail(f"Failed to import generate_performance_comparison_plots: {e}")


def test_create_performance_plot():
    """Test creating a performance plot with dummy data."""
    from scripts.generate_performance_comparison_plots import create_performance_plot
    
    # Create dummy data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    values = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    
    df = pd.DataFrame({
        'Date': dates,
        'Index_Value': values
    })
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / 'test_plot.png'
        
        # Create plot
        create_performance_plot(df, 'Test Index', output_file)
        
        # Check that file was created
        assert output_file.exists()
        assert output_file.stat().st_size > 0


def test_generate_final_results_import():
    """Test that we can import the final results generation module."""
    try:
        from scripts.generate_final_results import main, calculate_metrics
        assert callable(main)
        assert callable(calculate_metrics)
    except ImportError as e:
        pytest.fail(f"Failed to import generate_final_results: {e}")


def test_calculate_metrics():
    """Test metric calculation function."""
    from scripts.generate_final_results import calculate_metrics
    
    # Create dummy data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    values = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    
    df = pd.DataFrame({
        'Date': dates,
        'Index_Value': values
    })
    
    # Calculate metrics
    metrics = calculate_metrics(df, 'Test Index')
    
    # Check that all expected metrics are present
    expected_keys = [
        'Index', 'Start Date', 'End Date', 'Total Return',
        'Annual Return', 'Annual Volatility', 'Sharpe Ratio',
        'Max Drawdown', 'Final Value'
    ]
    
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
    
    # Check that metrics have reasonable values
    assert metrics['Index'] == 'Test Index'
    assert metrics['Start Date'] == '2020-01-01'
    assert metrics['End Date'] == '2020-12-31'


def test_generate_deliverables_with_dummy_data():
    """Test full deliverable generation with dummy data."""
    from scripts.generate_final_results import main
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        results_dir = tmpdir / 'results'
        results_dir.mkdir()
        
        # Create dummy CVaR data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        cvar_values = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
        
        cvar_df = pd.DataFrame({
            'Date': dates,
            'Index_Value': cvar_values
        })
        cvar_df.to_csv(results_dir / 'cvar_index_gui.csv', index=False)
        
        # Temporarily change working directory
        original_cwd = Path.cwd()
        try:
            # Change to temp directory
            import os
            os.chdir(tmpdir)
            
            # Run main function
            main()
            
            # Check that deliverables were created
            assert (results_dir / 'daily_index_values.csv').exists()
            assert (results_dir / 'performance_summary.csv').exists()
            assert (results_dir / 'index_performance_analysis.png').exists()
            
            # Verify content of daily_index_values.csv
            daily_df = pd.read_csv(results_dir / 'daily_index_values.csv')
            assert len(daily_df) == len(dates)
            assert 'Date' in daily_df.columns
            assert 'Index_Value' in daily_df.columns
            
            # Verify content of performance_summary.csv
            perf_df = pd.read_csv(results_dir / 'performance_summary.csv')
            assert len(perf_df) >= 1  # At least CVaR index
            assert 'Index' in perf_df.columns
            assert 'Annual Return' in perf_df.columns
            
        finally:
            # Restore original directory
            os.chdir(original_cwd)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 