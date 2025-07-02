"""Tests for results cache utilities."""

import pytest
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from src.utils.results_cache import (
    save_results, 
    load_results, 
    list_cached_results,
    clear_cache
)


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


def test_save_and_load_results(temp_cache_dir):
    """Test basic save/load round trip."""
    # Create test data
    test_results = {
        'daily_values': [100, 101, 102],
        'returns': [0.01, 0.0099],
        'metrics': {'sharpe': 1.5}
    }
    
    # Save
    save_results('test_strategy', test_results, temp_cache_dir)
    
    # Verify file exists
    expected_file = temp_cache_dir / 'test_strategy_results.pkl'
    assert expected_file.exists()
    
    # Load
    loaded = load_results('test_strategy', path=temp_cache_dir)
    
    # Verify data matches
    assert loaded == test_results


def test_load_with_refresh(temp_cache_dir):
    """Test refresh flag forces regeneration."""
    # Save initial data
    initial_data = {'value': 1}
    save_results('test', initial_data, temp_cache_dir)
    
    # Load with refresh should fail (no generator implemented)
    with pytest.raises(ValueError, match="Unknown strategy"):
        load_results('test', refresh=True, path=temp_cache_dir)


def test_load_missing_ml_cleir(temp_cache_dir):
    """Test loading missing ML-CLEIR results raises helpful error."""
    with pytest.raises(FileNotFoundError, match="run_ml_backtest.py"):
        load_results('ml_cleir', path=temp_cache_dir)


def test_list_cached_results(temp_cache_dir):
    """Test listing cached results with metadata."""
    # Save multiple results
    save_results('strategy1', {'data': 1}, temp_cache_dir)
    save_results('strategy2', {'data': 2}, temp_cache_dir)
    
    # List cached
    cached = list_cached_results(temp_cache_dir)
    
    assert len(cached) == 2
    assert 'strategy1' in cached
    assert 'strategy2' in cached
    
    # Check metadata
    for name, meta in cached.items():
        assert 'file' in meta
        assert 'saved_at' in meta
        assert 'size_kb' in meta
        assert meta['size_kb'] > 0


def test_clear_cache_single(temp_cache_dir):
    """Test clearing single cached result."""
    # Save two results
    save_results('keep', {'data': 1}, temp_cache_dir)
    save_results('remove', {'data': 2}, temp_cache_dir)
    
    # Clear one
    clear_cache('remove', path=temp_cache_dir)
    
    # Verify
    assert (temp_cache_dir / 'keep_results.pkl').exists()
    assert not (temp_cache_dir / 'remove_results.pkl').exists()


def test_clear_cache_all(temp_cache_dir):
    """Test clearing all cached results."""
    # Save multiple results
    save_results('strategy1', {'data': 1}, temp_cache_dir)
    save_results('strategy2', {'data': 2}, temp_cache_dir)
    
    # Clear all
    clear_cache(name=None, path=temp_cache_dir)
    
    # Verify all removed
    assert len(list(temp_cache_dir.glob('*_results.pkl'))) == 0


def test_corrupted_cache(temp_cache_dir):
    """Test handling of corrupted cache file."""
    # Create corrupted file
    bad_file = temp_cache_dir / 'bad_results.pkl'
    bad_file.write_text("corrupted data")
    
    # Load should regenerate (but will fail since 'bad' is unknown)
    with pytest.raises(ValueError, match="Unknown strategy"):
        load_results('bad', path=temp_cache_dir)


def test_save_with_timestamp(temp_cache_dir):
    """Test that saved results include timestamp."""
    import pickle
    
    # Save results
    save_results('test', {'data': 123}, temp_cache_dir)
    
    # Load raw file
    with open(temp_cache_dir / 'test_results.pkl', 'rb') as f:
        raw = pickle.load(f)
    
    # Check structure
    assert 'data' in raw
    assert 'saved_at' in raw
    assert 'strategy_name' in raw
    assert raw['data'] == {'data': 123}
    assert raw['strategy_name'] == 'test'
    
    # Verify timestamp is valid
    datetime.fromisoformat(raw['saved_at'])


def test_directory_creation(temp_cache_dir):
    """Test that save_results creates directory if needed."""
    # Use non-existent subdirectory
    sub_dir = temp_cache_dir / 'subdir'
    assert not sub_dir.exists()
    
    # Save should create directory
    save_results('test', {'data': 1}, sub_dir)
    
    assert sub_dir.exists()
    assert (sub_dir / 'test_results.pkl').exists() 