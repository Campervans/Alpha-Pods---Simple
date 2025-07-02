"""Results caching utilities for strategy comparison.

This module provides functions to save and load strategy results to/from disk,
with automatic fallback generation for missing results.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import shutil
import pandas as pd
import numpy as np


def save_results(name: str, results: Dict[str, Any], path: Path = Path("results")) -> None:
    """Save results dictionary to disk with timestamp.
    
    Args:
        name: Strategy name (e.g., 'ml_cleir', 'cleir', 'spy')
        results: Dictionary containing strategy results
        path: Directory to save results (default: 'results')
    
    Raises:
        IOError: If unable to save file
    """
    # Ensure directory exists
    path.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to results
    results_with_meta = {
        'data': results,
        'saved_at': datetime.now().isoformat(),
        'strategy_name': name
    }
    
    # Save to pickle file
    filename = path / f"{name}_results.pkl"
    try:
        with open(filename, 'wb') as f:
            pickle.dump(results_with_meta, f)
        print(f"[green]✓ Saved {name} results to {filename}[/green]")
    except Exception as e:
        raise IOError(f"Failed to save {name} results: {e}")


def load_results(name: str, *, refresh: bool = False, path: Path = Path("results")) -> Dict[str, Any]:
    """Load results from cache, with automatic generation fallback.
    
    Args:
        name: Strategy name to load
        refresh: Force regeneration even if cache exists
        path: Directory containing cached results
    
    Returns:
        Dictionary containing strategy results
        
    Raises:
        FileNotFoundError: If results not found and cannot be generated
    """
    filename = path / f"{name}_results.pkl"
    
    # Check if we should use cache
    if not refresh and filename.exists():
        try:
            with open(filename, 'rb') as f:
                cached = pickle.load(f)
                print(f"[blue]ℹ Loaded {name} results from cache[/blue]")
                return cached['data']
        except Exception as e:
            print(f"[yellow]⚠ Cache corrupted, regenerating: {e}[/yellow]")
    
    # Generate results based on strategy type
    print(f"[yellow]⚡ Generating {name} results...[/yellow]")
    
    if name == "cleir":
        results = _generate_baseline_cleir()
    elif name == "spy":
        results = _generate_spy_benchmark()
    elif name == "ml_cleir":
        # ML results must be pre-generated
        raise FileNotFoundError(
            f"ML-Enhanced CLEIR results not found at {filename}. "
            "Please run: python scripts/run_ml_backtest.py"
        )
    else:
        raise ValueError(f"Unknown strategy name: {name}")
    
    # Save generated results
    save_results(name, results, path)
    return results


def _generate_baseline_cleir() -> Dict[str, Any]:
    """Generate baseline CLEIR results by running optimization."""
    from src.utils.cleir_runner import run_baseline_cleir
    return run_baseline_cleir()


def _generate_spy_benchmark() -> Dict[str, Any]:
    """Generate SPY benchmark results.
    
    This is a placeholder - will be implemented in Step 6.
    """
    # TODO: Implement in Step 6
    raise NotImplementedError("SPY benchmark generation will be implemented in Step 6")


def list_cached_results(path: Path = Path("results")) -> Dict[str, Dict[str, Any]]:
    """List all cached results with metadata.
    
    Returns:
        Dictionary mapping strategy names to their metadata
    """
    cached = {}
    
    if not path.exists():
        return cached
    
    for file in path.glob("*_results.pkl"):
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                name = data.get('strategy_name', file.stem.replace('_results', ''))
                cached[name] = {
                    'file': str(file),
                    'saved_at': data.get('saved_at', 'Unknown'),
                    'size_kb': file.stat().st_size / 1024
                }
        except Exception:
            continue
    
    return cached


def clear_cache(name: Optional[str] = None, path: Path = Path("results")) -> None:
    """Clear cached results.
    
    Args:
        name: Specific strategy to clear, or None to clear all
        path: Directory containing cached results
    """
    if isinstance(name, (str, Path)) and path == Path("results"):
        # Handle old signature clear_cache(path) for backwards compatibility
        path = Path(name)
        name = None
    
    path = Path(path)
    
    if name:
        filename = path / f"{name}_results.pkl"
        if filename.exists():
            filename.unlink()
            print(f"[green]✓ Cleared cache for {name}[/green]")
        else:
            print(f"[yellow]⚠ No cache found for {name}[/yellow]")
    else:
        # Clear all
        count = 0
        if path.exists():
            for file in path.glob("*_results.pkl"):
                file.unlink()
                count += 1
        print(f"[green]✓ Cleared {count} cached results[/green]")


def load_spy_benchmark(start_date: str, end_date: str, 
                      cache_dir: str = "cache/results") -> Dict[str, Any]:
    """Load SPY benchmark data and format as backtest result.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        cache_dir: Directory for caching results
        
    Returns:
        Dictionary with SPY performance formatted like backtest results
    """
    from ..market_data.downloader import download_benchmark_data
    from ..utils.core import annualize_return, calculate_sharpe_ratio, calculate_max_drawdown
    
    # Check cache first
    cache_key = f"spy_benchmark_{start_date}_{end_date}"
    cached = load_results(cache_key, path=Path(cache_dir))
    if cached is not None:
        return cached
    
    # Download SPY data
    spy_data = download_benchmark_data(["SPY"], start_date, end_date)
    
    if "SPY" not in spy_data:
        raise ValueError("Failed to download SPY data")
    
    spy_prices = spy_data["SPY"]
    
    # Calculate returns
    spy_returns = spy_prices.pct_change().dropna()
    
    # Calculate cumulative returns (index starting at 100)
    cumulative_returns = (1 + spy_returns).cumprod()
    spy_index = pd.Series(index=spy_prices.index, dtype=float)
    spy_index.iloc[0] = 100.0
    spy_index.iloc[1:] = 100.0 * cumulative_returns.values
    
    # Calculate metrics
    total_return = (spy_prices.iloc[-1] / spy_prices.iloc[0]) - 1.0
    annual_return = annualize_return(total_return, len(spy_returns), 252)
    volatility = spy_returns.std() * np.sqrt(252)
    sharpe_ratio = calculate_sharpe_ratio(spy_returns, 0.0, 252)
    max_drawdown = calculate_max_drawdown(spy_returns)
    
    # Format as backtest result
    result = {
        'index_values': spy_index,
        'daily_returns': spy_returns,
        'metrics': {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_turnover': 0.0,  # Buy and hold
            'transaction_costs': 0.0  # No trading
        },
        'metadata': {
            'strategy': 'SPY Benchmark',
            'start_date': start_date,
            'end_date': end_date,
            'ticker': 'SPY'
        }
    }
    
    # Cache the result
    save_results(cache_key, result, Path(cache_dir))
    
    return result 