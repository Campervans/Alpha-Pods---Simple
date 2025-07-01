#!/usr/bin/env python3
"""
Test ML caching functionality.

This script demonstrates how the caching speeds up repeated runs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time
from datetime import datetime

from src.utils.ml_cache import MLTrainingCache
from src.backtesting.alpha_engine import AlphaEnhancedBacktest
from src.optimization.alpha_cleir_solver import AlphaOptimizationConfig
from src.features.feature_sets import FeatureSet
from src.market_data.enhanced_downloader import load_data_from_cache
from src.utils.parallel_utils import ParallelConfig


def main():
    """Test ML caching."""
    print("🧪 Testing ML Cache System")
    print("=" * 50)
    
    # Initialize cache
    cache = MLTrainingCache()
    
    # Show cache info
    info = cache.get_cache_info()
    print(f"\n📊 Current cache status:")
    print(f"   Models: {info['models']}")
    print(f"   Features: {info['features']}")
    print(f"   Predictions: {info['predictions']}")
    print(f"   Total size: {info['total_size_mb']} MB")
    
    # Load some test data
    print("\n📈 Loading test data...")
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Load from cache
    data = load_data_from_cache(test_tickers)
    
    if data is None:
        print("❌ No cached data found. Please run data download first.")
        return
    
    prices = data['prices']
    volumes = data['volumes']
    
    # Limit data for testing
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    
    prices = prices.loc[start_date:end_date]
    volumes = volumes.loc[start_date:end_date]
    
    print(f"   Loaded {len(prices.columns)} stocks, {len(prices)} days")
    
    # Configure alpha backtest
    alpha_config = AlphaOptimizationConfig(
        target_return=0.10,
        confidence_level=0.95,
        max_weight=0.30,
        min_weight=0.0,
        tracking_error_limit=0.15,
        benchmark_ticker='SPY',
        alpha_weight=0.3,
        prediction_horizon=21  # 1 month for testing
    )
    
    # Test 1: First run (no cache)
    print("\n🏃 Test 1: First run (no cache)")
    print("-" * 30)
    
    start_time = time.time()
    
    backtest1 = AlphaEnhancedBacktest(
        alpha_config=alpha_config,
        feature_lookback=126,  # 6 months
        retrain_frequency=21,  # Monthly
        feature_set=FeatureSet.LITE,
        use_cache=True,
        parallel_config=ParallelConfig(enabled=False)
    )
    
    results1 = backtest1.run(
        prices=prices,
        volumes=volumes,
        start_date='2023-01-01',
        end_date='2023-12-31',
        max_train_years=1  # Very limited for testing
    )
    
    time1 = time.time() - start_time
    print(f"✅ First run completed in {time1:.1f} seconds")
    print(f"   Generated {len(results1['ml_metrics']['predictions'])} predictions")
    
    # Show updated cache info
    info = cache.get_cache_info()
    print(f"\n📊 Cache after first run:")
    print(f"   Models: {info['models']}")
    print(f"   Features: {info['features']}")
    print(f"   Predictions: {info['predictions']}")
    
    # Test 2: Second run (with cache)
    print("\n🏃 Test 2: Second run (with cache)")
    print("-" * 30)
    
    start_time = time.time()
    
    backtest2 = AlphaEnhancedBacktest(
        alpha_config=alpha_config,
        feature_lookback=126,
        retrain_frequency=21,
        feature_set=FeatureSet.LITE,
        use_cache=True,
        parallel_config=ParallelConfig(enabled=False)
    )
    
    results2 = backtest2.run(
        prices=prices,
        volumes=volumes,
        start_date='2023-01-01',
        end_date='2023-12-31',
        max_train_years=1
    )
    
    time2 = time.time() - start_time
    print(f"✅ Second run completed in {time2:.1f} seconds")
    print(f"   Speedup: {time1/time2:.1f}x faster!")
    
    # Test 3: Different parameters (partial cache hit)
    print("\n🏃 Test 3: Different date range (partial cache)")
    print("-" * 30)
    
    start_time = time.time()
    
    backtest3 = AlphaEnhancedBacktest(
        alpha_config=alpha_config,
        feature_lookback=126,
        retrain_frequency=21,
        feature_set=FeatureSet.LITE,
        use_cache=True,
        parallel_config=ParallelConfig(enabled=False)
    )
    
    results3 = backtest3.run(
        prices=prices,
        volumes=volumes,
        start_date='2023-06-01',  # Different start date
        end_date='2023-12-31',
        max_train_years=1
    )
    
    time3 = time.time() - start_time
    print(f"✅ Third run completed in {time3:.1f} seconds")
    
    # Clear specific cache type
    print("\n🗑️ Clearing predictions cache...")
    cache.clear_cache('predictions')
    
    info = cache.get_cache_info()
    print(f"\n📊 Cache after clearing predictions:")
    print(f"   Models: {info['models']} (kept)")
    print(f"   Features: {info['features']} (kept)")
    print(f"   Predictions: {info['predictions']} (cleared)")
    
    print("\n✅ ML Cache test completed!")
    print("\n💡 Tips:")
    print("   - Cache is stored in 'cache/ml_training/'")
    print("   - Delete this folder to clear all caches")
    print("   - Caching provides significant speedup for repeated runs")
    print("   - Models and features are cached separately")


if __name__ == "__main__":
    main() 