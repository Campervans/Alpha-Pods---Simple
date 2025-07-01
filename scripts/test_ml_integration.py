#!/usr/bin/env python3
"""Quick integration test for ML enhancement on small date range."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

# Try imports
try:
    from src.models.simple_alpha_model import SimpleAlphaModel
    from src.features.simple_features import create_simple_features
    print("✅ Successfully imported ML modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic ML functionality with synthetic data."""
    print("\n🧪 Testing basic ML functionality...")
    
    # Create synthetic data
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    # Single stock test
    prices = pd.Series(100 + np.cumsum(np.random.randn(500) * 0.5), index=dates)
    volumes = pd.Series(np.random.randint(1000000, 2000000, 500), index=dates)
    
    # Test feature creation
    print("  Testing feature creation...")
    features = create_simple_features(prices, volumes)
    print(f"  ✅ Created {features.shape[1]} features for {features.shape[0]} days")
    print(f"  Features: {list(features.columns)}")
    
    # Test model training
    print("\n  Testing model training...")
    # Create target (3-month forward returns)
    target = prices.pct_change(63).shift(-63)
    
    # Align and clean data
    train_data = pd.concat([features, target], axis=1).dropna()
    if len(train_data) < 100:
        print("  ⚠️  Not enough data for training")
        return False
    
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    
    # Train model
    model = SimpleAlphaModel(alpha=1.0)
    model.fit(X_train, y_train)
    print(f"  ✅ Trained model on {len(X_train)} samples")
    
    # Test prediction
    print("\n  Testing prediction...")
    test_features = features.iloc[-10:].dropna()
    if len(test_features) > 0:
        predictions = model.predict(test_features)
        print(f"  ✅ Generated {len(predictions)} predictions")
        print(f"  Sample predictions: {predictions[:3]}")
    
    # Test feature importance
    print("\n  Testing feature importance...")
    importance = model.get_feature_importance()
    if importance is not None:
        print(f"  ✅ Got feature importance for {len(importance)} features")
        print(f"  Top 3 features: {importance.nlargest(3).to_dict()}")
    
    return True

def test_ml_backtest_minimal():
    """Test ML backtest on minimal date range."""
    print("\n🧪 Testing minimal ML backtest...")
    
    try:
        from src.backtesting.alpha_engine import AlphaEnhancedBacktest
        
        # Run on very short period
        backtest = AlphaEnhancedBacktest(top_k=10)  # Smaller universe for test
        
        # Check if we can at least initialize
        print("  ✅ AlphaEnhancedBacktest initialized successfully")
        
        # Test configuration
        config = backtest.config
        print(f"  Config: confidence_level={config.confidence_level}, "
              f"sparsity_bound={config.sparsity_bound}, "
              f"lookback_days={config.lookback_days}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

def main():
    """Run integration tests."""
    print("=" * 60)
    print("🚀 ML Enhancement Integration Test")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    test_results = []
    
    # Test 1: Basic functionality
    result1 = test_basic_functionality()
    test_results.append(("Basic ML Functionality", result1))
    
    # Test 2: Minimal backtest
    result2 = test_ml_backtest_minimal()
    test_results.append(("Minimal Backtest Setup", result2))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())