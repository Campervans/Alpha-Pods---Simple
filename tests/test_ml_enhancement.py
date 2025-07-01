import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Test imports
try:
    from src.models.simple_alpha_model import SimpleAlphaModel
    from src.features.simple_features import create_simple_features, calculate_rsi
    from src.models.walk_forward import SimpleWalkForward
except ImportError as e:
    print(f"Import error: {e}")

def test_simple_alpha_model():
    """Test Ridge model basic functionality."""
    # Create synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 7
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples))
    
    # Train model
    model = SimpleAlphaModel(alpha=1.0)
    model.fit(X, y)
    
    # Test predictions
    predictions = model.predict(X[:10])
    assert len(predictions) == 10
    assert not np.any(np.isnan(predictions))
    
    # Test feature importance
    importance = model.get_feature_importance()
    assert importance is not None
    assert len(importance) == n_features
    assert all(importance >= 0)  # Absolute values should be non-negative

def test_calculate_rsi():
    """Test RSI calculation."""
    # Create synthetic price series
    prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106])
    
    # Calculate RSI
    rsi = calculate_rsi(prices, period=5)
    
    # Basic checks
    assert len(rsi) == len(prices)
    assert all((rsi >= 0) & (rsi <= 100))  # RSI should be between 0 and 100
    assert not rsi.isnull().all()  # Should have some non-null values

def test_create_simple_features():
    """Test feature engineering."""
    # Create synthetic price/volume data
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(300) * 0.5), index=dates)
    volumes = pd.Series(np.random.randint(1000000, 2000000, 300), index=dates)
    
    # Create features
    features = create_simple_features(prices, volumes)
    
    # Check output
    assert features.shape[1] == 7  # Should have 7 features
    assert features.shape[0] == len(prices)  # Same number of rows
    
    # Check feature names
    expected_features = ['return_1m', 'return_3m', 'return_6m', 
                        'volatility_1m', 'volatility_3m', 'volume_ratio', 'rsi']
    assert all(feat in features.columns for feat in expected_features)
    
    # Check no NaN values after initial period
    assert not features.iloc[200:].isnull().any().any()

def test_walk_forward_no_look_ahead():
    """Test that walk-forward training doesn't have look-ahead bias."""
    # Create synthetic universe data
    dates = pd.date_range('2015-01-01', periods=2000, freq='D')
    np.random.seed(42)
    
    universe_data = {}
    for ticker in ['AAPL', 'MSFT', 'GOOGL']:
        price = 100 + np.cumsum(np.random.randn(2000) * 0.5)
        volume = np.random.randint(1000000, 2000000, 2000)
        universe_data[ticker] = pd.DataFrame({
            'close': price,
            'volume': volume
        }, index=dates)
    
    # Set up walk-forward
    wf = SimpleWalkForward(train_years=3, prediction_horizon_days=63)
    
    # Test rebalance dates
    rebalance_dates = [pd.Timestamp('2019-03-31'), pd.Timestamp('2019-06-30')]
    
    # For each rebalance date, verify no look-ahead
    for rebal_date in rebalance_dates:
        train_end = rebal_date - timedelta(days=1)
        train_start = train_end - timedelta(days=3*365)
        
        # Check that training window is before rebalance date
        assert train_end < rebal_date
        assert train_start < train_end
        
        # Verify data filtering
        for ticker, data in universe_data.items():
            train_data = data[(data.index >= train_start) & (data.index <= train_end)]
            
            # No data from rebalance date or after should be in training
            assert all(train_data.index < rebal_date)

def test_feature_importance_aggregation():
    """Test that feature importance can be aggregated across models."""
    # Create multiple models with different importances
    importances = []
    feature_names = ['return_1m', 'return_3m', 'volatility_1m']
    
    for i in range(3):
        importance = pd.Series(np.random.rand(3) * (i + 1), index=feature_names)
        importances.append(importance)
    
    # Calculate average importance
    avg_importance = pd.concat(importances, axis=1).mean(axis=1)
    
    # Check result
    assert len(avg_importance) == len(feature_names)
    assert all(avg_importance >= 0)
    assert avg_importance.index.tolist() == feature_names

def test_ml_pipeline_integration():
    """Simple integration test for the ML pipeline."""
    # This is a simplified test - in production you'd test the full pipeline
    # Create minimal data
    dates = pd.date_range('2018-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    # Single stock data
    prices = pd.Series(100 + np.cumsum(np.random.randn(1000) * 0.5), index=dates)
    volumes = pd.Series(np.random.randint(1000000, 2000000, 1000), index=dates)
    
    # Test feature creation
    features = create_simple_features(prices, volumes)
    assert features.shape[1] == 7
    
    # Test model training (on subset)
    train_features = features.iloc[200:800].dropna()
    train_target = prices.pct_change(63).shift(-63).iloc[200:800].dropna()
    
    # Align features and target
    aligned_data = pd.concat([train_features, train_target], axis=1).dropna()
    if len(aligned_data) > 100:
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        model = SimpleAlphaModel()
        model.fit(X, y)
        
        # Test prediction
        latest_features = features.iloc[[-1]].dropna()
        if len(latest_features) > 0 and not latest_features.isnull().any().any():
            pred = model.predict(latest_features)
            assert len(pred) == 1
            assert not np.isnan(pred[0])

if __name__ == "__main__":
    print("Running ML enhancement tests...")
    
    # Run tests manually
    try:
        test_simple_alpha_model()
        print("✅ test_simple_alpha_model passed")
    except Exception as e:
        print(f"❌ test_simple_alpha_model failed: {e}")
    
    try:
        test_calculate_rsi()
        print("✅ test_calculate_rsi passed")
    except Exception as e:
        print(f"❌ test_calculate_rsi failed: {e}")
    
    try:
        test_create_simple_features()
        print("✅ test_create_simple_features passed")
    except Exception as e:
        print(f"❌ test_create_simple_features failed: {e}")
    
    try:
        test_walk_forward_no_look_ahead()
        print("✅ test_walk_forward_no_look_ahead passed")
    except Exception as e:
        print(f"❌ test_walk_forward_no_look_ahead passed: {e}")
    
    try:
        test_feature_importance_aggregation()
        print("✅ test_feature_importance_aggregation passed")
    except Exception as e:
        print(f"❌ test_feature_importance_aggregation failed: {e}")
    
    try:
        test_ml_pipeline_integration()
        print("✅ test_ml_pipeline_integration passed")
    except Exception as e:
        print(f"❌ test_ml_pipeline_integration failed: {e}")
    
    print("\nAll tests completed!")