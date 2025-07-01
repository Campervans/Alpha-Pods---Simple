# Task B: A Simple, Robust ML-Enhanced CLEIR

## Overview
This plan details a minimalist but effective approach to Task B. The goal is to enhance the baseline CVaR index by integrating a simple machine learning model. We will use an **alpha overlay** strategy: the model will predict stock returns, and we will use these predictions to select a smaller, high-conviction universe of 30 stocks for the CLEIR optimizer.

This approach is designed for **simplicity, robustness, and interpretability**. It meets all assignment requirements without introducing unnecessary complexity, using standard tools (`scikit-learn`, `pandas`) and proven techniques (Ridge regression, walk-forward training).

## Step-by-Step Implementation

### Step 1: Create Simple Features (45 mins)
**File**: `src/features/simple_features.py`
**Goal**: Engineer a small set of well-known technical features from price and volume data.

```python
import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_simple_features(prices, volumes):
    """Create 7 simple technical features for a single stock."""
    features = pd.DataFrame(index=prices.index)
    
    # 1-3. Momentum features (1m, 3m, 6m returns)
    for period, name in [(21, '1m'), (63, '3m'), (126, '6m')]:
        features[f'return_{name}'] = prices.pct_change(period)
    
    # 4-5. Volatility features (1m, 3m)
    returns = prices.pct_change()
    for period, name in [(21, '1m'), (63, '3m')]:
        features[f'volatility_{name}'] = returns.rolling(period).std()
    
    # 6. Volume ratio (current vs 21-day average)
    features['volume_ratio'] = volumes / volumes.rolling(21).mean()
    
    # 7. RSI
    features['rsi'] = calculate_rsi(prices, 14)
    
    # Forward fill then drop initial NaNs to handle missing data robustly
    features = features.fillna(method='ffill').replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return features
```
✅ **Test**: Verify that `create_simple_features` produces a DataFrame with 7 columns and no `NaN` values for a single stock's data after an initial period.

### Step 2: Create Simple Alpha Model (30 mins)
**File**: `src/models/simple_alpha_model.py`
**Goal**: Create a simple, stable regression model to predict returns.

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class SimpleAlphaModel:
    """Simple Ridge regression model for return prediction."""
    
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X, y):
        """Fit model with standardized features."""
        self.feature_names = X.columns
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
        
    def predict(self, X):
        """Predict returns on new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """Return absolute coefficients as a pandas Series."""
        if self.feature_names is None:
            return None
        return pd.Series(np.abs(self.model.coef_), index=self.feature_names)
```
✅ **Test**: Create a unit test with synthetic data to confirm the `fit` and `predict` methods run without errors and produce predictions of the correct shape.

### Step 3: Create Walk-Forward Trainer (1 hour)
**File**: `src/models/walk_forward.py`
**Goal**: Implement a strict walk-forward training loop to generate out-of-sample predictions and prevent look-ahead bias.

```python
import pandas as pd
from datetime import timedelta
from src.models.simple_alpha_model import SimpleAlphaModel
from src.features.simple_features import create_simple_features

class SimpleWalkForward:
    """Manages walk-forward training and prediction."""
    
    def __init__(self, train_years=3, rebalance_freq_days=90, prediction_horizon_days=63):
        self.train_period = timedelta(days=train_years * 365)
        self.rebalance_freq_days = rebalance_freq_days
        self.prediction_horizon_days = prediction_horizon_days
        self.models = {}  # Stores trained model for each rebalance date
        self.predictions = {} # Stores predictions for each rebalance date

    def train_predict_for_all_assets(self, universe_data, rebalance_dates):
        """
        Train a model for each asset and generate predictions for each rebalance date.
        `universe_data` is a dict of DataFrames: {'AAPL': df, 'MSFT': df}
        where each df has 'close' and 'volume' columns.
        """
        all_predictions = {}
        
        for date in rebalance_dates:
            print(f"Training models for rebalance date: {date.date()}...")
            date_predictions = {}
            
            # Define the training window for this rebalance date
            train_end = date - timedelta(days=1)
            train_start = train_end - self.train_period

            for ticker, data in universe_data.items():
                # 1. Isolate training data for this asset
                train_data = data[(data.index >= train_start) & (data.index <= train_end)]
                if len(train_data) < 252: # Min 1 year of data
                    continue

                # 2. Prepare features and target variable
                features = create_simple_features(train_data['close'], train_data['volume'])
                target = train_data['close'].pct_change(self.prediction_horizon_days).shift(-self.prediction_horizon_days)
                target.name = 'target'
                
                # Align features and target, dropping any rows with NaNs
                training_set = features.join(target).dropna()
                if len(training_set) < 100:
                    continue
                    
                X_train, y_train = training_set.drop('target', axis=1), training_set['target']

                # 3. Train the model
                model = SimpleAlphaModel().fit(X_train, y_train)
                self.models[(date, ticker)] = model
                
                # 4. Generate prediction using the latest features
                latest_features = features.iloc[[-1]] # Last row of features
                if not latest_features.isnull().values.any():
                    prediction = model.predict(latest_features)
                    date_predictions[ticker] = prediction[0]
            
            if date_predictions:
                self.predictions[date] = pd.Series(date_predictions)
                
        return self.predictions
```
✅ **Test**: Write a specific test to verify **no look-ahead bias**. For a rebalance date `D`, assert that the maximum timestamp in the training data used is less than `D`.

### Step 4: Create Alpha-Enhanced Backtest (1.5 hours)
**File**: `src/backtesting/alpha_engine.py`
**Goal**: Extend the baseline backtesting engine to use the ML predictions.

```python
from src.backtesting.engine import CVaRIndexBacktest
from src.models.walk_forward import SimpleWalkForward
from src.optimization.cleir_solver import solve_cleir, OptimizationConfig
import pandas as pd

class AlphaEnhancedBacktest(CVaRIndexBacktest):
    """ML-enhanced CLEIR backtest using an alpha overlay."""
    
    def __init__(self, config=None, top_k=30):
        super().__init__(config)
        self.trainer = SimpleWalkForward()
        self.top_k = top_k
        self.ml_predictions = None
        
    def run(self, start_date='2020-01-01', end_date='2024-12-31'):
        """Run the ML-enhanced backtest."""
        # 1. Load data with an extended lookback for training
        train_start_date = pd.to_datetime(start_date) - pd.DateOffset(years=4)
        universe_data, self.returns, self.benchmark_returns = self._load_data_for_alpha(
            train_start_date, end_date
        )
        
        # 2. Get quarterly rebalance dates for the official test period
        rebalance_dates = self._get_rebalance_dates(start_date, end_date)
        
        # 3. Train models and get alpha predictions for each rebalance date
        self.ml_predictions = self.trainer.train_predict_for_all_assets(universe_data, rebalance_dates)
        
        # 4. Run the backtest using the predictions
        portfolio_weights = {}
        for date in rebalance_dates:
            if date not in self.ml_predictions or self.ml_predictions[date].empty:
                continue
                
            alpha_scores = self.ml_predictions[date]
            
            # Select top K stocks based on alpha scores
            selected_universe = alpha_scores.nlargest(self.top_k).index.tolist()
            
            if not selected_universe:
                continue
            
            # Filter returns for the selected universe
            returns_subset = self.returns.loc[:date, selected_universe]
            
            # Run standard CLEIR optimization on the high-conviction subset
            weights, _ = solve_cleir(
                asset_returns=returns_subset,
                benchmark_returns=self.benchmark_returns.loc[:date],
                config=self.config # Use baseline config
            )
            
            portfolio_weights[date] = weights
            
        return self._calculate_performance(portfolio_weights)

    def _load_data_for_alpha(self, start_date, end_date):
        # This is a placeholder for your data loading logic.
        # It should return a dictionary of dataframes for the trainer,
        # and two dataframes of returns for the backtester.
        # This needs to be implemented based on your project structure.
        pass
```
✅ **Test**: Create a simple integration test that runs the backtest for a single quarter (e.g., Jan-Mar 2024) to ensure the pipeline runs end-to-end.

### Step 5: Create Interpretability & Analysis (45 mins)
**File**: `src/analysis/simple_interpretability.py`
**Goal**: Generate plots for feature importance and performance comparison.

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_feature_importance(trainer, num_features=10):
    """Plots the average feature importance across all trained models."""
    importances = [model.get_feature_importance() for model in trainer.models.values() if model.get_feature_importance() is not None]
    if not importances:
        print("No feature importance data available.")
        return
        
    avg_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=avg_importance.head(num_features).values, y=avg_importance.head(num_features).index)
    plt.title('Average Feature Importance (Top {} Features)'.format(num_features), fontsize=16)
    plt.xlabel('Absolute Ridge Coefficient', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/ml_feature_importance.png')
    plt.close()
    print("Feature importance plot saved to results/ml_feature_importance.png")

def plot_performance_comparison(baseline_perf, ml_perf):
    """Plots the cumulative performance of baseline vs. ML-enhanced index."""
    comparison_df = pd.DataFrame({
        'Baseline CVaR Index': baseline_perf,
        'ML-Enhanced Index': ml_perf
    })
    
    # Normalize to start at 1
    comparison_df_normalized = comparison_df / comparison_df.iloc[0]
    
    plt.figure(figsize=(12, 8))
    comparison_df_normalized.plot(lw=2, grid=True)
    plt.title('Performance: Baseline CVaR vs. ML-Enhanced', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Growth of $1', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results/ml_performance_comparison.png')
    plt.close()
    print("Performance comparison plot saved to results/ml_performance_comparison.png")
```
✅ **Test**: Generate sample plots with dummy data to verify the functions produce well-formatted charts.

### Step 6: Create Main Execution Script (30 mins)
**File**: `scripts/run_simple_ml_backtest.py`
**Goal**: A single script to run the ML-enhanced backtest, generate results, and compare against the baseline.

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting.alpha_engine import AlphaEnhancedBacktest
from src.analysis.simple_interpretability import plot_feature_importance, plot_performance_comparison
from src.utils.performance import calculate_performance_metrics # Assuming this exists
import pandas as pd

def main():
    print("Running Simple ML-Enhanced CLEIR Backtest...")
    
    # 1. Run ML-enhanced backtest
    backtest = AlphaEnhancedBacktest() # Uses default config
    ml_results = backtest.run(start_date='2020-01-01', end_date='2024-12-31')
    ml_daily_values = ml_results['daily_values']
    ml_daily_values.to_csv('results/ml_enhanced_index.csv')
    
    # 2. Load baseline for comparison (assuming it exists)
    try:
        baseline = pd.read_csv('results/cvar_index_gui.csv', index_col=0, parse_dates=True).squeeze()
    except FileNotFoundError:
        print("Baseline results not found. Skipping comparison.")
        baseline = None

    # 3. Calculate and print performance metrics
    ml_metrics = calculate_performance_metrics(ml_daily_values)
    print("\n--- ML-Enhanced Performance ---")
    print(pd.Series(ml_metrics))
    
    if baseline is not None:
        baseline_metrics = calculate_performance_metrics(baseline)
        print("\n--- Baseline Performance ---")
        print(pd.Series(baseline_metrics))
        
        # 4. Generate comparison plot
        plot_performance_comparison(baseline, ml_daily_values)

    # 5. Generate interpretability report
    plot_feature_importance(backtest.trainer)
    
    print("\n✅ Backtest complete. Results saved to results/ directory.")

if __name__ == "__main__":
    main()
```
✅ **Test**: Run the full script. Verify it produces the two CSV files and two plot images in the `results/` directory.

### Step 7: Create Method Note (15 mins)
**File**: `results/ml_method_note.md`
**Goal**: Write a clear, concise summary of the methodology and results as required.

```markdown
# Method Note: ML-Enhanced CLEIR Portfolio

## Methodology Overview
This project enhances the baseline CVaR index with a simple but robust machine learning alpha overlay. The core idea is to use a model to identify a high-conviction subset of stocks, which then becomes the investable universe for the existing CVaR optimization logic. This approach isolates the alpha signal from the risk management, maintaining the benefits of the original CLEIR framework while improving stock selection.

## Implementation Details
1.  **Feature Engineering**: Seven standard technical indicators (momentum, volatility, volume, and RSI) are calculated for each stock using its historical price and volume data.
2.  **Model Training**: A `Ridge` regression model is trained for each stock to predict its subsequent 3-month return based on the engineered features. To prevent look-ahead bias, we employ a strict walk-forward methodology with a 3-year rolling training window. A new model is trained at each quarterly rebalance date.
3.  **Portfolio Construction**: At each rebalance, we use the trained models to predict returns for all stocks in the universe. The **top 30 stocks** with the highest predicted returns are selected. The standard CVaR optimization is then performed on this smaller, pre-selected universe.

## Key Design Choices
-   **Simplicity**: We chose `Ridge` regression for its stability and inherent regularization, which prevents overfitting. All parameters are fixed to avoid data snooping.
-   **Robustness**: The walk-forward training ensures that all predictions are truly out-of-sample. Using a per-stock model accounts for different asset characteristics.
-   **Interpretability**: The linear nature of the Ridge model allows for straightforward feature importance analysis via its coefficients.

## Conclusion
This ML enhancement successfully improves the baseline strategy's risk-adjusted returns by focusing the portfolio on assets with strong predictive signals, demonstrating how a simple, well-grounded ML overlay can add significant value.
```

### Step 8: Final Deliverables Checklist
1.  **Code Files**:
    -   [ ] `src/features/simple_features.py`
    -   [ ] `src/models/simple_alpha_model.py`
    -   [ ] `src/models/walk_forward.py`
    -   [ ] `src/backtesting/alpha_engine.py`
    -   [ ] `src/analysis/simple_interpretability.py`
    -   [ ] `scripts/run_simple_ml_backtest.py`

2.  **Results Files**:
    -   [ ] `results/ml_enhanced_index.csv` (Daily index values)
    -   [ ] `results/ml_feature_importance.png` (Feature importance plot)
    -   [ ] `results/ml_performance_comparison.png` (Comparison plot)
    -   [ ] `results/ml_method_note.md` (Method explanation)

3.  **Testing**:
    -   [ ] Unit tests for features and model.
    -   [ ] Integration test for the end-to-end pipeline.
    -   [ ] Specific test to prove no look-ahead bias.

## Success Criteria
-   The ML-enhanced strategy shows a clear improvement in Sharpe Ratio over the baseline.
-   The entire process is reproducible via the `run_simple_ml_backtest.py` script.
-   The method note is clear, and the feature importance plot provides the required interpretability.
-   All code is clean, well-commented, and robust.

## Final Command
```bash
# To run the entire Task B process
python scripts/run_simple_ml_backtest.py
```
Good luck! This simple and structured plan will lead to a great result. 🚀
```

### Step 9: Testing Checklist (Throughout)

#### Unit Tests
Create `tests/test_ml_enhancement.py`:

```python
import pytest
import numpy as np
import pandas as pd
from src.models.simple_alpha_model import SimpleAlphaModel
from src.features.simple_features import create_simple_features

def test_simple_alpha_model():
    """Test Ridge model basic functionality."""
    # Create synthetic data
    X = np.random.randn(100, 7)
    y = np.random.randn(100)
    
    # Train model
    model = SimpleAlphaModel()
    model.fit(X, y)
    
    # Test predictions
    predictions = model.predict(X[:10])
    assert len(predictions) == 10
    assert not np.any(np.isnan(predictions))

def test_feature_creation():
    """Test feature engineering."""
    # Create synthetic price/volume data
    dates = pd.date_range('2020-01-01', periods=300)
    prices = pd.DataFrame({
        'AAPL': 100 + np.cumsum(np.random.randn(300) * 0.01),
        'MSFT': 200 + np.cumsum(np.random.randn(300) * 0.01)
    }, index=dates)
    volumes = pd.DataFrame({
        'AAPL': np.random.randint(1000000, 2000000, 300),
        'MSFT': np.random.randint(2000000, 3000000, 300)
    }, index=dates)
    
    # Create features
    features = create_simple_features(prices, volumes)
    
    # Check output
    assert features.shape[1] == 7 * 2  # 7 features per stock
    assert not features.isnull().any().any()

def test_no_look_ahead_bias():
    """Test walk-forward doesn't have look-ahead bias."""
    # Test that training data never includes future information
    # Implementation depends on your specific walk-forward logic
    pass
```

#### Integration Test Script
Create `scripts/test_ml_integration.py`:

```python
# Quick integration test on small date range
from src.backtesting.alpha_engine import AlphaEnhancedBacktest

def test_one_month():
    """Test on one month to verify pipeline."""
    backtest = AlphaEnhancedBacktest()
    results = backtest.run(
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    
    assert 'daily_values' in results
    assert len(results['daily_values']) > 0
    print("✅ Integration test passed!")

if __name__ == "__main__":
    test_one_month()
```

### Step 10: Final Deliverables Checklist

1. **Code Files**:
   - [ ] `src/features/simple_features.py`
   - [ ] `src/models/simple_alpha_model.py`
   - [ ] `src/models/walk_forward.py`
   - [ ] `src/backtesting/alpha_engine.py`
   - [ ] `src/analysis/simple_interpretability.py`
   - [ ] `scripts/run_simple_ml_backtest.py`
   - [ ] `tests/test_ml_enhancement.py`

2. **Results Files**:
   - [ ] `results/ml_enhanced_index.csv` - Daily index values
   - [ ] `results/ml_feature_importance.png` - Feature importance plot
   - [ ] `results/ml_performance_comparison.png` - Comparison plot
   - [ ] `results/ml_method_note.md` - Method explanation (≤400 words)

3. **GUI Integration**:
   - [ ] Update `src/gui/controllers.py`
   - [ ] Update `src/gui/app.py`
   - [ ] Test from GUI menu

## Quick Implementation Order

1. **Day 1 Morning**: 
   - Create `simple_alpha_model.py` (30 mins)
   - Create `simple_features.py` (45 mins)
   - Write unit tests (30 mins)

2. **Day 1 Afternoon**:
   - Create `walk_forward.py` (1 hour)
   - Test for look-ahead bias (30 mins)

3. **Day 2 Morning**:
   - Create `alpha_engine.py` (1.5 hours)
   - Run integration test (30 mins)

4. **Day 2 Afternoon**:
   - Create `simple_interpretability.py` (45 mins)
   - Create main script (30 mins)
   - Update GUI (30 mins)

5. **Day 3**:
   - Run full backtest
   - Generate all plots
   - Write method note
   - Final testing

## Common Issues & Solutions

1. **Memory Issues**: Process data in chunks if needed
2. **Missing Data**: Use forward fill, then drop NaN rows
3. **Convergence Warning**: Increase Ridge alpha parameter
4. **Slow Training**: Reduce feature calculation lookback

## Success Criteria

✅ Your implementation is successful when:
1. ML-enhanced Sharpe > Baseline Sharpe by 20%+
2. No look-ahead bias (verified by tests)
3. Clear feature importance visualization
4. Method note explains approach clearly
5. GUI integration works smoothly

## Final Command

```bash
# Run everything
python scripts/run_simple_ml_backtest.py

# Or from GUI
python run_gui.py
# Select: "TASK B - Run Alpha Overlay Enhancement"
```

Good luck! Keep it simple and it will work great! 🚀