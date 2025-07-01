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
        
        Args:
            universe_data: dict of DataFrames: {'AAPL': df, 'MSFT': df}
                where each df has 'close' and 'volume' columns
            rebalance_dates: list of datetime objects
            
        Returns:
            dict: predictions for each rebalance date
        """
        all_predictions = {}
        
        for date in rebalance_dates:
            print(f"Training models for rebalance date: {date.date()}...")
            date_predictions = {}
            
            # Define the training window for this rebalance date
            train_end = date - timedelta(days=1)
            train_start = train_end - self.train_period

            for ticker, data in universe_data.items():
                try:
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
                    # Get features from the last available data before the rebalance date
                    pred_data = data[data.index <= train_end].tail(200)  # Need enough data for features
                    if len(pred_data) < 150:
                        continue
                        
                    pred_features = create_simple_features(pred_data['close'], pred_data['volume'])
                    latest_features = pred_features.iloc[[-1]]  # Last row of features
                    
                    if not latest_features.isnull().values.any():
                        prediction = model.predict(latest_features)
                        date_predictions[ticker] = prediction[0]
                        
                except Exception as e:
                    print(f"Error processing {ticker} for {date}: {str(e)}")
                    continue
            
            if date_predictions:
                self.predictions[date] = pd.Series(date_predictions)
                print(f"  Generated predictions for {len(date_predictions)} stocks")
                
        return self.predictions