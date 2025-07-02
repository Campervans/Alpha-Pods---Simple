import pandas as pd
from datetime import timedelta
from src.models.simple_alpha_model import SimpleAlphaModel
from src.features.simple_features import create_simple_features

class SimpleWalkForward:
    """manages walk-forward training and prediction."""
    
    def __init__(self, train_years=3, rebalance_freq_days=90, prediction_horizon_days=63, 
                 fixed_train_start='2014-01-01', fixed_train_end='2019-12-31'):
        self.train_period = timedelta(days=train_years * 365)
        self.rebalance_freq_days = rebalance_freq_days
        self.prediction_horizon_days = prediction_horizon_days
        self.models = {}  # stores trained model for each rebalance date
        self.predictions = {} # stores predictions for each rebalance date
        self.fixed_train_start = pd.to_datetime(fixed_train_start)
        self.fixed_train_end = pd.to_datetime(fixed_train_end)

    def train_predict_for_all_assets(self, universe_data, rebalance_dates):

        all_predictions = {}
        
        for date in rebalance_dates:
            print(f"Training models for rebalance date: {date.date()}...")
            date_predictions = {}
            
            # define training window
            # for dates in 2020+, use fixed 2014-2019 training period
            if date >= pd.to_datetime('2020-01-01'):
                train_start = self.fixed_train_start
                train_end = self.fixed_train_end
            else:
                # for earlier dates, use rolling window
                # TODO: this part is not used in the current setup, might be dead code
                train_end = date - timedelta(days=1)
                train_start = train_end - self.train_period

            for ticker, data in universe_data.items():
                try:
                    # 1. isolate training data
                    train_data = data[(data.index >= train_start) & (data.index <= train_end)]
                    if len(train_data) < 252: # min 1 year of data
                        continue

                    # 2. prepare features and target
                    features = create_simple_features(train_data['close'], train_data['volume'])
                    target = train_data['close'].pct_change(self.prediction_horizon_days).shift(-self.prediction_horizon_days)
                    target.name = 'target'
                    
                    # align features and target, drop NaNs
                    training_set = features.join(target).dropna()
                    if len(training_set) < 100:
                        continue
                        
                    X_train, y_train = training_set.drop('target', axis=1), training_set['target']

                    # 3. train model
                    model = SimpleAlphaModel().fit(X_train, y_train)
                    self.models[(date, ticker)] = model
                    
                    # 4. generate prediction using latest features
                    pred_data = data[data.index <= train_end].tail(200)  # need enough data for features
                    if len(pred_data) < 150:
                        continue
                        
                    pred_features = create_simple_features(pred_data['close'], pred_data['volume'])
                    latest_features = pred_features.iloc[[-1]]  # last row
                    
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