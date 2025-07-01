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
        """Fit model with standardized features.
        
        Args:
            X: pd.DataFrame of features
            y: pd.Series of target returns
            
        Returns:
            self
        """
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
        
    def predict(self, X):
        """Predict returns on new data.
        
        Args:
            X: pd.DataFrame of features
            
        Returns:
            np.array of predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """Return absolute coefficients as a pandas Series."""
        if self.feature_names is None:
            return None
        return pd.Series(np.abs(self.model.coef_), index=self.feature_names)