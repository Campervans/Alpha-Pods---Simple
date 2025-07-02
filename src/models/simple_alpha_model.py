from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class SimpleAlphaModel:
    """Simple Ridge regression model for predicting returns."""
    
    def __init__(self, alpha=1.0):
        # TODO: experiment with other alpha values
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train_ = None  # store training data for SHAP
        
    def fit(self, X, y):
        """fit model with standardized features.
        
        Args:
            X: pd.DataFrame of features
            y: pd.Series of target returns
            
        Returns:
            self
        """
        self.feature_names = X.columns.tolist()
        self.X_train_ = X.copy()  # store original training data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
        
    def predict(self, X):
        """predict returns on new data.
        
        Args:
            X: pd.DataFrame of features
            
        Returns:
            np.array of predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """return absolute coefficients as a pandas Series."""
        if self.feature_names is None:
            return None
        return pd.Series(np.abs(self.model.coef_), index=self.feature_names)