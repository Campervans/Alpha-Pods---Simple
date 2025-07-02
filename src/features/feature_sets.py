"""
Feature set definitions for Alpha Enhancement.

defines different feature sets (Lite, Standard, Full)
to trade off compute time and model complexity.
"""

from enum import Enum
from typing import List, Dict, Any, Set, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)


class FeatureSet(Enum):
    """Feature set options."""
    LITE = "lite"          # top 20-30 features
    STANDARD = "standard"  # top 50-100 features  
    FULL = "full"          # all 698+ features


# predefined feature sets from financial literature and empirical results
FEATURE_SETS = {
    FeatureSet.LITE: {
        # core momentum features (most predictive)
        'price_features': [
            'momentum_1m', 'momentum_3m', 'momentum_6m',
            'volatility_30d', 'sharpe_30d',
            'rsi', 'bb_position'
        ],
        'volume_features': [
            'dollar_volume_mean_30d',
            'volume_momentum_5d'
        ],
        'fundamental_features': [
            'pe_ratio', 'pb_ratio', 'roe'
        ],
        'market_features': [
            'vix_level', 'market_regime'
        ],
        # total: ~20 features per ticker
    },
    
    FeatureSet.STANDARD: {
        # extended momentum and technicals
        'price_features': [
            'momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m',
            'volatility_30d', 'volatility_60d', 'volatility_90d',
            'sharpe_30d', 'sharpe_60d', 'sharpe_90d',
            'rsi', 'bb_position', 'bb_width',
            'high_52w_ratio', 'low_52w_ratio',
            'price_acceleration', 'volatility_change'
        ],
        'volume_features': [
            'dollar_volume_mean_30d', 'dollar_volume_mean_60d',
            'volume_momentum_5d', 'volume_momentum_20d',
            'volume_volatility_30d',
            'relative_volume'
        ],
        'fundamental_features': [
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio',
            'roe', 'roa', 'debt_to_equity',
            'earnings_growth', 'revenue_growth'
        ],
        'market_features': [
            'vix_level', 'vix_percentile', 'vix_change',
            'treasury_yield_10y', 'yield_curve_slope',
            'dollar_index', 'oil_price',
            'market_regime', 'trend_strength'
        ],
        # total: ~50-60 features per ticker
    },
    
    FeatureSet.FULL: {
        # all available features
        # TODO: this is a bit of a kitchen sink, should probably be more selective
        'price_features': 'all',
        'volume_features': 'all',
        'fundamental_features': 'all',
        'market_features': 'all',
        # total: 698+ features
    }
}


class FeatureSelector:
    """intelligent feature selection based on importance."""
    
    def __init__(self, feature_set: FeatureSet = FeatureSet.STANDARD):
        """
        init feature selector.
        
        Parameters
        ----------
        feature_set : FeatureSet
            which feature set to use
        """
        self.feature_set = feature_set
        self.feature_importance_ = None
        self.selected_features_ = None
        
    def get_feature_list(self) -> Dict[str, List[str]]:
        """
        get the list of features for the selected set.
        
        Returns
        -------
        Dict[str, List[str]]
            dict of feature categories and their features
        """
        return FEATURE_SETS[self.feature_set]
    
    def filter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        filter dataframe to only include features from the selected set.
        
        Parameters
        ----------
        df : pd.DataFrame
            full feature dataframe
            
        Returns
        -------
        pd.DataFrame
            filtered dataframe
        """
        if self.feature_set == FeatureSet.FULL:
            return df
        
        feature_config = FEATURE_SETS[self.feature_set]
        selected_columns = ['date', 'ticker']  # always keep these
        
        # add features from each category
        for category, features in feature_config.items():
            if features == 'all':
                # include all features from this category
                category_prefix = category.split('_')[0]  # e.g., 'price' from 'price_features'
                category_cols = [col for col in df.columns if category_prefix in col.lower()]
                selected_columns.extend(category_cols)
            else:
                # include specific features
                for feature in features:
                    # find columns that match this feature pattern
                    matching_cols = [col for col in df.columns if feature in col]
                    selected_columns.extend(matching_cols)
        
        # remove duplicates and filter
        selected_columns = list(set(selected_columns))
        available_columns = [col for col in selected_columns if col in df.columns]
        
        logger.info(
            f"Feature set '{self.feature_set.value}': "
            f"selected {len(available_columns)} out of {len(df.columns)} features"
        )
        
        return df[available_columns]
    
    def compute_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'mutual_info'
    ) -> pd.Series:
        """
        Compute feature importance scores.
        
        Parameters
        ----------
        X : pd.DataFrame
            feature matrix
        y : pd.Series
            target variable
        method : str
            method to use ('mutual_info' or 'random_forest')
            
        Returns
        -------
        pd.Series
            feature importance scores
        """
        # remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        if method == 'mutual_info':
            # use mutual information
            importance = mutual_info_regression(X_numeric, y, random_state=42)
            importance_series = pd.Series(importance, index=numeric_cols)
        
        elif method == 'random_forest':
            # use random forest feature importance
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_numeric, y)
            importance_series = pd.Series(
                rf.feature_importances_,
                index=numeric_cols
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # sort by importance
        importance_series = importance_series.sort_values(ascending=False)
        self.feature_importance_ = importance_series
        
        return importance_series
    
    def select_top_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Select top features based on importance.
        
        Parameters
        ----------
        X : pd.DataFrame
            feature matrix
        y : pd.Series
            target variable
        n_features : int, optional
            number of top features to select
        threshold : float, optional
            min importance threshold
            
        Returns
        -------
        List[str]
            selected feature names
        """
        if self.feature_importance_ is None:
            self.compute_feature_importance(X, y)
        
        importance = self.feature_importance_
        
        if n_features is not None:
            # select top N features
            selected = importance.nlargest(n_features).index.tolist()
        elif threshold is not None:
            # select features above threshold
            selected = importance[importance > threshold].index.tolist()
        else:
            # default: select based on feature set
            if self.feature_set == FeatureSet.LITE:
                n_features = min(30, len(importance))
            elif self.feature_set == FeatureSet.STANDARD:
                n_features = min(100, len(importance))
            else:
                n_features = len(importance)
            
            selected = importance.nlargest(n_features).index.tolist()
        
        self.selected_features_ = selected
        logger.info(f"Selected {len(selected)} features based on importance")
        
        return selected
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        transform dataframe to only include selected features.
        
        Parameters
        ----------
        X : pd.DataFrame
            feature matrix
            
        Returns
        -------
        pd.DataFrame
            transformed dataframe
        """
        if self.selected_features_ is None:
            # use predefined feature set
            return self.filter_features(X)
        else:
            # use dynamically selected features
            available_features = [f for f in self.selected_features_ if f in X.columns]
            
            # always include date and ticker if present
            meta_cols = [col for col in ['date', 'ticker'] if col in X.columns]
            all_cols = meta_cols + available_features
            
            return X[all_cols]


def get_feature_descriptions() -> Dict[str, str]:
    """
    Get human-readable descriptions of features.
    
    Returns
    -------
    Dict[str, str]
        feature descriptions
    """
    return {
        # price features
        'momentum_1m': '1-month price momentum',
        'momentum_3m': '3-month price momentum',
        'momentum_6m': '6-month price momentum',
        'momentum_12m': '12-month price momentum',
        'volatility_30d': '30-day price volatility',
        'volatility_60d': '60-day price volatility',
        'volatility_90d': '90-day price volatility',
        'sharpe_30d': '30-day Sharpe ratio',
        'sharpe_60d': '60-day Sharpe ratio',
        'sharpe_90d': '90-day Sharpe ratio',
        'rsi': 'Relative Strength Index',
        'bb_position': 'Position within Bollinger Bands',
        'bb_width': 'Bollinger Band width',
        'high_52w_ratio': 'Price relative to 52-week high',
        'low_52w_ratio': 'Price relative to 52-week low',
        
        # volume features
        'dollar_volume_mean_30d': '30-day average dollar volume',
        'volume_momentum_5d': '5-day volume momentum',
        'volume_volatility_30d': '30-day volume volatility',
        'relative_volume': 'Volume relative to average',
        
        # fundamental features
        'pe_ratio': 'Price-to-Earnings ratio',
        'pb_ratio': 'Price-to-Book ratio',
        'ps_ratio': 'Price-to-Sales ratio',
        'pcf_ratio': 'Price-to-Cash Flow ratio',
        'roe': 'Return on Equity',
        'roa': 'Return on Assets',
        'debt_to_equity': 'Debt-to-Equity ratio',
        
        # market features
        'vix_level': 'VIX (volatility index) level',
        'vix_percentile': 'VIX percentile rank',
        'treasury_yield_10y': '10-year Treasury yield',
        'yield_curve_slope': 'Yield curve slope (10Y-2Y)',
        'dollar_index': 'US Dollar Index level',
        'market_regime': 'Market regime classification',
    }