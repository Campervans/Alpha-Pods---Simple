"""
ML Training Cache System.

Provides caching functionality for ML training results to avoid
recomputing expensive operations.
"""

import os
import pickle
import hashlib
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from pathlib import Path


class MLTrainingCache:
    """
    Cache for ML training results including:
    - Trained models
    - Feature engineering results
    - Predictions
    - Model metrics
    """
    
    def __init__(self, cache_dir: str = "cache/ml_training"):
        """Initialize ML cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-directories for different cache types
        self.models_dir = self.cache_dir / "models"
        self.features_dir = self.cache_dir / "features"
        self.predictions_dir = self.cache_dir / "predictions"
        self.metrics_dir = self.cache_dir / "metrics"
        
        for dir_path in [self.models_dir, self.features_dir, 
                         self.predictions_dir, self.metrics_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"📦 Initialized ML cache at: {self.cache_dir}")
    
    def get_cache_key(self, params: Dict[str, Any]) -> str:
        """
        Generate unique cache key from parameters.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            Unique hash key
        """
        # Convert params to stable string representation
        param_str = json.dumps(params, sort_keys=True, default=str)
        
        # Generate hash
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]
    
    def save_features(self, 
                     features: pd.DataFrame,
                     date_range: Tuple[str, str],
                     feature_params: Dict[str, Any],
                     feature_set: str) -> str:
        """
        Save engineered features to cache.
        
        Args:
            features: Feature DataFrame
            date_range: (start_date, end_date) tuple
            feature_params: Parameters used for feature engineering
            feature_set: Name of feature set used
            
        Returns:
            Cache key
        """
        # Create cache key
        cache_params = {
            'type': 'features',
            'date_range': date_range,
            'feature_params': feature_params,
            'feature_set': feature_set,
            'shape': features.shape
        }
        cache_key = self.get_cache_key(cache_params)
        
        # Save features
        features_path = self.features_dir / f"{cache_key}.pkl"
        features.to_pickle(features_path)
        
        # Save metadata
        meta_path = self.features_dir / f"{cache_key}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(cache_params, f, indent=2, default=str)
        
        print(f"💾 Saved features to cache: {cache_key}")
        return cache_key
    
    def load_features(self,
                     date_range: Tuple[str, str],
                     feature_params: Dict[str, Any],
                     feature_set: str) -> Optional[pd.DataFrame]:
        """
        Load features from cache if available.
        
        Returns:
            Cached features or None if not found
        """
        # Create cache key
        cache_params = {
            'type': 'features',
            'date_range': date_range,
            'feature_params': feature_params,
            'feature_set': feature_set
        }
        
        # Look for matching cache
        for meta_file in self.features_dir.glob("*_meta.json"):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            # Check if parameters match (ignoring shape)
            meta_check = meta.copy()
            meta_check.pop('shape', None)
            cache_check = cache_params.copy()
            
            if meta_check == cache_check:
                # Load features
                cache_key = meta_file.stem.replace('_meta', '')
                features_path = self.features_dir / f"{cache_key}.pkl"
                
                if features_path.exists():
                    features = pd.read_pickle(features_path)
                    print(f"✅ Loaded features from cache: {cache_key}")
                    return features
        
        return None
    
    def save_model(self,
                  model: Any,
                  train_date: pd.Timestamp,
                  train_params: Dict[str, Any],
                  metrics: Dict[str, float]) -> str:
        """
        Save trained model to cache.
        
        Args:
            model: Trained model
            train_date: Training date
            train_params: Training parameters
            metrics: Model performance metrics
            
        Returns:
            Cache key
        """
        # Create cache key
        cache_params = {
            'type': 'model',
            'train_date': str(train_date),
            'train_params': train_params,
            'metrics': metrics
        }
        cache_key = self.get_cache_key(cache_params)
        
        # Save model
        model_path = self.models_dir / f"{cache_key}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'train_date': train_date,
                'metrics': metrics
            }, f)
        
        # Save metadata
        meta_path = self.models_dir / f"{cache_key}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(cache_params, f, indent=2, default=str)
        
        return cache_key
    
    def load_model(self,
                  train_date: pd.Timestamp,
                  train_params: Dict[str, Any]) -> Optional[Tuple[Any, Dict[str, float]]]:
        """
        Load model from cache if available.
        
        Returns:
            Tuple of (model, metrics) or None if not found
        """
        # Look for matching model
        target_params = {
            'type': 'model',
            'train_date': str(train_date),
            'train_params': train_params
        }
        
        for meta_file in self.models_dir.glob("*_meta.json"):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            # Check if parameters match (ignoring metrics)
            meta_check = {k: v for k, v in meta.items() if k != 'metrics'}
            
            if meta_check == target_params:
                # Load model
                cache_key = meta_file.stem.replace('_meta', '')
                model_path = self.models_dir / f"{cache_key}.pkl"
                
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    print(f"✅ Loaded model from cache: {train_date}")
                    return data['model'], data['metrics']
        
        return None
    
    def save_predictions(self,
                        predictions: pd.DataFrame,
                        date_range: Tuple[str, str],
                        model_params: Dict[str, Any]) -> str:
        """
        Save model predictions to cache.
        """
        cache_params = {
            'type': 'predictions',
            'date_range': date_range,
            'model_params': model_params,
            'shape': predictions.shape
        }
        cache_key = self.get_cache_key(cache_params)
        
        # Save predictions
        pred_path = self.predictions_dir / f"{cache_key}.pkl"
        predictions.to_pickle(pred_path)
        
        # Save metadata
        meta_path = self.predictions_dir / f"{cache_key}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(cache_params, f, indent=2, default=str)
        
        print(f"💾 Saved predictions to cache: {cache_key}")
        return cache_key
    
    def load_predictions(self,
                        date_range: Tuple[str, str],
                        model_params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Load predictions from cache if available.
        """
        target_params = {
            'type': 'predictions',
            'date_range': date_range,
            'model_params': model_params
        }
        
        for meta_file in self.predictions_dir.glob("*_meta.json"):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            # Check if parameters match (ignoring shape)
            meta_check = {k: v for k, v in meta.items() if k != 'shape'}
            
            if meta_check == target_params:
                # Load predictions
                cache_key = meta_file.stem.replace('_meta', '')
                pred_path = self.predictions_dir / f"{cache_key}.pkl"
                
                if pred_path.exists():
                    predictions = pd.read_pickle(pred_path)
                    print(f"✅ Loaded predictions from cache")
                    return predictions
        
        return None
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache files.
        
        Args:
            cache_type: Type to clear ('models', 'features', 'predictions', 'metrics')
                       If None, clears all caches
        """
        if cache_type:
            target_dir = self.cache_dir / cache_type
            if target_dir.exists():
                for file in target_dir.glob("*"):
                    file.unlink()
                print(f"🗑️ Cleared {cache_type} cache")
        else:
            # Clear all caches
            for subdir in [self.models_dir, self.features_dir, 
                          self.predictions_dir, self.metrics_dir]:
                for file in subdir.glob("*"):
                    file.unlink()
            print("🗑️ Cleared all ML caches")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached items.
        
        Returns:
            Dictionary with cache statistics
        """
        info = {
            'cache_dir': str(self.cache_dir),
            'models': len(list(self.models_dir.glob("*.pkl"))),
            'features': len(list(self.features_dir.glob("*.pkl"))),
            'predictions': len(list(self.predictions_dir.glob("*.pkl"))),
            'metrics': len(list(self.metrics_dir.glob("*.pkl"))),
            'total_size_mb': 0
        }
        
        # Calculate total size
        for subdir in [self.models_dir, self.features_dir, 
                      self.predictions_dir, self.metrics_dir]:
            for file in subdir.glob("*"):
                info['total_size_mb'] += file.stat().st_size / (1024 * 1024)
        
        info['total_size_mb'] = round(info['total_size_mb'], 2)
        
        return info 