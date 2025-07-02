"""
ML Training Cache System.

provides caching for ML training results to avoid
recomputing expensive stuff.
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
    - trained models
    - feature engineering results
    - predictions
    - model metrics
    """
    
    def __init__(self, cache_dir: str = "cache/ml_training"):
        """init ML cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # sub-directories for different cache types
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
        Generate unique cache key from params.
        
        Args:
            params: dict of parameters
            
        Returns:
            unique hash key
        """
        # convert params to stable string
        param_str = json.dumps(params, sort_keys=True, default=str)
        
        # generate hash
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]
    
    def save_features(self, 
                     features: pd.DataFrame,
                     date_range: Tuple[str, str],
                     feature_params: Dict[str, Any],
                     feature_set: str) -> str:
        """
        Save engineered features to cache.
        
        Args:
            features: feature DataFrame
            date_range: (start, end) tuple
            feature_params: params for feature engineering
            feature_set: name of feature set
            
        Returns:
            cache key
        """
        # create cache key
        cache_params = {
            'type': 'features',
            'date_range': date_range,
            'feature_params': feature_params,
            'feature_set': feature_set,
            'shape': features.shape
        }
        cache_key = self.get_cache_key(cache_params)
        
        # save features
        features_path = self.features_dir / f"{cache_key}.pkl"
        features.to_pickle(features_path)
        
        # save metadata
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
        Load features from cache.
        
        Returns:
            cached features or None
        """
        # create cache key
        cache_params = {
            'type': 'features',
            'date_range': date_range,
            'feature_params': feature_params,
            'feature_set': feature_set
        }
        
        # look for matching cache
        for meta_file in self.features_dir.glob("*_meta.json"):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            # check if params match (ignoring shape)
            meta_check = meta.copy()
            meta_check.pop('shape', None)
            cache_check = cache_params.copy()
            
            if meta_check == cache_check:
                # load features
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
            model: trained model
            train_date: training date
            train_params: training parameters
            metrics: model performance metrics
            
        Returns:
            cache key
        """
        # create cache key
        cache_params = {
            'type': 'model',
            'train_date': str(train_date),
            'train_params': train_params,
            'metrics': metrics
        }
        cache_key = self.get_cache_key(cache_params)
        
        # save model
        model_path = self.models_dir / f"{cache_key}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'train_date': train_date,
                'metrics': metrics
            }, f)
        
        # save metadata
        meta_path = self.models_dir / f"{cache_key}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(cache_params, f, indent=2, default=str)
        
        return cache_key
    
    def load_model(self,
                  train_date: pd.Timestamp,
                  train_params: Dict[str, Any]) -> Optional[Tuple[Any, Dict[str, float]]]:
        """
        Load model from cache.
        
        Returns:
            (model, metrics) or None
        """
        # look for matching model
        target_params = {
            'type': 'model',
            'train_date': str(train_date),
            'train_params': train_params
        }
        
        for meta_file in self.models_dir.glob("*_meta.json"):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            # check if params match (ignoring metrics)
            meta_check = {k: v for k, v in meta.items() if k != 'metrics'}
            
            if meta_check == target_params:
                # load model
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
        
        # save predictions
        pred_path = self.predictions_dir / f"{cache_key}.pkl"
        predictions.to_pickle(pred_path)
        
        # save metadata
        meta_path = self.predictions_dir / f"{cache_key}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(cache_params, f, indent=2, default=str)
        
        print(f"💾 Saved predictions to cache: {cache_key}")
        return cache_key
    
    def load_predictions(self,
                        date_range: Tuple[str, str],
                        model_params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Load predictions from cache.
        """
        target_params = {
            'type': 'predictions',
            'date_range': date_range,
            'model_params': model_params
        }
        
        for meta_file in self.predictions_dir.glob("*_meta.json"):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            # check if params match (ignoring shape)
            meta_check = {k: v for k, v in meta.items() if k != 'shape'}
            
            if meta_check == target_params:
                # load predictions
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
            cache_type: type to clear ('models', 'features', etc)
                       if None, clears all
        """
        if cache_type:
            target_dir = self.cache_dir / cache_type
            if target_dir.exists():
                for file in target_dir.glob("*"):
                    file.unlink()
                print(f"🗑️ Cleared {cache_type} cache")
        else:
            # clear all caches
            for subdir in [self.models_dir, self.features_dir, 
                          self.predictions_dir, self.metrics_dir]:
                for file in subdir.glob("*"):
                    file.unlink()
            print("🗑️ Cleared all ML caches")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get info about cached items.
        
        Returns:
            dict with cache stats
        """
        info = {
            'cache_dir': str(self.cache_dir),
            'models': len(list(self.models_dir.glob("*.pkl"))),
            'features': len(list(self.features_dir.glob("*.pkl"))),
            'predictions': len(list(self.predictions_dir.glob("*.pkl"))),
            'metrics': len(list(self.metrics_dir.glob("*.pkl"))),
            'total_size_mb': 0
        }
        
        # calculate total size
        for subdir in [self.models_dir, self.features_dir, 
                      self.predictions_dir, self.metrics_dir]:
            for file in subdir.glob("*"):
                info['total_size_mb'] += file.stat().st_size / (1024 * 1024)
        
        info['total_size_mb'] = round(info['total_size_mb'], 2)
        
        return info 