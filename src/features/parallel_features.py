"""
Parallel feature engineering for Alpha Enhancement.

this module has parallel versions of feature engineering functions
to speed up processing big datasets.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Callable
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
from datetime import datetime

from src.utils.parallel_utils import ParallelConfig, is_feature_enabled
from src.features.alpha_features import AlphaFeatureEngine

logger = logging.getLogger(__name__)


class ParallelFeatureEngine:
    """Parallel feature engineering."""
    
    def __init__(
        self,
        feature_engine: AlphaFeatureEngine,
        parallel_config: Optional[ParallelConfig] = None
    ):
        """
        init parallel feature engine.
        
        Parameters
        ----------
        feature_engine : AlphaFeatureEngine
            base feature engine to parallelize
        parallel_config : ParallelConfig, optional
            parallel processing config
        """
        self.feature_engine = feature_engine
        self.config = parallel_config or ParallelConfig(enabled=False)
        
    def create_features_batch(
        self,
        date_batch: List[pd.Timestamp],
        prices_df: pd.DataFrame,
        volumes_df: pd.DataFrame,
        fundamentals_df: Optional[pd.DataFrame] = None,
        market_indicators_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create features for a batch of dates.
        
        Parameters
        ----------
        date_batch : List[pd.Timestamp]
            batch of dates to process
        prices_df : pd.DataFrame
            price data
        volumes_df : pd.DataFrame
            volume data
        fundamentals_df : pd.DataFrame, optional
            fundamental data
        market_indicators_df : pd.DataFrame, optional
            market indicators data
            
        Returns
        -------
        pd.DataFrame
            features for all dates in batch
        """
        batch_features = []
        
        for date in date_batch:
            try:
                features = self.feature_engine.create_features(
                    date=date,
                    prices_df=prices_df,
                    volumes_df=volumes_df,
                    fundamentals_df=fundamentals_df,
                    market_indicators_df=market_indicators_df
                )
                if features is not None:
                    batch_features.append(features)
            except Exception as e:
                logger.warning(f"Failed to create features for {date}: {e}")
                continue
                
        if batch_features:
            return pd.concat(batch_features, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def create_features_parallel(
        self,
        dates: List[pd.Timestamp],
        prices_df: pd.DataFrame,
        volumes_df: pd.DataFrame,
        fundamentals_df: Optional[pd.DataFrame] = None,
        market_indicators_df: Optional[pd.DataFrame] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Create features for multiple dates in parallel.
        
        Parameters
        ----------
        dates : List[pd.Timestamp]
            list of dates to process
        prices_df : pd.DataFrame
            price data
        volumes_df : pd.DataFrame
            volume data
        fundamentals_df : pd.DataFrame, optional
            fundamental data
        market_indicators_df : pd.DataFrame, optional
            market indicators data
        show_progress : bool
            whether to show progress bar
            
        Returns
        -------
        pd.DataFrame
            features for all dates
        """
        if not self.config.enabled or len(dates) < 10:
            # fallback to sequential for small datasets
            logger.info("Using sequential processing")
            return self._create_features_sequential(
                dates, prices_df, volumes_df, fundamentals_df, market_indicators_df, show_progress
            )
        
        # determine chunk size
        if self.config.chunk_size:
            chunk_size = self.config.chunk_size
        else:
            # auto-determine chunk size
            chunk_size = max(10, len(dates) // (self.config.n_workers * 4))
            chunk_size = min(chunk_size, 200)  # cap at 200 dates per chunk
        
        logger.info(
            f"Using parallel processing with {self.config.n_workers} workers, "
            f"chunk size: {chunk_size}"
        )
        
        # split dates into chunks
        date_chunks = [
            dates[i:i + chunk_size] 
            for i in range(0, len(dates), chunk_size)
        ]
        
        # process chunks in parallel
        if self.config.backend == 'loky' or self.config.backend == 'multiprocessing':
            results = self._parallel_joblib(
                date_chunks, prices_df, volumes_df, fundamentals_df, 
                market_indicators_df, show_progress
            )
        else:
            # threading backend
            results = self._parallel_threading(
                date_chunks, prices_df, volumes_df, fundamentals_df,
                market_indicators_df, show_progress
            )
        
        # combine results
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _create_features_sequential(
        self,
        dates: List[pd.Timestamp],
        prices_df: pd.DataFrame,
        volumes_df: pd.DataFrame,
        fundamentals_df: Optional[pd.DataFrame],
        market_indicators_df: Optional[pd.DataFrame],
        show_progress: bool
    ) -> pd.DataFrame:
        """sequential feature creation with progress bar."""
        all_features = []
        
        iterator = tqdm(dates, desc="Creating features") if show_progress else dates
        
        for date in iterator:
            try:
                features = self.feature_engine.create_features(
                    date=date,
                    prices_df=prices_df,
                    volumes_df=volumes_df,
                    fundamentals_df=fundamentals_df,
                    market_indicators_df=market_indicators_df
                )
                if features is not None:
                    all_features.append(features)
            except Exception as e:
                logger.warning(f"Failed to create features for {date}: {e}")
                continue
        
        if all_features:
            return pd.concat(all_features, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _parallel_joblib(
        self,
        date_chunks: List[List[pd.Timestamp]],
        prices_df: pd.DataFrame,
        volumes_df: pd.DataFrame,
        fundamentals_df: Optional[pd.DataFrame],
        market_indicators_df: Optional[pd.DataFrame],
        show_progress: bool
    ) -> List[pd.DataFrame]:
        """process chunks using joblib."""
        from joblib import parallel_backend
        
        with parallel_backend(self.config.backend, n_jobs=self.config.n_workers):
            if show_progress:
                # use tqdm with joblib
                results = Parallel(n_jobs=self.config.n_workers)(
                    delayed(self.create_features_batch)(
                        chunk, prices_df, volumes_df, fundamentals_df, market_indicators_df
                    )
                    for chunk in tqdm(date_chunks, desc="Processing chunks")
                )
            else:
                results = Parallel(n_jobs=self.config.n_workers)(
                    delayed(self.create_features_batch)(
                        chunk, prices_df, volumes_df, fundamentals_df, market_indicators_df
                    )
                    for chunk in date_chunks
                )
        
        return [r for r in results if not r.empty]
    
    def _parallel_threading(
        self,
        date_chunks: List[List[pd.Timestamp]],
        prices_df: pd.DataFrame,
        volumes_df: pd.DataFrame,
        fundamentals_df: Optional[pd.DataFrame],
        market_indicators_df: Optional[pd.DataFrame],
        show_progress: bool
    ) -> List[pd.DataFrame]:
        """process chunks using threading (for I/O bound stuff)."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            # submit all tasks
            future_to_chunk = {
                executor.submit(
                    self.create_features_batch,
                    chunk, prices_df, volumes_df, fundamentals_df, market_indicators_df
                ): chunk
                for chunk in date_chunks
            }
            
            # process completed tasks
            if show_progress:
                iterator = tqdm(
                    as_completed(future_to_chunk),
                    total=len(date_chunks),
                    desc="Processing chunks"
                )
            else:
                iterator = as_completed(future_to_chunk)
            
            for future in iterator:
                try:
                    result = future.result()
                    if not result.empty:
                        results.append(result)
                except Exception as e:
                    chunk = future_to_chunk[future]
                    logger.error(f"Error processing chunk {chunk[0]} - {chunk[-1]}: {e}")
        
        return results


def optimize_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize data types to reduce memory usage.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimize
        
    Returns
    -------
    pd.DataFrame
        optimized DataFrame
    """
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # optimize numeric columns
    for col in df.select_dtypes(include=['float64']).columns:
        # check if we can use float32
        col_min = df[col].min()
        col_max = df[col].max()
        
        if pd.isna(col_min) or pd.isna(col_max):
            continue
            
        # float32 range: ±3.4e38
        if abs(col_min) < 3.4e38 and abs(col_max) < 3.4e38:
            df[col] = df[col].astype('float32')
    
    # optimize integer columns
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= 0:
            # unsigned ints
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            # signed ints
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype('int32')
    
    # convert object columns to category if it makes sense
    # TODO: this is a simple heuristic, could be improved
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    memory_reduction = (initial_memory - final_memory) / initial_memory * 100
    
    logger.info(
        f"Memory usage reduced from {initial_memory:.2f} MB to {final_memory:.2f} MB "
        f"({memory_reduction:.1f}% reduction)"
    )
    
    return df