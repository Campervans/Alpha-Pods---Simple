"""
Parallel processing utilities for Alpha Enhancement optimization.

This module provides utilities for detecting system resources and managing
parallel execution of computationally intensive tasks.
"""

import os
import multiprocessing as mp
import psutil
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for optimization decisions."""
    return {
        'cpu_count': mp.cpu_count(),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'memory_percent_used': psutil.virtual_memory().percent,
    }


def get_optimal_workers(memory_per_worker_gb: float = 2.0) -> int:
    """
    Get optimal number of workers based on CPU count and available memory.
    
    Parameters
    ----------
    memory_per_worker_gb : float
        Estimated memory requirement per worker in GB
        
    Returns
    -------
    int
        Optimal number of workers
    """
    cpu_count = mp.cpu_count()
    
    # Leave 1-2 cores for system (based on total cores)
    if cpu_count <= 4:
        reserved_cores = 1
    else:
        reserved_cores = 2
    
    max_workers_cpu = max(1, cpu_count - reserved_cores)
    
    # Also consider memory constraints
    available_memory = psutil.virtual_memory().available / (1024**3)
    # Reserve 2GB for system
    usable_memory = max(0, available_memory - 2.0)
    max_workers_memory = max(1, int(usable_memory / memory_per_worker_gb))
    
    # Take minimum of CPU and memory constraints
    optimal_workers = min(max_workers_cpu, max_workers_memory)
    
    logger.info(f"System has {cpu_count} CPUs, {available_memory:.1f}GB available memory")
    logger.info(f"Optimal workers: {optimal_workers} (CPU limit: {max_workers_cpu}, Memory limit: {max_workers_memory})")
    
    return optimal_workers


def should_use_parallel(data_size: int, threshold: int = 10000) -> bool:
    """
    Determine if parallel processing would be beneficial.
    
    Parameters
    ----------
    data_size : int
        Size of data (e.g., number of rows or dates)
    threshold : int
        Minimum data size to justify parallel processing overhead
        
    Returns
    -------
    bool
        True if parallel processing is recommended
    """
    return data_size >= threshold and get_optimal_workers() > 1


class ParallelConfig:
    """Configuration for parallel processing."""
    
    def __init__(
        self,
        enabled: bool = True,
        n_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        backend: str = 'loky',
        memory_per_worker_gb: float = 2.0
    ):
        """
        Initialize parallel configuration.
        
        Parameters
        ----------
        enabled : bool
            Whether parallel processing is enabled
        n_workers : int, optional
            Number of workers (if None, auto-detect)
        chunk_size : int, optional
            Size of chunks for batch processing
        backend : str
            Joblib backend ('loky', 'threading', 'multiprocessing')
        memory_per_worker_gb : float
            Estimated memory per worker for auto-detection
        """
        self.enabled = enabled
        self.backend = backend
        self.memory_per_worker_gb = memory_per_worker_gb
        
        if n_workers is None and enabled:
            self.n_workers = get_optimal_workers(memory_per_worker_gb)
        else:
            self.n_workers = n_workers or 1
            
        self.chunk_size = chunk_size
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'enabled': self.enabled,
            'n_workers': self.n_workers,
            'chunk_size': self.chunk_size,
            'backend': self.backend,
            'memory_per_worker_gb': self.memory_per_worker_gb
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ParallelConfig':
        """Create from dictionary."""
        return cls(**config_dict)


# Feature flags for gradual rollout
FEATURE_FLAGS = {
    'parallel_features': False,
    'parallel_training': False,
    'feature_caching': False,
    'lite_mode': False,
    'batch_processing': True,  # Start with this enabled
}


def is_feature_enabled(feature: str) -> bool:
    """Check if a feature flag is enabled."""
    return FEATURE_FLAGS.get(feature, False)


def set_feature_flag(feature: str, enabled: bool):
    """Set a feature flag."""
    if feature in FEATURE_FLAGS:
        FEATURE_FLAGS[feature] = enabled
        logger.info(f"Feature '{feature}' set to {enabled}")
    else:
        logger.warning(f"Unknown feature flag: {feature}") 