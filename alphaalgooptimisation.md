# Alpha Algorithm Optimization Plan

## Overview
This document outlines the optimization strategy for the Alpha Enhancement feature, focusing on:
1. Parallelization (multiprocessing/multithreading)
2. Feature pruning and optimization
3. Performance improvements

## Current Performance Bottlenecks

### Identified Issues
1. **Sequential Feature Engineering**: Processing 1000+ dates one by one (seen in logs)
2. **Redundant Computations**: Creating full feature matrix for each date
3. **Memory Usage**: Storing 698 features × N stocks × T dates
4. **No Parallelization**: Single-threaded execution

## Phase 1: Parallelization Infrastructure

### Step 1.1: Add CPU Detection Utility
```python
# src/utils/parallel_utils.py
import os
import multiprocessing as mp

def get_optimal_workers():
    """Get optimal number of workers based on CPU count."""
    cpu_count = mp.cpu_count()
    # Leave 1-2 cores for system
    return max(1, cpu_count - 2)
```

### Step 1.2: Add User Configuration
- Add to GUI config dialog:
  - [ ] Checkbox: "Enable parallel processing"
  - [ ] Slider: Number of workers (1 to cpu_count)
  - [ ] Warning about memory usage

### Step 1.3: Create Parallel Feature Engineering
```python
# Modify feature engineering to support batch processing
def create_features_batch(date_batch, prices_df, volumes_df, ...):
    """Process multiple dates in parallel."""
    return [create_features(date, ...) for date in date_batch]
```

## Phase 2: Feature Engineering Optimization

### Step 2.1: Profile Feature Importance
- [ ] Run feature importance analysis on historical data
- [ ] Identify top 50% most predictive features
- [ ] Create feature importance report

### Step 2.2: Implement Feature Sets
```python
class FeatureSet(Enum):
    LITE = "lite"      # Top 20-30 features
    STANDARD = "standard"  # Top 50-100 features  
    FULL = "full"      # All 698+ features
```

### Step 2.3: Lazy Feature Computation
- [ ] Compute basic features first
- [ ] Add complex features only if needed
- [ ] Cache intermediate calculations

## Phase 3: Parallel Implementation

### Step 3.1: Parallelize Feature Creation
```python
def create_features_parallel(dates, n_workers=None):
    if n_workers is None:
        n_workers = get_optimal_workers()
    
    # Split dates into chunks
    date_chunks = np.array_split(dates, n_workers)
    
    with mp.Pool(n_workers) as pool:
        results = pool.map(create_features_batch, date_chunks)
    
    return pd.concat(results)
```

### Step 3.2: Parallelize Model Training
- [ ] Use joblib for scikit-learn models
- [ ] Enable LightGBM parallel training
- [ ] Implement parallel cross-validation

### Step 3.3: Add Progress Tracking
```python
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_with_progress(func, items, n_workers):
    with ProcessPoolExecutor(n_workers) as executor:
        futures = {executor.submit(func, item): item 
                  for item in items}
        
        for future in tqdm(as_completed(futures), 
                          total=len(futures)):
            yield future.result()
```

## Phase 4: Memory Optimization

### Step 4.1: Implement Chunked Processing
- [ ] Process data in chunks of 100-200 dates
- [ ] Use generators instead of lists
- [ ] Clear intermediate results

### Step 4.2: Optimize Data Types
```python
def optimize_dtypes(df):
    """Reduce memory by optimizing data types."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df
```

### Step 4.3: Implement Feature Caching
- [ ] Cache computed features to disk
- [ ] Use HDF5 or Parquet for fast I/O
- [ ] Implement cache invalidation

## Phase 5: Algorithm Optimizations

### Step 5.1: Vectorize Operations
- [ ] Replace loops with numpy operations
- [ ] Use pandas rolling functions efficiently
- [ ] Leverage numba for critical loops

### Step 5.2: Optimize IMCA
- [ ] Pre-compute weight update matrices
- [ ] Use sparse matrices where applicable
- [ ] Implement early stopping

### Step 5.3: Smart Feature Selection
```python
def select_features_adaptive(features, target, max_features=50):
    """Dynamically select best features."""
    # Use mutual information or SHAP
    importance = mutual_info_regression(features, target)
    top_features = importance.nlargest(max_features).index
    return features[top_features]
```

## Implementation Timeline

### Week 1: Infrastructure
1. Create parallel utilities (Step 1.1)
2. Add GUI configuration (Step 1.2)
3. Test basic parallelization

### Week 2: Feature Optimization
1. Profile features (Step 2.1)
2. Implement feature sets (Step 2.2)
3. Add lazy computation (Step 2.3)

### Week 3: Parallel Processing
1. Parallelize features (Step 3.1)
2. Parallelize training (Step 3.2)
3. Add progress tracking (Step 3.3)

### Week 4: Memory & Performance
1. Implement chunking (Step 4.1)
2. Optimize data types (Step 4.2)
3. Add caching (Step 4.3)

## Testing Strategy

### Performance Benchmarks
- [ ] Measure baseline performance
- [ ] Test with 1, 2, 4, 8 workers
- [ ] Compare memory usage
- [ ] Validate results consistency

### Compatibility Testing
- [ ] Test on Windows/Mac/Linux
- [ ] Test with different Python versions
- [ ] Handle pickle/multiprocessing issues

## Research Questions for Perplexity

1. **Optimal chunk size for financial time series feature engineering?**
   - Balance between memory and computation overhead

2. **Best practices for parallelizing scikit-learn and LightGBM in production?**
   - Avoiding memory leaks and ensuring reproducibility

3. **Feature selection techniques for high-frequency financial data?**
   - Comparing mutual information, SHAP, and permutation importance

4. **Memory-efficient storage formats for financial features?**
   - HDF5 vs Parquet vs Feather for time series

5. **Handling non-pickle-able objects in multiprocessing?**
   - Strategies for complex model objects

## Quick Wins (Implement First)

1. **Add Basic Multiprocessing Toggle**
   ```python
   if config.get('enable_parallel', False):
       n_workers = config.get('n_workers', 4)
   else:
       n_workers = 1
   ```

2. **Batch Feature Creation**
   - Process 50 dates at a time instead of 1

3. **Feature Subset Options**
   - Offer "Quick" mode with top 30 features
   - Keep "Full" mode as optional

4. **Progress Bar Enhancement**
   - Show ETA and processing rate
   - Display current memory usage

## Monitoring & Metrics

### Key Metrics to Track
- Feature engineering time
- Model training time
- Memory peak usage
- CPU utilization
- Result quality (IC, Sharpe)

### Logging Enhancement
```python
import psutil

def log_performance_metrics():
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
        'n_threads': threading.active_count()
    }
```

## Rollback Strategy

### Feature Flags
```python
FEATURE_FLAGS = {
    'parallel_features': False,
    'parallel_training': False,
    'feature_caching': False,
    'lite_mode': False
}
```

### Graceful Degradation
- Catch multiprocessing errors
- Fall back to sequential processing
- Log performance degradation

## Success Criteria

1. **Performance**: 3-5x speedup on 8-core machine
2. **Memory**: <50% reduction in peak usage
3. **Quality**: No degradation in prediction accuracy
4. **Usability**: Simple on/off switch for users
5. **Compatibility**: Works on all major platforms 