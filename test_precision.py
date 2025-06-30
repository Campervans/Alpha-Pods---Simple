#!/usr/bin/env python3

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from src.utils.precision import (
    round_index_values, ensure_index_starts_at_100, 
    normalize_weights, debug_precision_issue
)

def test_precision_management():
    print("Testing precision management utilities...")
    
    # Test 1: Index values that don't start at 100.0
    print("\n=== Test 1: Index starting at 99.895 ===")
    problematic_index = pd.Series([99.89528237334773, 102.083212537271038, 104.567])
    print(f"Original first value: {problematic_index.iloc[0]:.15f}")
    
    fixed_index = ensure_index_starts_at_100(problematic_index)
    print(f"Fixed first value: {fixed_index.iloc[0]:.15f}")
    print(f"Is close to 100.0? {np.isclose(fixed_index.iloc[0], 100.0, atol=1e-6)}")
    
    # Test 2: Weight normalization
    print("\n=== Test 2: Weight normalization ===")
    messy_weights = np.array([0.2001, 0.3002, 0.4999, 0.0001])
    print(f"Original weights sum: {messy_weights.sum():.10f}")
    
    clean_weights = normalize_weights(messy_weights)
    print(f"Cleaned weights sum: {clean_weights.sum():.10f}")
    print(f"Clean weights: {clean_weights}")
    
    # Test 3: Index rounding
    print("\n=== Test 3: Index value rounding ===")
    noisy_values = pd.Series([100.0000000001, 102.0832125372710384, 104.5678901234567])
    rounded_values = round_index_values(noisy_values)
    print(f"Original: {noisy_values.values}")
    print(f"Rounded:  {rounded_values.values}")
    
    print("\n✓ All precision tests completed!")

if __name__ == "__main__":
    test_precision_management() 