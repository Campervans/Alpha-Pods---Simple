"""
Float precision management utilities for the CVaR/CLEIR optimization system.

This module provides consistent float precision handling to avoid
numerical precision issues in financial calculations.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Any
from decimal import Decimal, ROUND_HALF_UP


# Configuration for financial precision
PRICE_PRECISION = 6  # 6 decimal places for prices
WEIGHT_PRECISION = 8  # 8 decimal places for portfolio weights
RETURN_PRECISION = 8  # 8 decimal places for returns
INDEX_PRECISION = 6  # 6 decimal places for index values
PERCENTAGE_PRECISION = 4  # 4 decimal places for percentages


def round_financial(value: Union[float, np.ndarray, pd.Series], precision: int = PRICE_PRECISION) -> Union[float, np.ndarray, pd.Series]:
    """
    Round financial values to specified precision using banker's rounding.
    
    Args:
        value: Value(s) to round
        precision: Number of decimal places
        
    Returns:
        Rounded value(s) with same type as input
    """
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.round(precision)
    elif isinstance(value, np.ndarray):
        return np.round(value, precision)
    elif isinstance(value, (float, np.floating)):
        return round(float(value), precision)
    else:
        return value


def round_prices(prices: Union[float, np.ndarray, pd.Series, pd.DataFrame]) -> Union[float, np.ndarray, pd.Series, pd.DataFrame]:
    """Round price values to standard financial precision."""
    return round_financial(prices, PRICE_PRECISION)


def round_weights(weights: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Round portfolio weights to standard precision."""
    return round_financial(weights, WEIGHT_PRECISION)


def round_returns(returns: Union[float, np.ndarray, pd.Series, pd.DataFrame]) -> Union[float, np.ndarray, pd.Series, pd.DataFrame]:
    """Round return values to standard precision."""
    return round_financial(returns, RETURN_PRECISION)


def round_index_values(index_values: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Round index values to standard precision."""
    return round_financial(index_values, INDEX_PRECISION)


def round_percentages(percentages: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Round percentage values to standard precision."""
    return round_financial(percentages, PERCENTAGE_PRECISION)


def normalize_weights(weights: np.ndarray, precision: int = WEIGHT_PRECISION) -> np.ndarray:
    """
    Normalize weights to sum to 1.0 with proper precision handling.
    
    Args:
        weights: Array of weights
        precision: Decimal precision for rounding
        
    Returns:
        Normalized weights that sum to exactly 1.0
    """
    # Round weights first
    weights_rounded = round_weights(weights)
    
    # Calculate sum and adjust if needed
    weights_sum = weights_rounded.sum()
    
    if not np.isclose(weights_sum, 1.0, atol=10**(-precision)):
        # Adjust the largest weight to make sum exactly 1.0
        diff = 1.0 - weights_sum
        max_idx = np.argmax(weights_rounded)
        weights_rounded[max_idx] += diff
        weights_rounded = round_weights(weights_rounded)
    
    return weights_rounded


def ensure_index_starts_at_100(index_values: pd.Series, precision: int = INDEX_PRECISION) -> pd.Series:
    """
    Ensure index values start at exactly 100.0 with proper precision.
    
    Args:
        index_values: Series of index values
        precision: Decimal precision
        
    Returns:
        Adjusted index series starting at exactly 100.0
    """
    if len(index_values) == 0:
        return index_values
    
    # Get the first value and calculate adjustment factor
    first_value = float(index_values.iloc[0])
    
    if not np.isclose(first_value, 100.0, atol=10**(-precision)):
        # Scale all values to start at exactly 100.0
        adjustment_factor = 100.0 / first_value
        adjusted_values = index_values * adjustment_factor
        adjusted_values = round_index_values(adjusted_values)
        
        # Ensure first value is exactly 100.0
        adjusted_values.iloc[0] = 100.0
        return adjusted_values
    else:
        # Round to proper precision
        rounded_values = round_index_values(index_values)
        # Ensure first value is exactly 100.0
        rounded_values.iloc[0] = 100.0
        return rounded_values


def validate_financial_precision(value: float, expected: float, precision: int = 6, tolerance_factor: float = 10.0) -> bool:
    """
    Validate that a financial value matches expected value within precision tolerance.
    
    Args:
        value: Actual value
        expected: Expected value
        precision: Decimal precision
        tolerance_factor: Multiplier for tolerance (default allows 10x precision error)
        
    Returns:
        True if value is within acceptable tolerance
    """
    tolerance = tolerance_factor * (10 ** (-precision))
    return abs(value - expected) <= tolerance


def format_financial(value: float, precision: int = 2, percentage: bool = False) -> str:
    """
    Format financial values for display.
    
    Args:
        value: Value to format
        precision: Decimal places to show
        percentage: Whether to format as percentage
        
    Returns:
        Formatted string
    """
    if percentage:
        return f"{value:.{precision}%}"
    else:
        return f"{value:,.{precision}f}"


def clean_financial_dataframe(df: pd.DataFrame, price_cols: List[str] = None, 
                            weight_cols: List[str] = None, return_cols: List[str] = None) -> pd.DataFrame:
    """
    Clean a DataFrame by applying appropriate precision to different column types.
    
    Args:
        df: DataFrame to clean
        price_cols: Columns containing price data
        weight_cols: Columns containing weight data  
        return_cols: Columns containing return data
        
    Returns:
        Cleaned DataFrame with proper precision
    """
    df_clean = df.copy()
    
    if price_cols:
        for col in price_cols:
            if col in df_clean.columns:
                df_clean[col] = round_prices(df_clean[col])
    
    if weight_cols:
        for col in weight_cols:
            if col in df_clean.columns:
                df_clean[col] = round_weights(df_clean[col])
    
    if return_cols:
        for col in return_cols:
            if col in df_clean.columns:
                df_clean[col] = round_returns(df_clean[col])
    
    return df_clean


def debug_precision_issue(value: Any, expected: float, label: str = "Value") -> None:
    """
    Debug precision issues by printing detailed information.
    
    Args:
        value: Value to debug
        expected: Expected value
        label: Label for the value
    """
    print(f"\n=== Precision Debug: {label} ===")
    print(f"Value: {value}")
    print(f"Type: {type(value)}")
    print(f"Expected: {expected}")
    
    if isinstance(value, (float, np.floating)):
        print(f"Exact value: {value:.15f}")
        print(f"Difference: {value - expected:.15f}")
        print(f"Close (1e-6): {np.isclose(value, expected, atol=1e-6)}")
        print(f"Close (1e-4): {np.isclose(value, expected, atol=1e-4)}")
        print(f"Close (1e-2): {np.isclose(value, expected, atol=1e-2)}")
    
    if hasattr(value, 'iloc'):
        print(f"First value: {value.iloc[0]:.15f}")
        print(f"First diff: {value.iloc[0] - expected:.15f}")
    
    print("=" * 40) 