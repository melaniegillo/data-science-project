"""
Shared test utilities to eliminate code duplication across test files.

This module provides common fixtures and assertion helpers used by multiple test files.
"""

import pandas as pd
import numpy as np

__all__ = ["create_synthetic_data", "assert_var_output_valid"]


def create_synthetic_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic BTC/VIX data for testing VaR models.

    Args:
        n: Number of observations to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with Returns, VIX_decimal, and RealizedVol_21d columns
    """
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n)

    data = pd.DataFrame(
        {
            "Returns": np.random.normal(0, 0.02, n),
            "VIX_decimal": np.random.uniform(0.1, 0.3, n),
            "RealizedVol_21d": np.random.uniform(0.3, 0.6, n),
        },
        index=dates,
    )

    return data


def assert_var_output_valid(
    results: dict[str, pd.DataFrame],
    window_label: str,
    expected_columns: list[str] | None = None,
) -> None:
    """
    Assert that VaR model output has the expected structure.

    Common assertions for VaR model outputs:
    - Window label exists in results
    - Result is a DataFrame
    - Has expected VaR columns
    - Has at least one forecast row

    Args:
        results: Dictionary mapping window labels to VaR DataFrames
        window_label: Expected window label (e.g., "test", "1m")
        expected_columns: List of expected column names. Defaults to ["VaR_95", "VaR_99"]

    Raises:
        AssertionError: If any validation fails
    """
    if expected_columns is None:
        expected_columns = ["VaR_95", "VaR_99"]

    # Check window label exists
    assert window_label in results, f"Results should contain '{window_label}' window"

    # Get VaR DataFrame
    var_df = results[window_label]

    # Validate structure
    assert isinstance(var_df, pd.DataFrame), "Result should be a DataFrame"

    # Check expected columns
    for col in expected_columns:
        assert col in var_df.columns, f"Result should have '{col}' column"

    # Check has data
    assert len(var_df) > 0, "Result should have at least one forecast"
