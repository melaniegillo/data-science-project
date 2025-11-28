"""
Shared utilities for VaR model implementations.

This module contains common functions used across all VaR models to eliminate
code duplication and ensure consistent behavior.
"""

from typing import Callable
import pandas as pd
from src import config

__all__ = ["compute_var_rolling_window"]


def compute_var_rolling_window(
    dates: pd.DatetimeIndex,
    window_size: int,
    confidence_levels: list[float],
    calculate_var_point: Callable[[int, list[float]], dict[str, float] | None],
) -> pd.DataFrame:
    """
    Generic rolling window VaR calculator that eliminates code duplication.

    This function handles all the boilerplate code shared across VaR models:
    - Initializing result containers
    - Looping through time points
    - Collecting VaR values for each confidence level
    - Tracking and reporting skipped forecasts
    - Building the output DataFrame

    The model-specific calculation logic is injected via the calculate_var_point callback.

    Args:
        dates: DatetimeIndex of all time points in the data
        window_size: Size of rolling window for estimation
        confidence_levels: List of confidence levels for VaR (e.g., [0.95, 0.99])
        calculate_var_point: Callback function that calculates VaR at a single time point.
            - Takes (time_index: int, confidence_levels: list[float])
            - Returns dict mapping "VaR_95" -> value, "VaR_99" -> value, etc.
            - Returns None if the calculation should be skipped (invalid data)

    Returns:
        DataFrame with VaR forecasts, indexed by date, with columns VaR_95, VaR_99, etc.

    Example:
        >>> def my_var_calc(i: int, cls: list[float]) -> dict[str, float] | None:
        ...     # Model-specific logic here
        ...     return {"VaR_95": 0.02, "VaR_99": 0.03}
        >>> result = compute_var_rolling_window(dates, 21, [0.95, 0.99], my_var_calc)
    """
    # Initialize result containers
    var_results: dict[str, list[float]] = {
        f"VaR_{int(cl * 100)}": [] for cl in confidence_levels
    }
    out_dates = []
    skipped = 0

    # Rolling window loop
    start = window_size
    for i in range(start, len(dates)):
        # Call model-specific VaR calculation
        var_point = calculate_var_point(i, confidence_levels)

        # Skip if calculation returned None (invalid data)
        if var_point is None:
            skipped += 1
            continue

        # Store results
        out_dates.append(dates[i])
        for var_label, var_value in var_point.items():
            var_results[var_label].append(var_value)

    # Report skipped forecasts
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} forecasts due to invalid data")

    # Create and return output DataFrame
    result_df = pd.DataFrame(var_results, index=out_dates)
    return result_df
