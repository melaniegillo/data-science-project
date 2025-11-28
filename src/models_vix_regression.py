"""
VIX-Regression Value-at-Risk model.

This model predicts Bitcoin volatility using the VIX (CBOE Volatility Index)
and converts it to Value-at-Risk forecasts.

This implementation uses LAGGED VIX (t-1) to forecast volatility at time t,
ensuring we have no look-ahead bias and create true out-of-sample forecasts.
"""

import math
import numpy as np
import pandas as pd
from src import config
from src.validation import validate_model_inputs, validate_required_columns

__all__ = ["calculate_vix_regression_var"]


def calculate_vix_regression_var(
    data: pd.DataFrame,
    rolling_windows: dict[str, int] | None = None,
    confidence_levels: list[float] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Calculate Value-at-Risk using VIX regression model with LAGGED VIX for forecasting.

    This function:
    1. Uses a rolling window to estimate the relationship between VIX and realized volatility
    2. Uses VIX from time (i-1) to forecast volatility at time i (avoiding look-ahead bias)
    3. Converts predicted volatility to VaR using normal distribution quantiles

    Args:
        data: DataFrame with columns ['Returns', 'RealizedVol_21d', 'VIX_decimal'], Date index
        rolling_windows: Dict mapping window labels to sizes. Defaults to config.ROLLING_WINDOWS
        confidence_levels: List of confidence levels. Defaults to config.CONFIDENCE_LEVELS

    Returns:
        Dictionary mapping window labels to DataFrames with VaR forecasts

    Raises:
        ValueError: If required columns are missing from data or if inputs are invalid
    """
    # Use shared validation (replaces ~20 lines of duplicated code)
    rolling_windows, confidence_levels = validate_model_inputs(
        data, rolling_windows, confidence_levels
    )
    validate_required_columns(data, ["RealizedVol_21d", "VIX_decimal"])

    results = {}

    for window_label, window_size in rolling_windows.items():
        print(f"\n  Computing VIX regression VaR for {window_label} window (size={window_size})...")

        var_df = _compute_var_for_window(data, window_size, confidence_levels)

        results[window_label] = var_df
        print(f"  ✓ Generated {len(var_df)} VaR forecasts for {window_label}")

    return results


def _compute_var_for_window(
    df: pd.DataFrame, window_size: int, confidence_levels: list[float]
) -> pd.DataFrame:
    """
    Compute VaR forecasts for a single rolling window size.

    CRITICAL IMPLEMENTATION NOTE:
    At time i, we fit regression on window [i-window_size:i] to learn the VIX-volatility relationship.
    Then we use VIX from time (i-1) to predict volatility for time i.
    This ensures we only use information available at time i-1 to forecast time i.

    Args:
        df: DataFrame with realized vol and VIX
        window_size: Size of rolling window for regression
        confidence_levels: List of confidence levels for VaR

    Returns:
        VaR forecasts with columns VaR_95, VaR_99, etc.
    """
    # Clean data
    df_clean = df.dropna(subset=["RealizedVol_21d", "VIX_decimal"]).copy()
    df_clean = df_clean.sort_index()

    x = df_clean["VIX_decimal"].values  # VIX values
    y = df_clean["RealizedVol_21d"].values  # Realized volatility
    dates = df_clean.index

    start = window_size
    out_dates = []
    var_results: dict[str, list[float]] = {f"VaR_{int(cl * 100)}": [] for cl in confidence_levels}
    skipped = 0

    for i in range(start, len(df_clean)):
        # Training window: [i - window_size : i]
        x_win = x[i - window_size : i]
        y_win = y[i - window_size : i]

        # Skip if missing data in window
        if np.isnan(x_win).any() or np.isnan(y_win).any():
            skipped += 1
            continue

        # Fit linear regression: realized_vol = intercept + slope * VIX
        slope, intercept = np.polyfit(x_win, y_win, 1)

        # CRITICAL: Use VIX from time (i-1) to forecast time i
        # At time i, we only have information up to time i-1, so we must use
        # the lagged VIX value to avoid look-ahead bias and create a true forecast.
        # Safety check: i >= start = window_size >= 21, so i-1 >= 20 (always valid)
        assert i - 1 >= 0, f"Index {i-1} out of bounds (i={i}, window_size={window_size})"
        sigma_ann = intercept + slope * x[i - 1]  # Use lagged VIX

        # Validate prediction (volatility must be positive and finite)
        if not np.isfinite(sigma_ann) or sigma_ann <= 0:
            skipped += 1
            continue

        # Convert annualized volatility to daily volatility
        sigma_daily = sigma_ann / math.sqrt(config.TRADING_DAYS_PER_YEAR)

        # Calculate VaR for each confidence level
        # VaR is the quantile of the loss distribution (positive = loss)
        out_dates.append(dates[i])
        for cl in confidence_levels:
            z_score = config.Z_SCORES[cl]
            var_value = z_score * sigma_daily
            var_results[f"VaR_{int(cl * 100)}"].append(var_value)

    # Report skipped forecasts
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} forecasts due to invalid data")

    # Create output DataFrame
    result_df = pd.DataFrame(var_results, index=out_dates)
    return result_df


if __name__ == "__main__":
    # Test the VIX regression model
    from src.data_loader import prepare_btc_vix_data

    print("\n" + "=" * 60)
    print("Testing VIX Regression VaR Model")
    print("=" * 60)

    try:
        # Load data
        data = prepare_btc_vix_data()

        # Set Date as index
        if "Date" in data.columns:
            data = data.set_index("Date")

        # Calculate VaR
        print("\nCalculating VIX regression VaR...")
        results = calculate_vix_regression_var(data)

        # Display results
        for window_label, var_df in results.items():
            print(f"\n{window_label} window:")
            print(var_df.head())
            print(f"Shape: {var_df.shape}")

        print("\n✓ VIX regression model test completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
