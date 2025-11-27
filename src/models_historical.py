"""
Historical Value-at-Risk model.

This model uses the empirical distribution of past returns to estimate VaR.
It's a non-parametric approach that makes no assumptions about the distribution
of returns.
"""

import numpy as np
import pandas as pd
from src import config


def calculate_historical_var(data, rolling_windows=None, confidence_levels=None):
    """
    Calculate Value-at-Risk using historical simulation (empirical quantiles).

    This function uses a rolling window approach where VaR at time t is calculated
    using the empirical quantile of returns from [t-window_size, ..., t-1].

    Args:
        data (pd.DataFrame): DataFrame with 'Returns' column and Date index
        rolling_windows (dict, optional): Dict mapping window labels to sizes.
                                          Defaults to config.ROLLING_WINDOWS
        confidence_levels (list, optional): List of confidence levels. Defaults to config.CONFIDENCE_LEVELS

    Returns:
        dict: Dictionary mapping window labels to DataFrames with VaR forecasts

    Raises:
        ValueError: If 'Returns' column is missing from data
    """
    if rolling_windows is None:
        rolling_windows = config.ROLLING_WINDOWS
    if confidence_levels is None:
        confidence_levels = config.CONFIDENCE_LEVELS

    # Validate required columns
    if "Returns" not in data.columns:
        raise ValueError("Data must have 'Returns' column")

    results = {}

    for window_label, window_size in rolling_windows.items():
        print(f"\n  Computing Historical VaR for {window_label} window (size={window_size})...")

        var_df = _compute_var_for_window(data, window_size, confidence_levels)

        results[window_label] = var_df
        print(f"  ✓ Generated {len(var_df)} VaR forecasts for {window_label}")

    return results


def _compute_var_for_window(data, window_size, confidence_levels):
    """
    Compute Historical VaR for a single rolling window size.

    Args:
        data: DataFrame with Returns column
        window_size: Size of rolling window
        confidence_levels: List of confidence levels

    Returns:
        pd.DataFrame: VaR forecasts with columns VaR_95, VaR_99, etc.
    """
    df = data.copy()

    # Reset index to use integer indexing
    if df.index.name == "Date" or "Date" in df.columns:
        dates = df.index if df.index.name == "Date" else df["Date"]
    else:
        dates = df.index

    df = df.reset_index(drop=True)

    # Initialize VaR columns
    var_results = {f"VaR_{int(cl * 100)}": [] for cl in confidence_levels}
    out_dates = []

    # Calculate VaR for each time point
    for i in range(window_size, len(df)):
        # Get returns window: [i-window_size : i-1]
        # This uses only past data to forecast time i
        returns_window = df.loc[i - window_size : i - 1, "Returns"]

        # Skip if window has NaN values
        if returns_window.isna().any():
            continue

        out_dates.append(dates[i])

        # Calculate VaR for each confidence level
        for cl in confidence_levels:
            # VaR is the negative of the quantile (we report losses as positive)
            # For 95% confidence, we take the 5th percentile of returns (losses)
            q = np.quantile(returns_window, 1 - cl)
            var_value = -q  # Negative because losses are negative returns
            var_results[f"VaR_{int(cl * 100)}"].append(var_value)

    # Create output DataFrame
    result_df = pd.DataFrame(var_results, index=out_dates)
    return result_df


if __name__ == "__main__":
    # Test the Historical VaR model
    from src.data_loader import prepare_btc_vix_data

    print("\n" + "=" * 60)
    print("Testing Historical VaR Model")
    print("=" * 60)

    try:
        # Load data
        data = prepare_btc_vix_data()

        # Set Date as index
        if "Date" in data.columns:
            data = data.set_index("Date")

        # Calculate VaR
        print("\nCalculating Historical VaR...")
        results = calculate_historical_var(data)

        # Display results
        for window_label, var_df in results.items():
            print(f"\n{window_label} window:")
            print(var_df.head())
            print(f"Shape: {var_df.shape}")

        print("\n✓ Historical VaR model test completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
