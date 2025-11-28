"""
Historical Value-at-Risk model.

This model uses the empirical distribution of past returns to estimate VaR.
It's a non-parametric approach that makes no assumptions about the distribution
of returns.
"""

import numpy as np
import pandas as pd
from src import config
from src.validation import validate_model_inputs, validate_required_columns
from src.model_utils import compute_var_rolling_window

__all__ = ["calculate_historical_var"]


def calculate_historical_var(
    data: pd.DataFrame,
    rolling_windows: dict[str, int] | None = None,
    confidence_levels: list[float] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Calculate Value-at-Risk using historical simulation (empirical quantiles).

    This function uses a rolling window approach where VaR at time t is calculated
    using the empirical quantile of returns from [t-window_size, ..., t-1].

    Args:
        data: DataFrame with 'Returns' column and Date index
        rolling_windows: Dict mapping window labels to sizes. Defaults to config.ROLLING_WINDOWS
        confidence_levels: List of confidence levels. Defaults to config.CONFIDENCE_LEVELS

    Returns:
        Dictionary mapping window labels to DataFrames with VaR forecasts

    Raises:
        ValueError: If 'Returns' column is missing from data or if inputs are invalid
    """
    # Use shared validation (replaces ~20 lines of duplicated code)
    rolling_windows, confidence_levels = validate_model_inputs(
        data, rolling_windows, confidence_levels
    )
    validate_required_columns(data, ["Returns"])

    results = {}

    for window_label, window_size in rolling_windows.items():
        print(f"\n  Computing Historical VaR for {window_label} window (size={window_size})...")

        var_df = _compute_var_for_window(data, window_size, confidence_levels)

        results[window_label] = var_df
        print(f"  ✓ Generated {len(var_df)} VaR forecasts for {window_label}")

    return results


def _compute_var_for_window(
    data: pd.DataFrame, window_size: int, confidence_levels: list[float]
) -> pd.DataFrame:
    """
    Compute Historical VaR for a single rolling window size.

    Args:
        data: DataFrame with Returns column
        window_size: Size of rolling window
        confidence_levels: List of confidence levels

    Returns:
        VaR forecasts with columns VaR_95, VaR_99, etc.
    """
    df = data.copy()

    # Extract dates and reset index for integer-based indexing
    if df.index.name == "Date" or "Date" in df.columns:
        dates = df.index if df.index.name == "Date" else df["Date"]
    else:
        dates = df.index

    df = df.reset_index(drop=True)
    returns = df["Returns"].values

    # Define VaR calculation for a single time point
    def calculate_var_at_point(i: int, cls: list[float]) -> dict[str, float] | None:
        """Calculate Historical VaR at time i using past returns."""
        # Get returns window: [i-window_size : i-1]
        returns_window = returns[i - window_size : i]

        # Skip if window has NaN values
        if pd.isna(returns_window).any():
            return None

        # Calculate VaR for each confidence level
        var_point = {}
        for cl in cls:
            # VaR is the negative of the quantile (we report losses as positive)
            # For 95% confidence, we take the 5th percentile of returns (losses)
            q = np.quantile(returns_window, 1 - cl)
            var_value = -q  # Negative because losses are negative returns
            var_point[f"VaR_{int(cl * 100)}"] = var_value

        return var_point

    # Use shared rolling window utility
    return compute_var_rolling_window(dates, window_size, confidence_levels, calculate_var_at_point)


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
