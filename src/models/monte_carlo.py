"""
Monte Carlo Value-at-Risk model.

This model uses parametric simulation to estimate VaR. It assumes returns
follow a normal distribution with mean and standard deviation estimated from
historical data.
"""

import numpy as np
import pandas as pd
from src import config
from src.utils.validation import validate_model_inputs, validate_required_columns
from src.models.utils import compute_var_rolling_window

__all__ = ["calculate_monte_carlo_var"]


def calculate_monte_carlo_var(
    data: pd.DataFrame,
    rolling_windows: dict[str, int] | None = None,
    confidence_levels: list[float] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Calculate Value-at-Risk using Monte Carlo simulation.

    This function uses a rolling window approach where VaR at time t is calculated
    by:
    1. Estimating mean (μ) and std dev (σ) from returns [t-window_size, ..., t-1]
    2. Simulating 100,000 returns from N(μ, σ²)
    3. Taking the empirical quantile of the simulations

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
        print(f"\n  Computing Monte Carlo VaR for {window_label} window (size={window_size})...")

        var_df = _compute_var_for_window(data["Returns"], window_size, confidence_levels)

        results[window_label] = var_df
        print(f"  ✓ Generated {len(var_df)} VaR forecasts for {window_label}")

    return results


def _compute_var_for_window(
    returns_series: pd.Series, window_size: int, confidence_levels: list[float]
) -> pd.DataFrame:
    """
    Compute Monte Carlo VaR for a single rolling window size.

    Args:
        returns_series: Series of returns with Date index
        window_size: Size of rolling window
        confidence_levels: List of confidence levels

    Returns:
        VaR forecasts with columns VaR_95, VaR_99, etc.
    """
    # Use fixed random seed for reproducibility
    rng = np.random.default_rng(config.RANDOM_SEED)

    # Pre-generate random draws (reused for all time points)
    z = rng.standard_normal(config.MONTE_CARLO_SIMS)

    values = returns_series.values
    dates = returns_series.index

    # Define VaR calculation for a single time point
    def calculate_var_at_point(i: int, cls: list[float]) -> dict[str, float] | None:
        """Calculate Monte Carlo VaR at time i using simulated returns."""
        # Get returns window: [i-window_size : i-1]
        window = values[i - window_size : i]

        # Skip if window has NaN values
        if np.isnan(window).any():
            return None

        # Estimate parameters from window
        mu = float(np.mean(window))
        sigma = float(np.std(window, ddof=1))  # Sample std dev

        # Skip if parameters are invalid
        if not np.isfinite(mu):
            return None

        # Handle edge case: zero volatility
        if not np.isfinite(sigma) or sigma == 0.0:
            # If no volatility, all simulations equal the mean
            sims = np.full(config.MONTE_CARLO_SIMS, mu)
        else:
            # Simulate returns: N(μ, σ²)
            sims = mu + sigma * z

        # Calculate VaR for each confidence level
        var_point = {}
        for cl in cls:
            # Get the quantile of simulated returns
            # For 95% confidence, we want the 5th percentile (worst 5% of outcomes)
            q = np.quantile(sims, 1 - cl)
            var_value = -q  # Negative because losses are negative returns
            var_point[f"VaR_{int(cl * 100)}"] = var_value

        return var_point

    # Use shared rolling window utility
    return compute_var_rolling_window(dates, window_size, confidence_levels, calculate_var_at_point)


if __name__ == "__main__":
    # Test the Monte Carlo VaR model
    from src.data_loader import prepare_btc_vix_data

    print("\n" + "=" * 60)
    print("Testing Monte Carlo VaR Model")
    print("=" * 60)

    try:
        # Load data
        data = prepare_btc_vix_data()

        # Set Date as index
        if "Date" in data.columns:
            data = data.set_index("Date")

        # Calculate VaR
        print("\nCalculating Monte Carlo VaR...")
        results = calculate_monte_carlo_var(data)

        # Display results
        for window_label, var_df in results.items():
            print(f"\n{window_label} window:")
            print(var_df.head())
            print(f"Shape: {var_df.shape}")

        print("\n✓ Monte Carlo VaR model test completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
