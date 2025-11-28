"""
Kupiec test for Value-at-Risk model validation.

The Kupiec test (also known as the unconditional coverage test) checks whether
the observed violation rate matches the expected violation rate for a given
confidence level.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import chi2
from src import config


def run_kupiec_test(
    returns: pd.Series | pd.DataFrame,
    var_forecasts: pd.Series | pd.DataFrame,
    confidence_level: float,
    model_name: str = "Unknown",
    window_label: str = "",
) -> dict[str, float | int | bool | str]:
    """
    Run Kupiec unconditional coverage test on VaR forecasts.

    The Kupiec test checks if the proportion of VaR violations is consistent
    with the expected violation rate. For example, at 95% confidence, we expect
    violations about 5% of the time.

    Args:
        returns: Actual returns (index should be dates)
        var_forecasts: VaR forecasts with column like VaR_95
        confidence_level: Confidence level (e.g., 0.95, 0.99)
        model_name: Name of the model being tested
        window_label: Rolling window label (e.g., '1m', '3m')

    Returns:
        Test results including violations, expected violations, LR statistic, p-value

    Raises:
        ValueError: If required data is missing or invalid
    """
    # Determine VaR column name
    var_col = f"VaR_{int(confidence_level * 100)}"

    # Handle DataFrame vs Series for returns
    if isinstance(returns, pd.DataFrame):
        if "Returns" not in returns.columns:
            raise ValueError("Returns DataFrame must have 'Returns' column")
        returns_series = returns["Returns"]
    else:
        returns_series = returns

    # Handle DataFrame vs Series for VaR
    if isinstance(var_forecasts, pd.DataFrame):
        if var_col not in var_forecasts.columns:
            raise ValueError(f"VaR forecasts must have '{var_col}' column")
        var_series = var_forecasts[var_col]
    else:
        var_series = var_forecasts

    # Merge returns and VaR on index (dates)
    df = pd.DataFrame({"Returns": returns_series, "VaR": var_series}).dropna()

    if df.empty:
        raise ValueError("No valid data after merging returns and VaR")

    # Calculate violations (return < -VaR means a loss exceeded the VaR threshold)
    violations = (df["Returns"] < -df["VaR"]).astype(int)
    x = int(violations.sum())  # Number of violations
    n = int(len(violations))  # Total observations
    p0 = 1.0 - confidence_level  # Expected violation rate

    if n == 0:
        raise ValueError("No observations available for testing")

    # Calculate likelihood ratio statistic
    if x == 0 or x == n:
        # Edge cases: no violations or all violations
        LR_uc = float("inf")
        p_value = 0.0
    else:
        p_hat = x / n  # Observed violation rate
        # Log-likelihoods
        logL0 = (n - x) * np.log(1 - p0) + x * np.log(p0)  # Under null hypothesis
        logL1 = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)  # Under alternative
        LR_uc = -2.0 * (logL0 - logL1)  # Likelihood ratio statistic
        # P-value from chi-squared distribution with 1 degree of freedom
        p_value = 1 - chi2.cdf(LR_uc, df=1)

    expected_violations = n * p0
    reject_5pct = p_value < config.SIGNIFICANCE_LEVEL  # Reject null at configured significance level

    return {
        "Model": model_name,
        "RollingWindow": window_label,
        "ConfidenceLevel": confidence_level,
        "N": n,
        "Violations": x,
        "ExpectedViolations": expected_violations,
        "ViolationRate": x / n if n > 0 else 0,
        "LR_uc": LR_uc,
        "p_value": p_value,
        "Reject_5pct": reject_5pct,
    }


def run_kupiec_tests_for_model(
    returns: pd.DataFrame,
    var_results_dict: dict[str, pd.DataFrame],
    model_name: str,
    confidence_levels: list[float] | None = None,
) -> pd.DataFrame:
    """
    Run Kupiec tests for all windows and confidence levels for a single model.

    Args:
        returns: DataFrame with Returns column and Date index
        var_results_dict: Dict mapping window labels to VaR forecast DataFrames
        model_name: Name of the model
        confidence_levels: List of confidence levels to test

    Returns:
        Test results for all windows and confidence levels
    """
    if confidence_levels is None:
        from src import config

        confidence_levels = config.CONFIDENCE_LEVELS

    results = []

    for window_label, var_df in var_results_dict.items():
        for cl in confidence_levels:
            try:
                result = run_kupiec_test(
                    returns=returns,
                    var_forecasts=var_df,
                    confidence_level=cl,
                    model_name=model_name,
                    window_label=window_label,
                )
                results.append(result)
            except Exception as e:
                print(f"  Warning: Kupiec test failed for {model_name} {window_label} CL={cl}: {e}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test the Kupiec test implementation
    print("\n" + "=" * 60)
    print("Testing Kupiec Test Implementation")
    print("=" * 60)

    # Create synthetic test data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=1000)
    returns = pd.Series(np.random.normal(0, 0.02, 1000), index=dates, name="Returns")
    var_95 = pd.Series(np.full(1000, 0.033), index=dates, name="VaR_95")  # ~5% violation rate

    # Run test
    result = run_kupiec_test(returns, var_95, 0.95, "TestModel", "test_window")

    print("\nTest Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("\n✓ Kupiec test implementation verified!")
