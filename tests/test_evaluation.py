"""
Tests for evaluation functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation_kupiec import run_kupiec_test
from src.evaluation_summary import compare_models, rank_models_by_coverage


def test_kupiec_test_basic():
    """Test Kupiec test with known violation rate."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2020-01-01", periods=n)

    # Create returns and VaR where we expect ~5% violations
    returns = pd.Series(np.random.normal(0, 0.02, n), index=dates)
    var_95 = pd.Series(np.full(n, 0.033), index=dates)  # Should give ~5% violations

    result = run_kupiec_test(returns, var_95, 0.95, "TestModel", "test_window")

    # Check result structure
    assert "Model" in result
    assert "RollingWindow" in result
    assert "ConfidenceLevel" in result
    assert "N" in result
    assert "Violations" in result
    assert "ExpectedViolations" in result
    assert "ViolationRate" in result
    assert "LR_uc" in result
    assert "p_value" in result
    assert "Reject_5pct" in result

    # Check basic properties
    assert result["N"] == n
    assert 0 <= result["Violations"] <= n
    assert result["ExpectedViolations"] == pytest.approx(n * 0.05)
    assert 0 <= result["ViolationRate"] <= 1
    assert 0 <= result["p_value"] <= 1

    print(f"✓ Kupiec test basic test passed")
    print(f"  - Violations: {result['Violations']} (expected: {result['ExpectedViolations']:.1f})")
    print(f"  - p-value: {result['p_value']:.4f}")


def test_kupiec_test_edge_cases():
    """Test Kupiec test handles edge cases."""
    dates = pd.date_range("2020-01-01", periods=100)

    # Case 1: No violations
    returns = pd.Series(np.zeros(100), index=dates)  # No losses
    var_95 = pd.Series(np.full(100, 0.01), index=dates)  # Positive VaR

    result = run_kupiec_test(returns, var_95, 0.95, "Test", "test")
    assert result["Violations"] == 0
    assert result["LR_uc"] == float("inf")  # Edge case handling

    print("✓ Kupiec test edge case (no violations) passed")

    # Case 2: Series input (not DataFrame)
    returns_series = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
    var_series = pd.Series(np.full(100, 0.033), index=dates)

    result = run_kupiec_test(returns_series, var_series, 0.95, "Test", "test")
    assert result["N"] > 0
    assert "Violations" in result

    print("✓ Kupiec test edge case (Series input) passed")


def test_compare_models():
    """Test model comparison function."""
    # Create synthetic Kupiec results
    test_data = []
    for model in ["Model_A", "Model_B"]:
        for window in ["1m", "3m"]:
            test_data.append(
                {
                    "Model": model,
                    "RollingWindow": window,
                    "ConfidenceLevel": 0.95,
                    "N": 1000,
                    "Violations": 50 if model == "Model_A" else 60,
                    "ExpectedViolations": 50.0,
                    "ViolationRate": 0.05 if model == "Model_A" else 0.06,
                    "p_value": 0.5,
                    "Reject_5pct": False,
                }
            )

    kupiec_df = pd.DataFrame(test_data)
    comparison = compare_models(kupiec_df)

    # Check that deviation columns were added
    assert "Deviation" in comparison.columns
    assert "Deviation_pct" in comparison.columns
    assert "AbsDeviation" in comparison.columns

    # Check deviation calculation for Model_A (perfect match)
    model_a_row = comparison[comparison["Model"] == "Model_A"].iloc[0]
    assert model_a_row["Deviation"] == pytest.approx(0.0)
    assert model_a_row["AbsDeviation"] == pytest.approx(0.0)

    # Check deviation calculation for Model_B (10 extra violations)
    model_b_row = comparison[comparison["Model"] == "Model_B"].iloc[0]
    assert model_b_row["Deviation"] == pytest.approx(10.0)
    assert model_b_row["AbsDeviation"] == pytest.approx(10.0)

    print("✓ Model comparison test passed")


def test_rank_models_by_coverage():
    """Test model ranking function."""
    # Create comparison data
    comparison_data = []
    for model in ["Model_A", "Model_B", "Model_C"]:
        comparison_data.append(
            {
                "Model": model,
                "RollingWindow": "1m",
                "ConfidenceLevel": 0.95,
                "N": 1000,
                "Violations": 50,
                "ExpectedViolations": 50.0,
                "Deviation": 0.0 if model == "Model_A" else (5.0 if model == "Model_B" else 10.0),
                "AbsDeviation": 0.0
                if model == "Model_A"
                else (5.0 if model == "Model_B" else 10.0),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    rankings = rank_models_by_coverage(comparison_df)

    # Check rankings
    assert "Rank" in rankings.columns

    # Model_A should be rank 1 (lowest absolute deviation)
    model_a_rank = rankings[rankings["Model"] == "Model_A"]["Rank"].iloc[0]
    assert model_a_rank == 1

    # Model_C should be rank 3 (highest absolute deviation)
    model_c_rank = rankings[rankings["Model"] == "Model_C"]["Rank"].iloc[0]
    assert model_c_rank == 3

    print("✓ Model ranking test passed")


if __name__ == "__main__":
    print("Running evaluation tests...")
    test_kupiec_test_basic()
    test_kupiec_test_edge_cases()
    test_compare_models()
    test_rank_models_by_coverage()
    print("\n✓ All evaluation tests passed!")
