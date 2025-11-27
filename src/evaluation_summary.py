"""
Evaluation summary and model comparison.

This module combines Kupiec test results from all models and computes
comparison statistics to determine which model performs best.
"""

import pandas as pd
from pathlib import Path
from src import config


def combine_kupiec_results(results_list):
    """
    Combine Kupiec test results from multiple models.

    Args:
        results_list (list): List of DataFrames with Kupiec results

    Returns:
        pd.DataFrame: Combined results sorted by model, window, and confidence level
    """
    if not results_list:
        raise ValueError("No results to combine")

    combined = pd.concat(results_list, ignore_index=True)

    # Sort for easy comparison
    sort_cols = ["ConfidenceLevel", "RollingWindow", "Model"]
    combined = combined.sort_values(sort_cols).reset_index(drop=True)

    return combined


def compare_models(kupiec_results):
    """
    Generate model comparison statistics from Kupiec test results.

    This function calculates:
    1. Deviation from expected violations (lower is better)
    2. Absolute deviation (for ranking)
    3. Coverage accuracy comparison

    Args:
        kupiec_results (pd.DataFrame): Combined Kupiec test results

    Returns:
        pd.DataFrame: Comparison statistics
    """
    # Add deviation columns
    results = kupiec_results.copy()
    results["Deviation"] = results["Violations"] - results["ExpectedViolations"]
    results["Deviation_pct"] = (results["Deviation"] / results["ExpectedViolations"]) * 100
    results["AbsDeviation"] = results["Deviation"].abs()

    # Select relevant columns for comparison
    comparison_cols = [
        "Model",
        "RollingWindow",
        "ConfidenceLevel",
        "N",
        "Violations",
        "ExpectedViolations",
        "ViolationRate",
        "Deviation",
        "Deviation_pct",
        "AbsDeviation",
        "p_value",
        "Reject_5pct",
    ]

    comparison = results[comparison_cols].copy()

    return comparison


def rank_models_by_coverage(comparison_df):
    """
    Rank models by coverage accuracy (smallest absolute deviation is best).

    Args:
        comparison_df (pd.DataFrame): Comparison statistics

    Returns:
        pd.DataFrame: Models ranked by performance for each window/CL combination
    """
    rankings = []

    for (window, cl), group in comparison_df.groupby(["RollingWindow", "ConfidenceLevel"]):
        # Sort by absolute deviation (lower is better)
        sorted_group = group.sort_values("AbsDeviation").copy()
        sorted_group["Rank"] = range(1, len(sorted_group) + 1)

        rankings.append(
            sorted_group[
                [
                    "Model",
                    "RollingWindow",
                    "ConfidenceLevel",
                    "Violations",
                    "ExpectedViolations",
                    "AbsDeviation",
                    "Rank",
                ]
            ]
        )

    return pd.concat(rankings, ignore_index=True)


def save_all_results(kupiec_results, comparison_stats, rankings, output_dir=None):
    """
    Save all evaluation results to CSV files.

    Args:
        kupiec_results (pd.DataFrame): Combined Kupiec results
        comparison_stats (pd.DataFrame): Model comparison statistics
        rankings (pd.DataFrame): Model rankings by coverage
        output_dir (Path, optional): Output directory. Defaults to config.RESULTS_DIR
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR

    # Ensure directories exist
    config.ensure_results_dir()

    # Save files
    kupiec_file = output_dir / "BTC_Kupiec_Results_All.csv"
    comparison_file = output_dir / "comparisons" / "model_comparison.csv"
    rankings_file = output_dir / "comparisons" / "model_rankings.csv"

    kupiec_results.to_csv(kupiec_file, index=False)
    comparison_stats.to_csv(comparison_file, index=False)
    rankings.to_csv(rankings_file, index=False)

    print(f"\n✓ Saved combined Kupiec results to: {kupiec_file}")
    print(f"✓ Saved model comparison to: {comparison_file}")
    print(f"✓ Saved model rankings to: {rankings_file}")

    return {"kupiec": kupiec_file, "comparison": comparison_file, "rankings": rankings_file}


def print_summary(comparison_stats, rankings):
    """
    Print a summary of model comparison results.

    Args:
        comparison_stats (pd.DataFrame): Model comparison statistics
        rankings (pd.DataFrame): Model rankings
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    print("\n1. Overall Violation Rates by Model:")
    print("-" * 80)
    summary = (
        comparison_stats.groupby("Model")
        .agg({"Violations": "sum", "ExpectedViolations": "sum", "AbsDeviation": "mean"})
        .round(2)
    )
    summary["Total_ViolationRate"] = (
        summary["Violations"] / comparison_stats.groupby("Model")["N"].sum()
    )
    print(summary)

    print("\n2. Best Model by Window and Confidence Level (Rank 1 = Best):")
    print("-" * 80)
    best_models = rankings[rankings["Rank"] == 1][
        ["RollingWindow", "ConfidenceLevel", "Model", "AbsDeviation"]
    ].sort_values(["ConfidenceLevel", "RollingWindow"])
    print(best_models.to_string(index=False))

    print("\n3. Average Ranking by Model (Lower is Better):")
    print("-" * 80)
    avg_rank = rankings.groupby("Model")["Rank"].mean().sort_values()
    print(avg_rank)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test the evaluation summary functions
    print("\n" + "=" * 60)
    print("Testing Evaluation Summary Functions")
    print("=" * 60)

    # Create synthetic test data
    test_data = []
    for model in ["Historical", "MonteCarlo", "VIXRegression"]:
        for window in ["1m", "3m", "6m", "12m"]:
            for cl in [0.95, 0.99]:
                test_data.append(
                    {
                        "Model": model,
                        "RollingWindow": window,
                        "ConfidenceLevel": cl,
                        "N": 1000,
                        "Violations": int(1000 * (1 - cl) * (1 + 0.1 * hash(model + window) % 3)),
                        "ExpectedViolations": 1000 * (1 - cl),
                        "ViolationRate": (1 - cl) * (1 + 0.1 * hash(model + window) % 3),
                        "p_value": 0.1,
                        "Reject_5pct": False,
                    }
                )

    kupiec_df = pd.DataFrame(test_data)

    # Test functions
    comparison = compare_models(kupiec_df)
    rankings = rank_models_by_coverage(comparison)

    print("\nComparison Statistics (first 5 rows):")
    print(comparison.head())

    print("\nModel Rankings (first 5 rows):")
    print(rankings.head())

    print_summary(comparison, rankings)

    print("\n✓ Evaluation summary functions verified!")
