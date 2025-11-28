"""
Visualization functions for VaR analysis results.

This module creates plots to visualize VaR model performance, violations,
and comparison statistics.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
from src import config

__all__ = [
    "plot_var_violations",
    "plot_model_comparison",
    "plot_coverage_accuracy",
    "generate_all_plots",
]


def plot_var_violations(
    returns: pd.DataFrame,
    var_forecasts: pd.DataFrame,
    model_name: str,
    window_label: str,
    confidence_level: float = 0.95,
    output_dir: Path | None = None,
    n_recent: int = 500,
) -> Path:
    """
    Plot actual returns with VaR thresholds and violations marked.

    Args:
        returns: DataFrame with Returns column and Date index
        var_forecasts: DataFrame with VaR forecasts
        model_name: Name of the model
        window_label: Rolling window label
        confidence_level: Confidence level to plot
        output_dir: Output directory for plot. Defaults to config.RESULTS_DIR / "figures"
        n_recent: Number of recent observations to plot (default 500)

    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "figures"

    output_dir.mkdir(exist_ok=True, parents=True)

    # Get VaR column
    var_col = f"VaR_{int(confidence_level * 100)}"

    # Merge returns and VaR
    df = pd.DataFrame({"Returns": returns["Returns"], "VaR": var_forecasts[var_col]}).dropna()

    # Take most recent n_recent observations
    df = df.tail(n_recent)

    # Identify violations (losses exceeding VaR)
    df["Violation"] = df["Returns"] < -df["VaR"]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot returns
    ax.plot(df.index, df["Returns"], label="Daily Returns", color="steelblue", linewidth=0.8, alpha=0.7)

    # Plot VaR bands
    ax.plot(df.index, -df["VaR"], label=f"VaR {int(confidence_level*100)}%", color="red", linewidth=1.5, linestyle="--")
    ax.fill_between(df.index, -df["VaR"], 0, alpha=0.1, color="red", label="Risk Zone")

    # Mark violations
    violations = df[df["Violation"]]
    if len(violations) > 0:
        ax.scatter(
            violations.index,
            violations["Returns"],
            color="darkred",
            s=50,
            marker="o",
            label=f"Violations ({len(violations)})",
            zorder=5,
        )

    # Add horizontal line at zero
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # Formatting
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Daily Returns", fontsize=11)
    ax.set_title(
        f"VaR Violations: {model_name} Model ({window_label} window, {int(confidence_level*100)}% confidence)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save plot
    filename = f"violations_{model_name}_{window_label}_CL{int(confidence_level*100)}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def plot_model_comparison(
    comparison_stats: pd.DataFrame,
    confidence_level: float = 0.95,
    output_dir: Path | None = None,
) -> Path:
    """
    Create bar chart comparing violation rates across models and windows.

    Args:
        comparison_stats: Model comparison DataFrame
        confidence_level: Confidence level to plot
        output_dir: Output directory. Defaults to config.RESULTS_DIR / "figures"

    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "figures"

    output_dir.mkdir(exist_ok=True, parents=True)

    # Filter for specific confidence level
    df = comparison_stats[comparison_stats["ConfidenceLevel"] == confidence_level].copy()

    # Pivot for grouped bar chart
    pivot = df.pivot(index="RollingWindow", columns="Model", values="ViolationRate")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar positions
    x = np.arange(len(pivot.index))
    width = 0.25
    models = pivot.columns

    # Plot bars for each model
    colors = ["steelblue", "orange", "green"]
    for i, (model, color) in enumerate(zip(models, colors)):
        offset = width * (i - 1)
        ax.bar(x + offset, pivot[model] * 100, width, label=model, color=color, alpha=0.8)

    # Add expected violation rate line
    expected_rate = (1 - confidence_level) * 100
    ax.axhline(y=expected_rate, color="red", linestyle="--", linewidth=2, label=f"Expected ({expected_rate}%)")

    # Formatting
    ax.set_xlabel("Rolling Window", fontsize=12)
    ax.set_ylabel("Violation Rate (%)", fontsize=12)
    ax.set_title(
        f"Model Comparison: Violation Rates ({int(confidence_level*100)}% Confidence)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", linestyle=":")

    plt.tight_layout()

    # Save plot
    filename = f"model_comparison_CL{int(confidence_level*100)}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def plot_coverage_accuracy(
    comparison_stats: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """
    Plot absolute deviation from expected violations for all models.

    Args:
        comparison_stats: Model comparison DataFrame
        output_dir: Output directory. Defaults to config.RESULTS_DIR / "figures"

    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "figures"

    output_dir.mkdir(exist_ok=True, parents=True)

    # Create figure with subplots for each confidence level
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    confidence_levels = sorted(comparison_stats["ConfidenceLevel"].unique())

    for idx, cl in enumerate(confidence_levels):
        ax = axes[idx]
        df = comparison_stats[comparison_stats["ConfidenceLevel"] == cl].copy()

        # Pivot for grouped bar chart
        pivot = df.pivot(index="RollingWindow", columns="Model", values="AbsDeviation")

        # Bar positions
        x = np.arange(len(pivot.index))
        width = 0.25
        models = pivot.columns

        # Plot bars
        colors = ["steelblue", "orange", "green"]
        for i, (model, color) in enumerate(zip(models, colors)):
            offset = width * (i - 1)
            ax.bar(x + offset, pivot[model], width, label=model, color=color, alpha=0.8)

        # Formatting
        ax.set_xlabel("Rolling Window", fontsize=11)
        ax.set_ylabel("Absolute Deviation from Expected", fontsize=11)
        ax.set_title(f"{int(cl*100)}% Confidence Level", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y", linestyle=":")

    fig.suptitle("Coverage Accuracy: Deviation from Expected Violations", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save plot
    filename = "coverage_accuracy.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def generate_all_plots(
    returns: pd.DataFrame,
    var_results: dict[str, dict[str, pd.DataFrame]],
    comparison_stats: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, list[Path]]:
    """
    Generate all visualization plots.

    Args:
        returns: DataFrame with Returns column and Date index
        var_results: Nested dict {model_name: {window_label: var_df}}
        comparison_stats: Model comparison DataFrame
        output_dir: Output directory. Defaults to config.RESULTS_DIR / "figures"

    Returns:
        Dictionary mapping plot types to lists of file paths
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "figures"

    output_dir.mkdir(exist_ok=True, parents=True)

    saved_plots: dict[str, list[Path]] = {"violations": [], "comparisons": [], "accuracy": []}

    print("\nGenerating visualizations...")

    # 1. Model comparison plots (one per confidence level)
    for cl in config.CONFIDENCE_LEVELS:
        filepath = plot_model_comparison(comparison_stats, confidence_level=cl, output_dir=output_dir)
        saved_plots["comparisons"].append(filepath)
        print(f"  ✓ Model comparison plot (CL={int(cl*100)}%): {filepath.name}")

    # 2. Coverage accuracy plot
    filepath = plot_coverage_accuracy(comparison_stats, output_dir=output_dir)
    saved_plots["accuracy"].append(filepath)
    print(f"  ✓ Coverage accuracy plot: {filepath.name}")

    # 3. Violation plots (selected examples - 12m window, 95% confidence for each model)
    for model_name, windows_dict in var_results.items():
        if "12m" in windows_dict:
            filepath = plot_var_violations(
                returns=returns,
                var_forecasts=windows_dict["12m"],
                model_name=model_name,
                window_label="12m",
                confidence_level=0.95,
                output_dir=output_dir,
                n_recent=500,
            )
            saved_plots["violations"].append(filepath)
            print(f"  ✓ Violations plot ({model_name}, 12m): {filepath.name}")

    total_plots = sum(len(v) for v in saved_plots.values())
    print(f"\n✓ Generated {total_plots} visualization plots in: {output_dir}")

    return saved_plots
