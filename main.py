"""
Main entry point for Bitcoin VaR Forecasting project.

This script orchestrates the entire VaR analysis pipeline:
1. Load and prepare BTC/VIX data
2. Calculate VaR using three models (Historical, Monte Carlo, VIX-Regression)
3. Run Kupiec tests for all models
4. Generate model comparison statistics
5. Save all results
"""

from src import config
from src.data_loader import prepare_btc_vix_data
from src.models_historical import calculate_historical_var
from src.models_monte_carlo import calculate_monte_carlo_var
from src.models_vix_regression import calculate_vix_regression_var
from src.evaluation_kupiec import run_kupiec_tests_for_model
from src.evaluation_summary import (
    combine_kupiec_results,
    compare_models,
    rank_models_by_coverage,
    save_all_results,
    print_summary
)


def main():
    """Run the complete VaR analysis pipeline."""
    print("\n" + "=" * 80)
    print("BITCOIN VALUE-AT-RISK FORECASTING")
    print("=" * 80)
    print("\nResearch Question:")
    print("Does incorporating information from the VIX improve the accuracy of one-day")
    print("Value-at-Risk forecasts for Bitcoin, compared with standard Historical and")
    print("Monte Carlo VaR models, across different rolling estimation windows?")
    print("=" * 80)

    try:
        # Step 1: Load and prepare data
        print("\n" + "=" * 80)
        print("STEP 1: Loading and Preparing Data")
        print("=" * 80)
        data = prepare_btc_vix_data()

        # Set Date as index for models
        if "Date" in data.columns:
            data = data.set_index("Date")

        print(f"\nData loaded successfully!")
        print(f"  Date range: {data.index.min()} to {data.index.max()}")
        print(f"  Total observations: {len(data)}")

        # Step 2: Calculate VaR for all models
        print("\n" + "=" * 80)
        print("STEP 2: Calculating VaR Forecasts")
        print("=" * 80)

        print("\n[1/3] Historical VaR Model")
        print("-" * 40)
        historical_results = calculate_historical_var(data)

        print("\n[2/3] Monte Carlo VaR Model")
        print("-" * 40)
        montecarlo_results = calculate_monte_carlo_var(data)

        print("\n[3/3] VIX-Regression VaR Model (with lagged VIX)")
        print("-" * 40)
        vix_regression_results = calculate_vix_regression_var(data)

        print("\n✓ All VaR models computed successfully!")

        # Step 3: Run Kupiec tests
        print("\n" + "=" * 80)
        print("STEP 3: Running Kupiec Tests")
        print("=" * 80)

        print("\n[1/3] Testing Historical VaR model...")
        historical_kupiec = run_kupiec_tests_for_model(
            data, historical_results, "Historical"
        )

        print("\n[2/3] Testing Monte Carlo VaR model...")
        montecarlo_kupiec = run_kupiec_tests_for_model(
            data, montecarlo_results, "MonteCarlo"
        )

        print("\n[3/3] Testing VIX-Regression VaR model...")
        vix_regression_kupiec = run_kupiec_tests_for_model(
            data, vix_regression_results, "VIXRegression"
        )

        print("\n✓ All Kupiec tests completed!")

        # Step 4: Combine results and generate comparisons
        print("\n" + "=" * 80)
        print("STEP 4: Generating Model Comparisons")
        print("=" * 80)

        # Combine all Kupiec results
        all_kupiec = combine_kupiec_results([
            historical_kupiec,
            montecarlo_kupiec,
            vix_regression_kupiec
        ])

        # Generate comparison statistics
        comparison_stats = compare_models(all_kupiec)
        rankings = rank_models_by_coverage(comparison_stats)

        print("\n✓ Model comparisons generated!")

        # Step 5: Save results
        print("\n" + "=" * 80)
        print("STEP 5: Saving Results")
        print("=" * 80)

        config.ensure_results_dir()
        save_all_results(all_kupiec, comparison_stats, rankings)

        # Print summary
        print_summary(comparison_stats, rankings)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {config.RESULTS_DIR}")
        print("\nKey files:")
        print(f"  - BTC_Kupiec_Results_All.csv (all Kupiec test results)")
        print(f"  - comparisons/model_comparison.csv (detailed comparison)")
        print(f"  - comparisons/model_rankings.csv (model rankings)")
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
