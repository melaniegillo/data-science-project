"""
Data loading and preprocessing module for Bitcoin VaR forecasting.

This module handles loading Bitcoin and VIX data, merging them,
and calculating necessary features like returns, realized volatility,
and lagged VIX values for forecasting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src import config
from src.validation import validate_required_columns


class DataLoadError(Exception):
    """Custom exception for data loading errors."""

    pass


def _load_excel_with_validation(
    file_path: Path, required_columns: list[str], data_name: str
) -> pd.DataFrame:
    """
    Generic Excel file loader with validation (eliminates code duplication).

    This helper function encapsulates the common pattern for loading Excel files:
    - Check file exists
    - Read Excel
    - Validate required columns
    - Process Date column
    - Sort and reset index

    Args:
        file_path: Path to Excel file
        required_columns: List of required column names (must include 'Date')
        data_name: Name for error messages and logging (e.g., "BTC", "VIX")

    Returns:
        DataFrame with validated columns, sorted by Date

    Raises:
        DataLoadError: If file not found, columns missing, or other errors
    """
    try:
        # Check file exists
        if not file_path.exists():
            raise DataLoadError(f"{data_name} data file not found: {file_path}")

        # Read Excel file
        df = pd.read_excel(file_path)

        # Validate required columns using shared validation
        try:
            validate_required_columns(df, required_columns)
        except ValueError as e:
            raise DataLoadError(f"{data_name} data {str(e)}")

        # Process Date column and sort
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        print(f"✓ Loaded {data_name} data: {len(df)} rows")
        return df

    except FileNotFoundError:
        raise DataLoadError(f"{data_name} data file not found: {file_path}")
    except DataLoadError:
        raise  # Re-raise our custom errors
    except Exception as e:
        raise DataLoadError(f"Error loading {data_name} data: {str(e)}")


def load_btc_data() -> pd.DataFrame:
    """
    Load Bitcoin price data from processed Excel file.

    Returns:
        pd.DataFrame: Bitcoin data with Date and btc_price columns

    Raises:
        DataLoadError: If file not found or required columns missing
    """
    return _load_excel_with_validation(
        file_path=config.BTC_PROCESSED_FILE,
        required_columns=["Date", "btc_price"],
        data_name="BTC",
    )


def load_vix_data() -> pd.DataFrame:
    """
    Load VIX index data from processed Excel file.

    Returns:
        pd.DataFrame: VIX data with Date and vix_level columns

    Raises:
        DataLoadError: If file not found or required columns missing
    """
    return _load_excel_with_validation(
        file_path=config.VIX_PROCESSED_FILE,
        required_columns=["Date", "vix_level"],
        data_name="VIX",
    )


def prepare_btc_vix_data() -> pd.DataFrame:
    """
    Load, merge, and prepare Bitcoin and VIX data for VaR analysis.

    This function:
    1. Loads BTC and VIX data
    2. Merges on Date (inner join)
    3. Calculates log returns
    4. Calculates realized volatility
    5. Creates VIX features (decimal, change, lagged)
    6. Saves merged data to CSV

    Returns:
        pd.DataFrame: Merged and processed data with all features

    Raises:
        DataLoadError: If data loading or processing fails
    """
    print("\n" + "=" * 60)
    print("Loading and Preparing BTC + VIX Data")
    print("=" * 60)

    try:
        # Load data
        btc = load_btc_data()
        vix = load_vix_data()

        # Merge on Date
        print("\nMerging BTC and VIX data...")
        data = pd.merge(btc, vix, on="Date", how="inner")
        print(f"✓ Merged data: {len(data)} rows (matched dates)")

        # Drop rows with missing BTC or VIX values
        initial_len = len(data)
        data = data.dropna(subset=["btc_price", "vix_level"]).copy()
        if len(data) < initial_len:
            print(f"  Dropped {initial_len - len(data)} rows with missing values")

        # Calculate log returns
        print("\nCalculating features...")
        data["Returns"] = np.log(data["btc_price"]).diff()

        # Drop first row with NaN return
        data = data.dropna(subset=["Returns"]).copy()
        print(f"✓ Returns calculated: {len(data)} rows (after dropping first)")

        # Calculate realized volatility (21-day rolling)
        window = config.REALIZED_VOL_WINDOW
        data["RealizedVol_21d"] = data["Returns"].rolling(window).std() * np.sqrt(
            config.TRADING_DAYS_PER_YEAR
        )
        print(f"✓ Realized volatility ({window}-day window)")

        # VIX in decimal form (e.g., 20 -> 0.20)
        data["VIX_decimal"] = data["vix_level"] / 100.0

        # VIX change (for analysis)
        data["VIX_change"] = data["vix_level"].diff()

        # CRITICAL: Create lagged VIX for forecasting
        # This is used by the VIX regression model to make true forecasts
        lag_days = config.VIX_LAG_DAYS
        data[f"VIX_lag{lag_days}"] = data["VIX_decimal"].shift(lag_days)
        print(f"✓ Lagged VIX (lag={lag_days} days) for forecasting")

        # Validate data quality
        quality_issues = validate_data_quality(data)
        if quality_issues["total_issues"] > 0:
            print(f"\n⚠ Data quality check found {quality_issues['total_issues']} issues")
            print("Review warnings above - results may be affected\n")

        # Save to CSV
        output_folder = config.SRC_DIR / "CSV_BTCVIX"
        output_folder.mkdir(exist_ok=True)
        output_file = output_folder / "BTC_VIX_returns.csv"
        data.to_csv(output_file, index=False)

        print(f"\n✓ Data saved to: {output_file}")
        print(f"✓ Final dataset: {len(data)} rows")
        print(f"✓ Columns: {list(data.columns)}")
        print("=" * 60 + "\n")

        return data

    except Exception as e:
        raise DataLoadError(f"Error preparing data: {str(e)}")


def validate_data_quality(data: pd.DataFrame) -> dict[str, int]:
    """
    Validate data quality and return summary of issues.

    Checks for:
    - Duplicate dates
    - Negative VIX values (impossible)
    - Extreme returns (>50% absolute suggests data error)

    Args:
        data: DataFrame with Returns, VIX_decimal columns

    Returns:
        Dictionary with keys: duplicates, negative_vix, extreme_returns, total_issues
    """
    issues = {"duplicates": 0, "negative_vix": 0, "extreme_returns": 0, "total_issues": 0}

    # Check for duplicate dates
    if "Date" in data.columns:
        duplicates = data["Date"].duplicated().sum()
    else:
        duplicates = data.index.duplicated().sum()

    if duplicates > 0:
        issues["duplicates"] = duplicates
        print(f"⚠ WARNING: Found {duplicates} duplicate dates")

    # Check for negative VIX (impossible)
    if "VIX_decimal" in data.columns:
        negative_vix = (data["VIX_decimal"] < 0).sum()
        if negative_vix > 0:
            issues["negative_vix"] = negative_vix
            print(f"⚠ WARNING: Found {negative_vix} negative VIX values")

    # Check for extreme returns (> 50% or < -50% suggests data error)
    if "Returns" in data.columns:
        extreme = ((data["Returns"].abs() > 0.5) & data["Returns"].notna()).sum()
        if extreme > 0:
            issues["extreme_returns"] = extreme
            print(f"⚠ WARNING: Found {extreme} extreme returns (>50% absolute)")

    issues["total_issues"] = sum(v for k, v in issues.items() if k != "total_issues")

    return issues


if __name__ == "__main__":
    # Run data preparation when executed as script
    try:
        data = prepare_btc_vix_data()
        print("\n✓ Data preparation completed successfully!")
        print(f"\nData Summary:")
        print(data.describe())
    except DataLoadError as e:
        print(f"\n✗ Error: {e}")
        exit(1)
