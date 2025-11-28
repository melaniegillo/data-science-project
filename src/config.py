"""
Configuration module for Bitcoin VaR forecasting project.

Contains all project-wide configuration including paths, rolling windows,
confidence levels, and model parameters.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SRC_DIR: Path = PROJECT_ROOT / "src"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Data file names
BTC_PROCESSED_FILE: Path = PROCESSED_DATA_DIR / "dataprocessedbtc_daily.xlsx"
VIX_PROCESSED_FILE: Path = PROCESSED_DATA_DIR / "dataprocessedvix_daily.xlsx"

# Rolling window configurations (in trading days)
ROLLING_WINDOWS: dict[str, int] = {
    "1m": 21,  # 1 month
    "3m": 63,  # 3 months
    "6m": 126,  # 6 months
    "12m": 252,  # 12 months (1 year)
}

# Confidence levels for VaR calculations
CONFIDENCE_LEVELS: list[float] = [0.95, 0.99]

# Z-scores for normal distribution quantiles
Z_SCORES: dict[float, float] = {
    0.95: 1.6448536269514722,
    0.99: 2.3263478740408408,
}

# Monte Carlo simulation parameters
MONTE_CARLO_SIMS: int = 100_000
RANDOM_SEED: int = 42

# VIX regression parameters
VIX_LAG_DAYS: int = 1  # Use VIX from t-1 to forecast time t

# Volatility calculation parameters
REALIZED_VOL_WINDOW: int = 21  # Days for realized volatility calculation
TRADING_DAYS_PER_YEAR: int = 252

# Backtesting configuration
SIGNIFICANCE_LEVEL: float = 0.05  # For hypothesis tests (5% significance)
MIN_OBSERVATIONS_KUPIEC: int = 5  # Minimum observations for Kupiec test validity
MIN_OBSERVATIONS_CHRISTOFFERSEN: int = 200  # Minimum for independence test

# Output configuration
RESULTS_DECIMAL_PLACES: int = 4  # Decimal places in output CSVs


def ensure_results_dir() -> None:
    """Create results directory and subdirectories if they don't exist."""
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "comparisons").mkdir(exist_ok=True)
    (RESULTS_DIR / "var_forecasts").mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)


if __name__ == "__main__":
    # Print configuration for debugging
    print("Bitcoin VaR Forecasting - Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"\nRolling Windows: {ROLLING_WINDOWS}")
    print(f"Confidence Levels: {CONFIDENCE_LEVELS}")
    print(f"VIX Lag Days: {VIX_LAG_DAYS}")
    print(f"Monte Carlo Simulations: {MONTE_CARLO_SIMS:,}")
