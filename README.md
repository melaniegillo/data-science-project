# Bitcoin Value-at-Risk Forecasting

A Python project that compares three Value-at-Risk (VaR) forecasting models for Bitcoin returns, investigating whether incorporating VIX information improves forecast accuracy.

## Research Question

**Does incorporating information from the VIX improve the accuracy of one-day Value-at-Risk forecasts for Bitcoin, compared with standard Historical and Monte Carlo VaR models, across different rolling estimation windows?**

## Project Overview

This project implements and evaluates three VaR forecasting models:

1. **Historical VaR**: Non-parametric approach using empirical quantiles of past returns
2. **Monte Carlo VaR**: Parametric approach assuming normally distributed returns
3. **VIX-Regression VaR**: Uses the CBOE Volatility Index (VIX) to predict Bitcoin volatility

All models are evaluated using:
- Kupiec unconditional coverage test
- Model comparison statistics (violation rates, deviations)
- Rankings across different rolling windows (1-month, 3-month, 6-month, 12-month)

### Key Feature: Lagged VIX for True Forecasting

**Critical Implementation Detail**: The VIX-Regression model uses **lagged VIX** (VIX from day t-1) to forecast volatility at day t. This avoids look-ahead bias and creates a genuine out-of-sample forecast, making it comparable to the Historical and Monte Carlo models.

## Installation

### Prerequisites

- Python >= 3.10
- `uv` package manager (recommended) or `pip`

### Setup Instructions

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd data-science-project
   ```

2. **Install dependencies using uv**:
   ```bash
   # Install uv if you don't have it
   pip install uv

   # Install project dependencies
   uv sync
   ```

   Alternatively, with pip:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Verify installation**:
   ```bash
   # Run tests
   uv run pytest

   # All 10 tests should pass
   ```

## Usage

### Run the Complete Analysis

```bash
uv run python main.py
```

This executes the full pipeline:
1. Loads and prepares BTC and VIX data
2. Calculates VaR forecasts for all three models
3. Runs Kupiec tests for model validation
4. Generates model comparison statistics
5. Saves results to `results/` directory

### Expected Output

The script will:
- Display progress through 5 steps
- Print a model comparison summary
- Save results to `results/` directory:
  - `BTC_Kupiec_Results_All.csv` - Combined Kupiec test results
  - `comparisons/model_comparison.csv` - Detailed comparison statistics
  - `comparisons/model_rankings.csv` - Model rankings by performance

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_models.py -v
```

## Project Structure

```
data-science-project/
├── README.md                   # This file
├── AI_LOG.md                   # AI usage documentation (required)
├── pyproject.toml              # Project configuration and dependencies
├── main.py                     # Main entry point
│
├── data/                       # Data files (not in git)
│   ├── raw/                    # Original BTC and VIX data
│   └── processed/              # Cleaned and merged data
│
├── src/                        # Source code
│   ├── config.py               # Centralized configuration
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── models_historical.py    # Historical VaR model
│   ├── models_monte_carlo.py   # Monte Carlo VaR model
│   ├── models_vix_regression.py # VIX-Regression VaR model (with lag fix)
│   ├── evaluation_kupiec.py    # Kupiec test implementation
│   └── evaluation_summary.py   # Model comparison and ranking
│
├── tests/                      # Test suite (10 tests)
│   ├── test_data_loader.py     # Data loading tests
│   ├── test_models.py          # VaR model tests (including VIX lag verification)
│   └── test_evaluation.py      # Evaluation function tests
│
└── results/                    # Output directory (created on first run)
    ├── BTC_Kupiec_Results_All.csv
    ├── comparisons/
    │   ├── model_comparison.csv
    │   └── model_rankings.csv
    └── var_forecasts/          # Individual VaR forecasts per model/window
```

## Models Explained

### 1. Historical VaR

**Method**: Empirical quantile estimation

At time t, VaR is calculated as the negative of the quantile of past returns from a rolling window:
```
VaR_95(t) = -quantile(returns[t-window:t-1], 0.05)
```

**Advantages**: Non-parametric, no distribution assumptions
**Disadvantages**: Requires long history, slow to adapt to regime changes

### 2. Monte Carlo VaR

**Method**: Parametric simulation assuming normal distribution

At time t:
1. Estimate mean (μ) and std dev (σ) from rolling window
2. Simulate 100,000 returns from N(μ, σ²)
3. Calculate VaR as quantile of simulations

**Advantages**: Smooth estimates, easy to implement
**Disadvantages**: Assumes normality (poor for crypto with fat tails)

### 3. VIX-Regression VaR

**Method**: Regression-based volatility forecasting

At time t:
1. Fit regression: `realized_volatility ~ VIX` on rolling window
2. **Use VIX(t-1)** to predict volatility for time t (avoiding look-ahead bias)
3. Convert predicted volatility to VaR using normal quantiles

**Key Implementation**:
```python
# At time i, fit regression on window [i-window_size:i]
slope, intercept = fit_regression(VIX[i-window:i], realized_vol[i-window:i])

# CRITICAL: Use lagged VIX for true forecast
predicted_volatility = intercept + slope * VIX[i-1]  # Not VIX[i]!

VaR = z_score * predicted_volatility
```

**Why Lagging Matters**: Using VIX[i] would be look-ahead bias (using information not available at forecast time). Using VIX[i-1] makes this a genuine forecast.

## Evaluation Methodology

### Kupiec Test

The Kupiec unconditional coverage test checks if the observed violation rate matches the expected rate.

- **Null Hypothesis**: Violation rate = expected rate (e.g., 5% for 95% confidence)
- **Test Statistic**: Likelihood ratio statistic
- **P-value**: From chi-squared distribution (df=1)
- **Decision**: Reject if p-value < 0.05

### Model Comparison

Models are compared on:
1. **Violation Rate**: How closely actual violations match expected (5% for 95% VaR)
2. **Absolute Deviation**: Distance from expected violations (lower is better)
3. **Coverage Accuracy**: Ranking based on absolute deviation

**Best Model**: The one with lowest absolute deviation from expected violations across windows.

## Results Interpretation

After running `main.py`, check the model comparison summary printed to console and saved in `results/comparisons/`.

### Key Questions to Answer

1. **Which model performs best overall?**
   - Check `results/comparisons/model_rankings.csv`
   - Look for model with lowest average rank

2. **Does VIX information help?**
   - Compare VIXRegression ranks to Historical and MonteCarlo
   - If VIXRegression ranks #1 frequently, VIX helps

3. **Does performance vary by window size?**
   - Check if certain models perform better with longer/shorter windows
   - Look at rankings across 1m, 3m, 6m, 12m windows

4. **Are violations statistically acceptable?**
   - Check Kupiec test p-values
   - p-value < 0.05 means rejection (model coverage is problematic)
   - p-value >= 0.05 means acceptance (model coverage is adequate)

## Dependencies

### Core Dependencies
- `pandas` >= 2.0.0 - Data manipulation
- `numpy` >= 1.24.0 - Numerical operations
- `scipy` >= 1.10.0 - Statistical functions (chi-squared test)
- `matplotlib` >= 3.7.0 - Plotting (future use)
- `openpyxl` >= 3.1.0 - Excel file reading

### Development Dependencies
- `pytest` >= 7.4.0 - Testing framework
- `ruff` >= 0.1.0 - Linting and formatting

## Testing

The project includes 10 comprehensive tests:

**Data Loader Tests** (2):
- Verify data loading and preprocessing
- **Verify VIX_lag1 is correctly lagged** (critical!)

**Model Tests** (4):
- Test output format for all models
- **Test VIX regression uses lagged VIX** (critical bug fix verification)
- Test VaR_99 >= VaR_95 (sanity check)

**Evaluation Tests** (4):
- Test Kupiec test calculation
- Test edge cases (no violations, etc.)
- Test model comparison functions
- Test ranking logic

Run tests:
```bash
uv run pytest -v
```

All tests should pass.

## AI Usage

This project was developed with assistance from Claude Code (Claude Sonnet 4.5). Full documentation of AI-assisted code is available in `AI_LOG.md`, as required by course policy.

### Key AI-Generated Components

- Configuration module
- Data loader with error handling
- All three VaR models (with critical VIX lag fix)
- Evaluation framework
- Comprehensive test suite

**Critical Fix**: The VIX regression model originally had a look-ahead bias bug (using contemporaneous VIX). This was identified and fixed with AI assistance to use lagged VIX for true forecasting.

For detailed AI usage log, see `AI_LOG.md`.

## Code Quality

- **Type Hints**: All functions have type annotations
- **Docstrings**: Comprehensive documentation for all modules, classes, and functions
- **Error Handling**: Try/except blocks for file I/O and data validation
- **Testing**: 10 unit tests with 100% pass rate
- **Formatting**: Code formatted with `ruff`

Run code formatter:
```bash
uv run ruff format .
```

Run linter:
```bash
uv run ruff check .
```

## License

This is a university course project. All rights reserved.

## Contact

For questions about this project, please refer to the course materials or contact the project author.

---

**Note**: This project is for educational purposes as part of a Python programming course. The VaR models are simplified implementations for demonstration and should not be used for actual trading or risk management without significant enhancement and validation.
