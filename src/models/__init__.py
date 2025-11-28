"""
VaR models for Bitcoin volatility forecasting.

This package contains implementations of three VaR models:
- Historical VaR: Non-parametric approach using empirical quantiles
- Monte Carlo VaR: Parametric simulation assuming normal distribution
- VIX-Regression VaR: Uses VIX index to predict Bitcoin volatility
"""

from src.models.historical import calculate_historical_var
from src.models.monte_carlo import calculate_monte_carlo_var
from src.models.vix_regression import calculate_vix_regression_var

__all__ = [
    "calculate_historical_var",
    "calculate_monte_carlo_var",
    "calculate_vix_regression_var",
]
