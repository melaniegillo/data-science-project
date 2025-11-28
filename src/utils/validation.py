"""
Common validation functions used across VaR models.

This module contains shared validation logic to eliminate code duplication
and ensure consistent input validation across all models.
"""

import pandas as pd
from src import config

__all__ = ["validate_model_inputs", "validate_required_columns"]


def validate_model_inputs(
    data: pd.DataFrame,
    rolling_windows: dict[str, int] | None,
    confidence_levels: list[float] | None,
) -> tuple[dict[str, int], list[float]]:
    """
    Validate inputs common to all VaR models.

    Checks:
    - Data is not None or empty
    - Window sizes are positive and within data length
    - Confidence levels are in valid range (0, 1)

    Args:
        data: Price/return data DataFrame
        rolling_windows: Window sizes dict or None (uses config default)
        confidence_levels: Confidence levels list or None (uses config default)

    Returns:
        Tuple of (validated_windows, validated_confidence_levels)

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate data
    if data is None or len(data) == 0:
        raise ValueError("data cannot be None or empty")

    # Set defaults from config if not provided
    if rolling_windows is None:
        rolling_windows = config.ROLLING_WINDOWS
    if confidence_levels is None:
        confidence_levels = config.CONFIDENCE_LEVELS

    # Validate window sizes
    for label, window_size in rolling_windows.items():
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size} for {label}")
        if window_size > len(data):
            raise ValueError(
                f"Window size {window_size} for {label} exceeds data length {len(data)}"
            )

    # Validate confidence levels
    for cl in confidence_levels:
        if not 0 < cl < 1:
            raise ValueError(f"Confidence level must be in (0, 1), got {cl}")

    return rolling_windows, confidence_levels


def validate_required_columns(data: pd.DataFrame, required: list[str]) -> None:
    """
    Check DataFrame has required columns.

    Args:
        data: DataFrame to check
        required: List of required column names

    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
