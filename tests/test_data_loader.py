"""
Tests for data loading functionality.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import DataLoadError, prepare_btc_vix_data


def test_prepare_btc_vix_data():
    """Test that data preparation runs without errors."""
    try:
        data = prepare_btc_vix_data()

        # Check that data is returned
        assert data is not None
        assert isinstance(data, pd.DataFrame)

        # Check required columns exist
        required_cols = ['Returns', 'VIX_decimal', 'RealizedVol_21d', 'VIX_lag1']
        for col in required_cols:
            assert col in data.columns, f"Missing required column: {col}"

        # Check data has rows
        assert len(data) > 0, "Data should not be empty"

        # Check VIX_lag1 is actually lagged (should be shifted by 1)
        # VIX_lag1[i] should equal VIX_decimal[i-1]
        data_clean = data.dropna(subset=['VIX_decimal', 'VIX_lag1'])
        if len(data_clean) > 1:
            # Check a few values to ensure lagging is correct
            for i in range(1, min(10, len(data_clean))):
                vix_current = data_clean['VIX_decimal'].iloc[i-1]
                vix_lag = data_clean['VIX_lag1'].iloc[i]
                assert np.isclose(vix_lag, vix_current, rtol=1e-9), \
                    f"VIX_lag1 should be lagged version of VIX_decimal"

        print("✓ Data loader test passed")

    except DataLoadError as e:
        # If data files don't exist, that's acceptable for testing
        pytest.skip(f"Data files not available: {e}")


def test_data_columns():
    """Test that all expected columns are present."""
    try:
        data = prepare_btc_vix_data()

        expected_columns = [
            'btc_price',
            'vix_level',
            'Returns',
            'RealizedVol_21d',
            'VIX_decimal',
            'VIX_change',
            'VIX_lag1'
        ]

        for col in expected_columns:
            assert col in data.columns, f"Expected column {col} not found"

        print("✓ Data columns test passed")

    except DataLoadError:
        pytest.skip("Data files not available")


if __name__ == "__main__":
    print("Running data loader tests...")
    test_prepare_btc_vix_data()
    test_data_columns()
    print("\n✓ All data loader tests passed!")
