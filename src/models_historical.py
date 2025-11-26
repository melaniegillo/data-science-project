import os
import numpy as np
import pandas as pd
from RollingWindows import rolling_windows

base_dir = os.path.dirname(__file__)

input_file = os.path.join(base_dir, "CSV_BTCVIX", "BTC_VIX_returns.csv")

base_output_folder = os.path.join(base_dir, "CSV_BTC_HistoricalEvaluation")
if not os.path.exists(base_output_folder):
    os.makedirs(base_output_folder)

confidence_levels = [0.95, 0.99]


def calculate_var(returns_window, confidence_level):
    """
    Historical VaR for one confidence level.
    VaR is returned as a positive loss: -quantile of past returns.
    """
    q = np.quantile(returns_window, 1 - confidence_level)
    return -q


def compute_historical_var_for_window(data, window_size):
    """
    Out-of-sample Historical VaR for a given rolling window size.
    VaR at time t uses returns from [t-window_size, ..., t-1].
    """
    df = data.copy().reset_index(drop=True)

    for cl in confidence_levels:
        df[f"VaR_{int(cl * 100)}"] = np.nan

    for i in range(window_size, len(df)):
        window_returns = df.loc[i - window_size:i - 1, "Returns"]

        for cl in confidence_levels:
            colname = f"VaR_{int(cl * 100)}"
            df.at[i, colname] = calculate_var(window_returns, cl)

    return df



data = pd.read_csv(input_file, parse_dates=["Date"])
data = data.sort_values("Date").reset_index(drop=True)

for window_label, window_size in rolling_windows.items():
    print(f"Computing Historical VaR for window {window_label} ({window_size} days)...")

    df_var = compute_historical_var_for_window(data, window_size)

    cols_to_save = ["Date", "Returns", "vix_level"]
    for cl in confidence_levels:
        cols_to_save.append(f"VaR_{int(cl * 100)}")

    df_out = df_var[cols_to_save]

    rolling_output_folder = os.path.join(base_output_folder, window_label)
    if not os.path.exists(rolling_output_folder):
        os.makedirs(rolling_output_folder)

    output_file = os.path.join(rolling_output_folder, "BTC_HistVaR.csv")
    df_out.to_csv(output_file, index=False)

    print(f"  -> Saved to {output_file}")

print("Historical VaR calculation for BTC completed.")
