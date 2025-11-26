import os
import math
import numpy as np
import pandas as pd
from RollingWindows import rolling_windows

base_dir = os.path.dirname(__file__)
hist_var_base = os.path.join(base_dir, "CSV_BTC_HistoricalEvaluation")
confidence_levels = [0.95, 0.99]


def chi2_df1_p_value(LR_uc: float) -> float:
    if not math.isfinite(LR_uc) or LR_uc < 0:
        return 0.0
    cdf = math.erf(math.sqrt(LR_uc / 2.0))
    return 1.0 - cdf


def kupiec_test(df: pd.DataFrame, confidence_level: float):
    colname = f"VaR_{int(confidence_level * 100)}"
    df = df.dropna(subset=["Returns", colname]).copy()
    if df.empty:
        return None

    violations = (df["Returns"] < -df[colname]).astype(int)
    x = int(violations.sum())
    n = int(len(violations))
    p0 = 1.0 - confidence_level
    if n == 0:
        return None

    if x == 0 or x == n:
        LR_uc = float("inf")
        p_value = 0.0
    else:
        p_hat = x / n
        logL0 = (n - x) * np.log(1 - p0) + x * np.log(p0)
        logL1 = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
        LR_uc = -2.0 * (logL0 - logL1)
        p_value = chi2_df1_p_value(LR_uc)

    expected_violations = n * p0
    reject_5pct = p_value < 0.05

    return {
        "Model": "Historical",
        "RollingWindow": "",
        "ConfidenceLevel": confidence_level,
        "N": n,
        "Violations": x,
        "ExpectedViolations": expected_violations,
        "LR_uc": LR_uc,
        "p_value": p_value,
        "Reject_5pct": reject_5pct,
    }


results = []

for window_label in rolling_windows.keys():
    window_folder = os.path.join(hist_var_base, window_label)
    file_path = os.path.join(window_folder, "BTC_HistVaR.csv")

    if not os.path.exists(file_path):
        print(f"File not found for window {window_label}, skipping.")
        continue

    print(f"Running Kupiec test for window {window_label}...")
    df = pd.read_csv(file_path, parse_dates=["Date"])

    for cl in confidence_levels:
        res = kupiec_test(df, cl)
        if res is not None:
            res["RollingWindow"] = window_label
            results.append(res)

output_file = os.path.join(base_dir, "BTC_Kupiec_Results_Historical.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

print("\nKupiec test finished. Results:")
print(results_df)
print(f"\nSaved to: {output_file}")
