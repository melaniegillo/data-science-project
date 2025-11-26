import os
import math
import numpy as np
import pandas as pd
from RollingWindows import rolling_windows

base_dir = os.path.dirname(os.path.abspath(__file__))

returns_file = os.path.join(base_dir, "CSV_BTCVIX", "BTC_VIX_returns.csv")
vix_var_base = os.path.join(base_dir, "CSV_BTC_VIXRegressionEvaluation")

confidence_levels = [0.95, 0.99]

z_scores = {
    0.95: 1.6448536269514722,
    0.99: 2.3263478740408408,
}


def chi2_df1_p_value(LR_uc: float) -> float:
    if not math.isfinite(LR_uc) or LR_uc < 0:
        return 0.0
    cdf = math.erf(math.sqrt(LR_uc / 2.0))
    return 1.0 - cdf


def compute_vix_regression_var(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    df = df.dropna(subset=["RealizedVol_21d", "VIX_decimal"]).copy()
    df = df.sort_index()

    x = df["VIX_decimal"].values
    y = df["RealizedVol_21d"].values
    dates = df.index

    start = window_size
    out_dates = []
    var_95 = []
    var_99 = []

    for i in range(start, len(df)):
        x_win = x[i - window_size : i]
        y_win = y[i - window_size : i]

        if np.isnan(x_win).any() or np.isnan(y_win).any():
            continue

        slope, intercept = np.polyfit(x_win, y_win, 1)
        sigma_ann = intercept + slope * x[i]

        if not np.isfinite(sigma_ann) or sigma_ann <= 0:
            continue

        sigma_daily = sigma_ann / math.sqrt(252.0)

        out_dates.append(dates[i])
        var_95.append(z_scores[0.95] * sigma_daily)
        var_99.append(z_scores[0.99] * sigma_daily)

    df_var = pd.DataFrame(
        {"VaR_95": var_95, "VaR_99": var_99},
        index=out_dates,
    )
    return df_var


def kupiec_on_df(df: pd.DataFrame, confidence_level: float, window_label: str):
    colname = f"VaR_{int(confidence_level * 100)}"
    if colname not in df.columns:
        return None

    tmp = df.dropna(subset=["Returns", colname]).copy()
    if tmp.empty:
        return None

    violations = (tmp["Returns"] < -tmp[colname]).astype(int)
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
        "Model": "VIXRegression",
        "RollingWindow": window_label,
        "ConfidenceLevel": confidence_level,
        "N": n,
        "Violations": x,
        "ExpectedViolations": expected_violations,
        "LR_uc": LR_uc,
        "p_value": p_value,
        "Reject_5pct": reject_5pct,
    }


def main():
    if not os.path.exists(returns_file):
        print(f"Returns file not found: {returns_file}")
        return

    returns_df = pd.read_csv(returns_file, parse_dates=["Date"])
    if "Returns" not in returns_df.columns:
        print("Column 'Returns' not found in BTC_VIX_returns.csv.")
        return

    returns_df = returns_df[
        ["Date", "Returns", "RealizedVol_21d", "VIX_decimal"]
    ].dropna(subset=["Returns"]).sort_values("Date")
    returns_df = returns_df.set_index("Date")

    os.makedirs(vix_var_base, exist_ok=True)

    results = []

    for window_label, window_size in rolling_windows.items():
        window_dir = os.path.join(vix_var_base, window_label)
        os.makedirs(window_dir, exist_ok=True)

        print(f"Computing VIX regression VaR for window {window_label} (size {window_size})...")
        var_df = compute_vix_regression_var(returns_df, int(window_size))

        var_path = os.path.join(window_dir, "BTC_VIXRegressionVaR.csv")
        var_df.to_csv(var_path)
        print(f"Saved VIX regression VaR to: {var_path}")

        merged = returns_df.join(var_df, how="inner")

        for cl in confidence_levels:
            res = kupiec_on_df(merged, cl, window_label)
            if res is not None:
                results.append(res)

    output_file = os.path.join(base_dir, "BTC_Kupiec_Results_VIXRegression.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    print("\nKupiec test (VIX regression) finished. Results:")
    print(results_df)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
