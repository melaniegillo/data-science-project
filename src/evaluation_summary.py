import os
import shutil
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir)
results_dir = os.path.join(project_dir, "results")
os.makedirs(results_dir, exist_ok=True)

files = [
    "BTC_Kupiec_Results_Historical.csv",
    "BTC_Kupiec_Results_VIXRegression.csv",
    "BTC_Kupiec_Results_MonteCarlo.csv",
]

dfs = []
for fname in files:
    path = os.path.join(base_dir, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
        dfs.append(df)
    else:
        print(f"Warning: {fname} not found or empty in {base_dir}, skipping.")

if not dfs:
    print(f"No results files found in {base_dir}. Nothing to summarize.")
else:
    combined = pd.concat(dfs, ignore_index=True)

    sort_cols = [c for c in ["ConfidenceLevel", "RollingWindow", "Model"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols)

    out_path = os.path.join(results_dir, "BTC_Kupiec_Results_All.csv")
    combined.to_csv(out_path, index=False)

    for fname in files:
        src_path = os.path.join(base_dir, fname)
        if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
            dst_path = os.path.join(results_dir, fname)
            shutil.copy(src_path, dst_path)

    print("Combined Kupiec results:")
    print(combined)
    print(f"\nSaved combined table to: {out_path}")
    print(f"Individual result files also copied to: {results_dir}")
