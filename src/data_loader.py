import os
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir)

processed_dir = os.path.join(project_dir, "data", "processed")

btc_input_file = os.path.join(processed_dir, "dataprocessedbtc_daily.xlsx")
vix_input_file = os.path.join(processed_dir, "dataprocessedvix_daily.xlsx")

base_output_folder = os.path.join(base_dir, "CSV_BTCVIX")
os.makedirs(base_output_folder, exist_ok=True)

def prepare_btc_vix_data():
    btc = pd.read_excel(btc_input_file)
    vix = pd.read_excel(vix_input_file)

    btc["Date"] = pd.to_datetime(btc["Date"])
    vix["Date"] = pd.to_datetime(vix["Date"])

    btc = btc.sort_values("Date")
    vix = vix.sort_values("Date")

    data = pd.merge(btc, vix, on="Date", how="inner")

    data = data.dropna(subset=["btc_price", "vix_level"]).copy()

    data["Returns"] = np.log(data["btc_price"]).diff()

    data = data.dropna(subset=["Returns"]).copy()

    window = 21
    data["RealizedVol_21d"] = data["Returns"].rolling(window).std() * np.sqrt(252)

    data["VIX_decimal"] = data["vix_level"] / 100.0

    data["VIX_change"] = data["vix_level"].diff()

    output_file = os.path.join(base_output_folder, "BTC_VIX_returns.csv")
    data.to_csv(output_file, index=False)

    print(f"BTC + VIX merged data with returns saved to: {output_file}")
    print(f"Number of observations: {len(data)}")
    print("Columns:", list(data.columns))

if __name__ == "__main__":
    prepare_btc_vix_data()
