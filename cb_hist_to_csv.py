import requests
import pandas as pd
from datetime import datetime, timedelta
import time


def fetch_historical_candles(product_id, start_date, end_date, granularity):
    """
    Fetches historical candlestick data from Coinbase API in batches.
    Parameters:
    - product_id (str): The trading pair (e.g., 'BTC-USD').
    - start_date (datetime): The start date for data retrieval.
    - end_date (datetime): The end date for data retrieval.
    - granularity (int): The granularity in seconds (e.g., 60 for 1 minute).

    Returns:
    - pd.DataFrame: A DataFrame containing the historical candlestick data.
    """
    delta = timedelta(seconds=granularity * 300)  # Maximum 300 data points per request
    all_candles = []

    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + delta, end_date)
        params = {
            'start': current_start.isoformat(),
            'end': current_end.isoformat(),
            'granularity': granularity
        }
        response = requests.get(f'https://api.exchange.coinbase.com/products/{product_id}/candles', params=params)
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            break
        data = response.json()
        if not data:
            print(f"No data returned for {current_start} to {current_end}")
        all_candles.extend(data)
        current_start = current_end
        time.sleep(0.34)  # To respect the API rate limit

    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    return df 


# Example usage:
start = datetime(2015, 5, 1)
end = datetime(2025, 5, 1)
granularity = 3600  # 1 day
product = 'BTC-GBP'
df = fetch_historical_candles(product, start, end, granularity)
df.to_csv("btc_gbp_hourly.csv", index=False)
print(df)
print(len(df))

