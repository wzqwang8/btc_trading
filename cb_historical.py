"""
Coinbase Historical Price Viewer
=================================
Fetches OHLCV candle data from the Coinbase Exchange public API
and saves it to a CSV for trend analysis.

Usage:
  python3 cb_historical.py                              # BTC-GBP, daily, all time
  python3 cb_historical.py --pair ETH-GBP               # different pair
  python3 cb_historical.py --pair BTC-GBP --days 90     # last 90 days
  python3 cb_historical.py --pair BTC-GBP --granularity 3600  # hourly
  python3 cb_historical.py --start 2024-01-01 --end 2024-12-31

Granularity options (seconds):
  60      = 1 minute
  300     = 5 minutes
  900     = 15 minutes
  3600    = 1 hour
  21600   = 6 hours
  86400   = 1 day  (default)
"""

import argparse
import time
import requests
import pandas as pd
from datetime import datetime, timedelta


def fetch_candles(product_id: str, start: datetime, end: datetime, granularity: int) -> pd.DataFrame:
    """
    Fetch OHLCV candles from the Coinbase Exchange public API in batches
    (max 300 candles per request).
    """
    batch_size = timedelta(seconds=granularity * 300)
    all_candles = []
    current = start

    while current < end:
        batch_end = min(current + batch_size, end)
        resp = requests.get(
            f"https://api.exchange.coinbase.com/products/{product_id}/candles",
            params={
                "start": current.isoformat(),
                "end": batch_end.isoformat(),
                "granularity": granularity,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"  ⚠️  API error {resp.status_code} for {current.date()} → {batch_end.date()}")
            break
        data = resp.json()
        if data:
            all_candles.extend(data)
        current = batch_end
        time.sleep(0.35)  # respect rate limit

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame, pair: str):
    """Print a quick stats summary to the console."""
    if df.empty:
        print("  No data.")
        return
    print(f"\n  {pair} — {len(df)} candles")
    print(f"  From:    {df['time'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"  To:      {df['time'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  High:    £{df['high'].max():,.2f}")
    print(f"  Low:     £{df['low'].min():,.2f}")
    print(f"  Latest:  £{df['close'].iloc[-1]:,.2f}")
    chg = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"  Change:  {chg:+.1f}% over period\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical OHLCV candles from Coinbase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pair", default="BTC-GBP", help="Trading pair (default: BTC-GBP)")
    parser.add_argument("--days", type=int, default=None, help="Last N days (overrides --start/--end)")
    parser.add_argument("--start", default="2013-01-01", help="Start date YYYY-MM-DD (default: 2013-01-01)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--granularity",
        type=int,
        default=86400,
        choices=[60, 300, 900, 3600, 21600, 86400],
        help="Candle size in seconds (default: 86400 = daily)",
    )
    parser.add_argument("--no-save", action="store_true", help="Print only, don't save CSV")
    args = parser.parse_args()

    end_dt = datetime.now() if args.end is None else datetime.fromisoformat(args.end)
    if args.days:
        start_dt = end_dt - timedelta(days=args.days)
    else:
        start_dt = datetime.fromisoformat(args.start)

    gran_labels = {60: "1m", 300: "5m", 900: "15m", 3600: "1h", 21600: "6h", 86400: "1d"}
    gran_label = gran_labels[args.granularity]

    print(f"\n📈 Fetching {args.pair} candles ({gran_label}) from {start_dt.date()} to {end_dt.date()}...")
    df = fetch_candles(args.pair, start_dt, end_dt, args.granularity)

    print_summary(df, args.pair)
    print(df.tail(10).to_string(index=False))

    if not args.no_save and not df.empty:
        filename = f"{args.pair.lower().replace('-', '_')}_{gran_label}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n  💾 Saved to {filename}")


if __name__ == "__main__":
    main()
