# btc_trading

Coinbase trading tools and UK crypto tax calculator for Self Assessment, plus BTC technical analysis backtests.

## Tools

### `tax_return.py` — UK CGT Tax Calculator

Fetches your full Coinbase trade history and produces a tax report using HMRC's share-pooling rules:

1. Same-day match
2. 30-day bed & breakfast rule
3. Section 104 pool (average cost)

**Outputs:**
- `crypto_tax_report.pdf` — full multi-asset report (like Koinly)
- `SA108_YYYY-YY.pdf` — one per tax year, ready for HMRC Self Assessment
- `{pair}_uk_tax_report.csv` — per-asset disposal log

```bash
python3 tax_return.py                        # fetch live, report all years
python3 tax_return.py --tax-year 2024/25     # specific tax year only
python3 tax_return.py --use-cache            # skip API, use all_fills.csv
python3 tax_return.py --use-cache --tax-year 2024/25
```

---

### `cb_trader.py` — Trading CLI

Place and manage orders on Coinbase from the terminal. Confirms before every order.

```bash
python3 cb_trader.py status                          # account balances
python3 cb_trader.py price BTC-GBP                   # live price + 5% levels
python3 cb_trader.py buy   BTC-GBP 100               # market buy £100
python3 cb_trader.py sell  BTC-GBP 0.001             # market sell
python3 cb_trader.py limit-buy  BTC-GBP 100 80000    # limit buy £100 at £80k
python3 cb_trader.py limit-sell BTC-GBP 0.001 90000  # limit sell at £90k
python3 cb_trader.py orders                          # list open orders
python3 cb_trader.py cancel <order_id>               # cancel an order
```

---

### `cb_historical.py` — Historical Price Fetcher

Pulls OHLCV candle data from Coinbase's public API and saves it as a CSV. No API key needed.

```bash
python3 cb_historical.py                               # BTC-GBP daily, all time
python3 cb_historical.py --pair ETH-GBP                # different pair
python3 cb_historical.py --pair BTC-GBP --days 90      # last 90 days
python3 cb_historical.py --granularity 3600            # hourly candles
python3 cb_historical.py --start 2024-01-01 --end 2024-12-31
```

Granularity options: `60` (1m), `300` (5m), `900` (15m), `3600` (1h), `21600` (6h), `86400` (1d)

---

### `backtests/` — Technical Analysis

BTC backtests and indicator scripts using Refinitiv Eikon data.

| File | Description |
|---|---|
| `test_btc.py` | BTC trend-following backtest |
| `test_btc2.py` | Extended backtest with additional signals |
| `test_btc_cta.py` | CTA-style momentum strategy |
| `boll_bands.py` | Bollinger band analysis |
| `cb_hist.py` | Technical indicators (MA, RSI, ATR, OBV, Bollinger Bands) on local CSV |
| `BTC_ML.PY` | ML model comparison for BTC price |
| `btc_gbp_hourly.csv` | Historical hourly BTC-GBP OHLCV data |

---

## Setup

**Install dependencies:**
```bash
pip install coinbase-advanced-py pandas requests reportlab
```

**API key:**

Download your CDP API key from [Coinbase Developer Platform](https://developer.coinbase.com) and save it as `cdp_api_key.json` in this folder. This file is excluded from git.

The `cb_historical.py` script uses Coinbase's public REST API and does not require a key.
