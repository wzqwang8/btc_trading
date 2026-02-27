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

BTC backtests and indicator scripts. Eikon-based scripts require a Refinitiv API key set via the `EIKON_APP_KEY` environment variable. CSV-based scripts use `btc_gbp_hourly.csv` (generate it with `cb_historical.py`).

#### `test_btc.py` — Golden Cross / RSI Dip Backtest (long only)

Entry: golden cross (MA50 > MA200) or price dips >5% in 7 days with RSI < 40.
Exit: +10% profit target, 2×ATR stop loss, or 14-bar time stop.

Outputs a price chart with entry/exit markers and an equity curve subplot. Prints a full performance summary:

```
Total return:  +12.4%
Closed trades: 8
Win rate:      62.5%
Avg win:       £18.32
Avg loss:      £-9.10
Max drawdown:  -4.1%
Sharpe ratio:  1.43
```

#### `test_btc2.py` — Long/Short Volatility Backtest

Bidirectional strategy using a 7/20 MA crossover and RSI-filtered dip/surge signals. Exposure limits prevent over-concentration. Same performance stats and equity curve output as `test_btc.py`.

#### `test_btc_cta.py` — Multi-Timeframe MA Crossover Score

Scores 39 MA pairs simultaneously (+1 bullish crossover, −1 bearish). Normalises the score and combines it with an RSI filter to identify high-consensus entry windows. Plots price with signals and the normalised score as a second panel.

Configurable at the top of the file:
```python
LOOKBACK_DAYS        = 365
SCORE_BUY_THRESHOLD  =  0.05
SCORE_SELL_THRESHOLD = -0.05
```

Data source: `btc_gbp_hourly.csv`

#### `boll_bands.py` — Bollinger Band + MACD + RSI Signal Chart

4-panel chart: price with Bollinger Bands and MA50/200, MACD line, MACD histogram (green/red), RSI with overbought/oversold fill zones. Buy signals trigger on BB lower band touch or RSI < 30 with MACD confirmation.

Data source: Eikon

#### `cb_hist.py` — Technical Indicator Dashboard

5-panel dashboard from local CSV: price + MAs + Bollinger Bands, RSI with fill zones, ATR, rolling std dev, volume with spike highlighting and OBV on a twin axis.

```python
LOOKBACK_DAYS = 60  # configurable at top of file
```

Data source: `btc_gbp_hourly.csv`

#### `BTC_ML.PY` — ML Classifier

Predicts whether BTC will be ≥1% higher in 8 hours. Compares Logistic Regression, Random Forest, Gradient Boosting, and XGBoost via TimeSeriesSplit cross-validation with SMOTE resampling. Selects the best model by F1 score and runs a SHAP feature importance analysis.

Data source: `btc_gbp_hourly.csv`

---

## Setup

**Install dependencies:**
```bash
# Core tools
pip install coinbase-advanced-py pandas requests reportlab

# Backtests
pip install eikon matplotlib numpy

# ML backtest only
pip install scikit-learn xgboost imbalanced-learn shap seaborn
```

**Coinbase API key:**

Download your CDP API key from [Coinbase Developer Platform](https://developer.coinbase.com) and save it as `cdp_api_key.json` in the root of this repo. This file is excluded from git.

**Eikon API key:**

Set your Refinitiv Eikon key as an environment variable before running any Eikon-based backtest:
```bash
export EIKON_APP_KEY="your_key_here"
```

**Historical data (for CSV-based scripts):**

```bash
python3 cb_historical.py --pair BTC-GBP --granularity 3600
# saves btc_gbp_1h_....csv — rename to backtests/btc_gbp_hourly.csv
```
