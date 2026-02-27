"""
BTC Backtest — Golden Cross + RSI Dip Strategy (Long Only)
===========================================================
Entry signals:
  - Golden Cross: 50-day MA crosses above 200-day MA
  - RSI Dip:      price drops >5% in 7 days AND RSI < 40

Exit signals (per trade):
  - Profit target: +10%
  - Stop loss:     price - 2x ATR
  - Time stop:     14 bars (forced close)

Data source: Refinitiv Eikon (BTC=)
"""

import os
import numpy as np
import pandas as pd
import eikon as ek
import matplotlib.pyplot as plt
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
EIKON_APP_KEY       = os.environ.get("EIKON_APP_KEY", "YOUR_KEY_HERE")
START               = "2024-01-01"
END                 = datetime.today().strftime("%Y-%m-%d")
STARTING_CAPITAL    = 100_000
TRADE_SIZE          = 100       # GBP per trade
FEE                 = 0.005     # 0.5% per side
PROFIT_TARGET_PCT   = 0.10      # 10% profit target
ATR_STOP_MULT       = 2.0       # stop = entry - N * ATR
TIME_STOP_BARS      = 14        # force close after N bars

# ── Data ──────────────────────────────────────────────────────────────────────
ek.set_app_key(EIKON_APP_KEY)
btc = ek.get_timeseries("BTC=", start_date=START, end_date=END)

# Indicators
btc["MA50"]  = btc["CLOSE"].rolling(50).mean()
btc["MA200"] = btc["CLOSE"].rolling(200).mean()
btc["std"]   = btc["CLOSE"].rolling(14).std()

delta      = btc["CLOSE"].diff()
gain       = delta.where(delta > 0, 0).rolling(14).mean()
loss_r     = (-delta.where(delta < 0, 0)).rolling(14).mean()
btc["RSI"] = 100 - (100 / (1 + gain / loss_r))

prev_close = btc["CLOSE"].shift(1).bfill()
btc["TR"]  = pd.concat([
    btc["HIGH"] - btc["LOW"],
    (btc["HIGH"] - prev_close).abs(),
    (btc["LOW"]  - prev_close).abs(),
], axis=1).max(axis=1)
btc["ATR"] = btc["TR"].rolling(14).mean()

btc.dropna(inplace=True)
print(btc.tail())

# ── Backtest ──────────────────────────────────────────────────────────────────
cash          = STARTING_CAPITAL
active_trades = []
trade_log     = []
equity_curve  = []

for i in range(len(btc)):
    price   = btc["CLOSE"].iloc[i]
    ma50    = btc["MA50"].iloc[i]
    ma200   = btc["MA200"].iloc[i]
    rsi     = btc["RSI"].iloc[i]
    atr     = btc["ATR"].iloc[i]
    p7      = btc["CLOSE"].iloc[i - 7] if i >= 7 else price

    # ── Entry signal ──────────────────────────────────────────────────────────
    golden_cross = (
        btc["MA50"].iloc[i - 1] < btc["MA200"].iloc[i - 1]
        and ma50 > ma200
    )
    rsi_dip = (price < p7 * 0.95) and (rsi < 40)

    if (golden_cross or rsi_dip) and cash >= TRADE_SIZE:
        entry  = price * (1 + FEE)
        stop   = entry - ATR_STOP_MULT * atr
        target = entry * (1 + PROFIT_TARGET_PCT)
        cash  -= TRADE_SIZE
        active_trades.append({
            "entry":  entry,
            "stop":   stop,
            "target": target,
            "bar":    i,
        })
        trade_log.append({"Date": btc.index[i], "Action": "Buy", "Price": entry})

    # ── Exit checks ───────────────────────────────────────────────────────────
    for trade in active_trades[:]:
        exit_price = None
        reason     = None

        if i - trade["bar"] >= TIME_STOP_BARS:
            exit_price = price * (1 - FEE)
            reason     = "Time Stop"
        elif price >= trade["target"]:
            exit_price = price * (1 - FEE)
            reason     = "Profit Target"
        elif price <= trade["stop"]:
            exit_price = price * (1 - FEE)
            reason     = "Stop Loss"

        if exit_price is not None:
            units = TRADE_SIZE / trade["entry"]
            pnl   = (exit_price - trade["entry"]) * units
            cash += TRADE_SIZE + pnl
            active_trades.remove(trade)
            trade_log.append({
                "Date":   btc.index[i],
                "Action": reason,
                "Price":  exit_price,
                "PnL":    pnl,
            })

    # Track equity (cash + mark-to-market of open trades)
    unrealised = sum(
        (price - t["entry"]) * (TRADE_SIZE / t["entry"])
        for t in active_trades
    )
    equity_curve.append(cash + unrealised)

# ── Performance stats ─────────────────────────────────────────────────────────
closed = [t for t in trade_log if "PnL" in t]
pnls   = [t["PnL"] for t in closed]

total_pnl    = sum(pnls)
wins         = [p for p in pnls if p > 0]
losses       = [p for p in pnls if p <= 0]
win_rate     = len(wins) / len(pnls) * 100 if pnls else 0
avg_win      = np.mean(wins)  if wins   else 0
avg_loss     = np.mean(losses) if losses else 0

equity = pd.Series(equity_curve, index=btc.index)
peak   = equity.cummax()
dd     = (equity - peak) / peak * 100
max_dd = dd.min()

returns = equity.pct_change().dropna()
sharpe  = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
total_return = (equity.iloc[-1] / STARTING_CAPITAL - 1) * 100

print("\n" + "=" * 50)
print("BACKTEST RESULTS")
print("=" * 50)
print(f"Period:          {START} → {END}")
print(f"Starting capital: £{STARTING_CAPITAL:,.0f}")
print(f"Final equity:     £{equity.iloc[-1]:,.0f}")
print(f"Total return:     {total_return:+.1f}%")
print(f"Total P&L:        £{total_pnl:+,.2f}")
print(f"Closed trades:    {len(closed)}")
print(f"Win rate:         {win_rate:.1f}%")
print(f"Avg win:          £{avg_win:+,.2f}")
print(f"Avg loss:         £{avg_loss:+,.2f}")
print(f"Max drawdown:     {max_dd:.1f}%")
print(f"Sharpe ratio:     {sharpe:.2f}")
print("=" * 50)

# ── Chart ─────────────────────────────────────────────────────────────────────
buys  = [t for t in trade_log if t["Action"] == "Buy"]
exits = [t for t in trade_log if "PnL" in t]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False,
                                gridspec_kw={"height_ratios": [2, 1]})

# Price + signals
ax1.plot(btc.index, btc["CLOSE"],  label="BTC", color="black", linewidth=1)
ax1.plot(btc.index, btc["MA50"],   label="MA50",  color="blue",   linewidth=1)
ax1.plot(btc.index, btc["MA200"],  label="MA200", color="orange", linewidth=1)
ax1.scatter([t["Date"] for t in buys],
            [t["Price"] for t in buys],
            marker="^", color="green", s=100, label="Buy", zorder=5)
ax1.scatter([t["Date"] for t in exits],
            [t["Price"] for t in exits],
            marker="v", color="red", s=80, label="Exit", zorder=5)
ax1.set_title(f"BTC Golden Cross / RSI Dip Backtest  |  Win rate {win_rate:.0f}%  |  Sharpe {sharpe:.2f}")
ax1.set_ylabel("Price (£)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Equity curve
ax2.plot(btc.index, equity_curve, color="steelblue", linewidth=1.5, label="Equity")
ax2.fill_between(btc.index, STARTING_CAPITAL, equity_curve,
                 where=[e >= STARTING_CAPITAL for e in equity_curve],
                 color="green", alpha=0.15)
ax2.fill_between(btc.index, STARTING_CAPITAL, equity_curve,
                 where=[e < STARTING_CAPITAL for e in equity_curve],
                 color="red", alpha=0.15)
ax2.axhline(STARTING_CAPITAL, color="gray", linestyle="--", linewidth=0.8)
ax2.set_title(f"Equity Curve  |  Max drawdown {max_dd:.1f}%  |  Return {total_return:+.1f}%")
ax2.set_ylabel("Equity (£)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Trade table
if closed:
    rows = [[t["Date"].strftime("%d-%m-%y"), t["Action"], f"£{t['Price']:,.0f}", f"£{t['PnL']:+,.2f}"]
            for t in closed]
    fig2, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.35 + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=rows,
                   colLabels=["Date", "Exit Reason", "Price", "P&L"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width([0, 1, 2, 3])
    fig2.suptitle(f"Trade Log  —  Net P&L £{total_pnl:+,.2f}", fontsize=11)
    plt.tight_layout()
    plt.show()
