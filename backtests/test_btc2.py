"""
BTC Backtest — Long/Short Volatility Strategy
==============================================
Entry signals:
  - Long:  Golden Cross (7/20 MA) OR price dip >5% in 5 days AND RSI < 40
  - Short: Death Cross (7/20 MA)  OR price surge >5% in 5 days AND RSI > 70

Exit signals (per trade):
  - Profit target: entry ± 1.5x std dev
  - Stop loss:     entry ∓ 0.8x std dev
  - Time stop:     14 bars

Exposure limits prevent over-concentration.

Data source: Refinitiv Eikon (BTC=)
"""

import os
import numpy as np
import pandas as pd
import eikon as ek
import matplotlib.pyplot as plt
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
EIKON_APP_KEY    = os.environ.get("EIKON_APP_KEY", "YOUR_KEY_HERE")
START            = "2024-06-01"
END              = datetime.today().strftime("%Y-%m-%d")
STARTING_CAPITAL = 100_000
TRADE_SIZE       = 0.001       # BTC per trade
FEE              = 0.005       # 0.5% per side
EXPO_LIMIT       = 0.2         # max BTC exposure either direction
STD_STOP_MULT    = 0.8
STD_TARGET_MULT  = 1.5
TIME_STOP_BARS   = 14

# ── Data ──────────────────────────────────────────────────────────────────────
ek.set_app_key(EIKON_APP_KEY)
btc = ek.get_timeseries("BTC=", start_date=START, end_date=END)

btc["MA7"]  = btc["CLOSE"].rolling(7).mean()
btc["MA20"] = btc["CLOSE"].rolling(20).mean()
btc["std"]  = btc["CLOSE"].rolling(14).std()

delta      = btc["CLOSE"].diff()
gain       = delta.where(delta > 0, 0).rolling(14).mean()
loss_r     = (-delta.where(delta < 0, 0)).rolling(14).mean()
btc["RSI"] = 100 - (100 / (1 + gain / loss_r))

btc.dropna(inplace=True)
print(btc.tail())

# ── Backtest ──────────────────────────────────────────────────────────────────
expo          = 0.0
trade_id      = 0
active_trades = []
trade_log     = []
equity_curve  = []
cash          = STARTING_CAPITAL

for i in range(len(btc)):
    price = btc["CLOSE"].iloc[i]
    ma7   = btc["MA7"].iloc[i]
    ma20  = btc["MA20"].iloc[i]
    rsi   = btc["RSI"].iloc[i]
    std   = btc["std"].iloc[i]
    p5    = btc["CLOSE"].iloc[i - 5] if i >= 5 else price

    within_limits = -EXPO_LIMIT <= expo <= EXPO_LIMIT

    # ── Long entry ────────────────────────────────────────────────────────────
    golden_cross = (btc["MA7"].iloc[i - 1] < btc["MA20"].iloc[i - 1]) and (ma7 > ma20)
    rsi_dip      = (price < p5 * 0.95) and (rsi < 40)

    if (golden_cross or rsi_dip) and within_limits:
        trade_id += 1
        entry   = price * (1 + FEE)
        stop    = entry - STD_STOP_MULT * std
        target  = entry + STD_TARGET_MULT * std
        expo   += TRADE_SIZE
        cash   -= TRADE_SIZE * entry
        active_trades.append({
            "id": trade_id, "side": "Long",
            "entry": entry, "stop": stop, "target": target, "bar": i,
        })
        trade_log.append({"id": trade_id, "Date": btc.index[i], "Action": "Buy",
                          "Price": entry, "Expo": expo})

    # ── Short entry ───────────────────────────────────────────────────────────
    death_cross = (btc["MA7"].iloc[i - 1] > btc["MA20"].iloc[i - 1]) and (ma7 < ma20)
    rsi_surge   = (price > p5 * 1.05) and (rsi > 70)

    if (death_cross or rsi_surge) and within_limits:
        trade_id += 1
        entry   = price * (1 - FEE)
        stop    = entry + STD_STOP_MULT * std
        target  = entry - STD_TARGET_MULT * std
        expo   -= TRADE_SIZE
        cash   += TRADE_SIZE * entry   # received from short sale
        active_trades.append({
            "id": trade_id, "side": "Short",
            "entry": entry, "stop": stop, "target": target, "bar": i,
        })
        trade_log.append({"id": trade_id, "Date": btc.index[i], "Action": "Sell",
                          "Price": entry, "Expo": expo})

    # ── Exit checks ───────────────────────────────────────────────────────────
    for trade in active_trades[:]:
        reason     = None
        exit_price = None

        if trade["side"] == "Long":
            time_out = i - trade["bar"] >= TIME_STOP_BARS
            hit_stop = price <= trade["stop"]
            hit_tgt  = price >= trade["target"]
            exit_mult = 1 - FEE
        else:
            time_out = i - trade["bar"] >= TIME_STOP_BARS
            hit_stop = price >= trade["stop"]
            hit_tgt  = price <= trade["target"]
            exit_mult = 1 + FEE

        if time_out:
            reason = "Time Stop"
        elif hit_stop:
            reason = "Stop Loss"
        elif hit_tgt:
            reason = "Profit Target"

        if reason:
            exit_price = price * exit_mult
            if trade["side"] == "Long":
                pnl   = (exit_price - trade["entry"]) * TRADE_SIZE
                expo -= TRADE_SIZE
                cash += TRADE_SIZE * exit_price
            else:
                pnl   = (trade["entry"] - exit_price) * TRADE_SIZE
                expo += TRADE_SIZE
                cash -= TRADE_SIZE * exit_price   # buy back
            active_trades.remove(trade)
            trade_log.append({
                "id": trade["id"], "Date": btc.index[i],
                "Action": f"{reason} ({'Sell' if trade['side']=='Long' else 'Buy'})",
                "Price": exit_price, "PnL": pnl, "Expo": expo,
            })

    # Mark-to-market equity
    mtm = sum(
        (price - t["entry"]) * TRADE_SIZE if t["side"] == "Long"
        else (t["entry"] - price) * TRADE_SIZE
        for t in active_trades
    )
    equity_curve.append(cash + mtm)

# ── Performance stats ─────────────────────────────────────────────────────────
closed = [t for t in trade_log if "PnL" in t]
pnls   = [t["PnL"] for t in closed]

total_pnl = sum(pnls)
wins      = [p for p in pnls if p > 0]
losses    = [p for p in pnls if p <= 0]
win_rate  = len(wins) / len(pnls) * 100 if pnls else 0
avg_win   = np.mean(wins)   if wins   else 0
avg_loss  = np.mean(losses) if losses else 0

equity = pd.Series(equity_curve, index=btc.index)
peak   = equity.cummax()
dd     = (equity - peak) / peak * 100
max_dd = dd.min()

returns      = equity.pct_change().dropna()
sharpe       = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
total_return = (equity.iloc[-1] / STARTING_CAPITAL - 1) * 100

print("\n" + "=" * 50)
print("BACKTEST RESULTS")
print("=" * 50)
print(f"Period:           {START} → {END}")
print(f"Starting capital: £{STARTING_CAPITAL:,.0f}")
print(f"Final equity:     £{equity.iloc[-1]:,.0f}")
print(f"Total return:     {total_return:+.1f}%")
print(f"Total P&L:        £{total_pnl:+,.2f}")
print(f"Closed trades:    {len(closed)}")
print(f"Win rate:         {win_rate:.1f}%")
print(f"Avg win:          £{avg_win:+,.4f}")
print(f"Avg loss:         £{avg_loss:+,.4f}")
print(f"Max drawdown:     {max_dd:.1f}%")
print(f"Sharpe ratio:     {sharpe:.2f}")
print("=" * 50)

# ── Chart ─────────────────────────────────────────────────────────────────────
longs  = [t for t in trade_log if t["Action"] == "Buy"]
shorts = [t for t in trade_log if t["Action"] == "Sell"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False,
                                gridspec_kw={"height_ratios": [2, 1]})

ax1.plot(btc.index, btc["CLOSE"], label="BTC",  color="black",  linewidth=1)
ax1.plot(btc.index, btc["MA7"],   label="MA7",  color="blue",   linewidth=1)
ax1.plot(btc.index, btc["MA20"],  label="MA20", color="orange", linewidth=1)
ax1.scatter([t["Date"] for t in longs],  [t["Price"] for t in longs],
            marker="^", color="green", s=100, label="Long entry", zorder=5)
ax1.scatter([t["Date"] for t in shorts], [t["Price"] for t in shorts],
            marker="v", color="red",   s=100, label="Short entry", zorder=5)
ax1.set_title(f"BTC Long/Short Backtest  |  Win rate {win_rate:.0f}%  |  Sharpe {sharpe:.2f}")
ax1.set_ylabel("Price (£)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

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
