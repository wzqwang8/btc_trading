import eikon as ek
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Eikon API key
ek.set_app_key('1e01b72982374e88971ea95fe42801910d7207ef')

# Get BTC data
start = '2024-06-01'
end = datetime.today().strftime('%Y-%m-%d')
btc = ek.get_timeseries('BTC=', start_date=start, end_date=end)
btc.dropna(inplace=True)

# --- Indicators ---

# Moving Averages
btc['MA_50'] = btc['CLOSE'].rolling(window=50).mean()
btc['MA_200'] = btc['CLOSE'].rolling(window=200).mean()

# Bollinger Bands
window = 20
std_dev = 2
btc['BB_Mid'] = btc['CLOSE'].rolling(window).mean()
btc['BB_Std'] = btc['CLOSE'].rolling(window).std()
btc['BB_High'] = btc['BB_Mid'] + std_dev * btc['BB_Std']
btc['BB_Low'] = btc['BB_Mid'] - std_dev * btc['BB_Std']

# RSI
delta = btc['CLOSE'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
btc['RSI'] = 100 - (100 / (1 + rs))

# MACD and Signal Line
ema12 = btc['CLOSE'].ewm(span=12, adjust=False).mean()
ema26 = btc['CLOSE'].ewm(span=26, adjust=False).mean()
btc['MACD'] = ema12 - ema26
btc['MACD_Signal'] = btc['MACD'].ewm(span=9, adjust=False).mean()
btc['MACD_Hist'] = btc['MACD'] - btc['MACD_Signal']

# --- Signals ---
btc['Buy'] = (btc['CLOSE'] < btc['BB_Low']) | ((btc['RSI'] < 30) & (btc['MACD'] > btc['MACD_Signal']))
btc['Sell'] = (btc['RSI'] > 70) & (btc['MACD'] < btc['MACD_Signal']) 

# --- Plotting ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

# Price and BB
ax1.plot(btc.index, btc['CLOSE'], label='BTC Price', color='black')
ax1.plot(btc.index, btc['BB_Mid'], label='BB Mid', linestyle='--')
ax1.plot(btc.index, btc['BB_High'], label='BB High', linestyle='--', color='green')
ax1.plot(btc.index, btc['BB_Low'], label='BB Low', linestyle='--', color='red')
ax1.plot(btc.index, btc['MA_50'], label='50-day MA', color='blue', linewidth=1)
ax1.plot(btc.index, btc['MA_200'], label='200-day MA', color='orange', linewidth=1)
ax1.scatter(btc.index[btc['Buy']], btc['CLOSE'][btc['Buy']], marker='^', color='green', label='Buy Signal', zorder=5)
ax1.scatter(btc.index[btc['Sell']], btc['CLOSE'][btc['Sell']], marker='v', color='red', label='Sell Signal', zorder=5)
ax1.set_title('BTC Price with Bollinger Bands & Buy/Sell Signals')
ax1.legend()

# MACD Line and Signal Line
ax2.plot(btc.index, btc['MACD'], label='MACD', color='blue')
ax2.plot(btc.index, btc['MACD_Signal'], label='Signal Line', color='orange')
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_title('MACD & Signal Line')
ax2.legend()

# MACD Histogram
ax3.bar(btc.index, btc['MACD_Hist'], label='MACD Histogram', color='gray')
ax3.axhline(0, color='black', linestyle='--')
ax3.set_title('MACD Histogram')
ax3.legend()

# RSI
ax4.plot(btc.index, btc['RSI'], label='RSI', color='purple')
ax4.axhline(70, color='red', linestyle='--')
ax4.axhline(30, color='green', linestyle='--')
ax4.set_title('RSI')
ax4.set_ylim(0, 100)
ax4.legend()

plt.tight_layout()
plt.show()

