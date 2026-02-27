import pandas as pd
import eikon as ek
import numpy as np
import matplotlib.pyplot as plt
import time
import dateutil.relativedelta as rd
pd.set_option('display.max_columns', None)
import requests
from datetime import datetime, timedelta

file_path = f'M:/24.Naphtha/Python scripts/bot_testing/btc_gbp_hourly.csv'
btc = pd.read_csv(file_path)
btc['time'] = pd.to_datetime(btc['time'])  
btc.set_index('time', inplace=True)

btc['50'] = btc['close'].rolling(window=50).mean()
btc['200'] = btc['close'].rolling(window=200).mean()
btc.dropna(inplace=True)
btc['h-1'] = btc['close'].shift(1)
btc.iloc[0, btc.columns.get_loc('h-1')] = btc.iloc[1, btc.columns.get_loc('h-1')] # first row = second row
btc['std'] = btc['close'].rolling(window=14).std()

# Bollinger Bands
window = 20
std_dev = 2
btc['BB_Mid'] = btc['close'].rolling(window).mean()
btc['BB_Std'] = btc['close'].rolling(window).std()
btc['BB_High'] = btc['BB_Mid'] + std_dev * btc['BB_Std']
btc['BB_Low'] = btc['BB_Mid'] - std_dev * btc['BB_Std']

# Calculate RSI (14-day by default)
delta = btc['close'].diff()  # Difference between consecutive prices
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Average gains
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Average losses
rs = gain / loss  # Relative strength
btc['RSI'] = 100 - (100 / (1 + rs))  # RSI Calculation

# Calculate True Range (TR)
btc['TR'] = btc[['high', 'low', 'h-1']].apply(
    lambda row: max(
        row['high'] - row['low'], 
        abs(row['high'] - row['h-1']), 
        abs(row['low'] - row['h-1'])
    ),
    axis=1
)
btc['ATR'] = btc['TR'].rolling(window=14).mean()
btc.dropna(inplace=True)
print(btc)

##########################

mva_pairs = [
    ('MA_3', 'MA_5'),
    ('MA_3', 'MA_10'),
    ('MA_5', 'MA_8'),
    ('MA_5', 'MA_13'),
    ('MA_5', 'MA_20'),
    ('MA_7', 'MA_21'),
    ('MA_7', 'MA_48'),
    ('MA_9', 'MA_21'),
    ('MA_10', 'MA_20'),
    ('MA_10', 'MA_50'),
    ('MA_12', 'MA_26'),
    ('MA_13', 'MA_34'),
    ('MA_14', 'MA_28'),
    ('MA_15', 'MA_30'),
    ('MA_18', 'MA_50'),
    ('MA_20', 'MA_40'),
    ('MA_21', 'MA_48'),
    ('MA_21', 'MA_100'),
    ('MA_25', 'MA_75'),
    ('MA_30', 'MA_60'),
    ('MA_30', 'MA_90'),
    ('MA_35', 'MA_100'),
    ('MA_40', 'MA_80'),
    ('MA_40', 'MA_200'),
    ('MA_45', 'MA_90'),
    ('MA_48', 'MA_96'),
    ('MA_50', 'MA_100'),
    ('MA_50', 'MA_150'),
    ('MA_50', 'MA_200'),
    ('MA_55', 'MA_110'),
    ('MA_60', 'MA_120'),
    ('MA_70', 'MA_140'),
    ('MA_80', 'MA_160'),
    ('MA_90', 'MA_180'),
    ('MA_100', 'MA_200'),
    ('MA_120', 'MA_240'),
    ('MA_150', 'MA_300'),
    ('MA_200', 'MA_400'),
    ('MA_250', 'MA_500'),
    ('MA_300', 'MA_600')
]

ma_windows = sorted({int(ma.replace("MA_", "")) for pair in mva_pairs for ma in pair})

# Step 3: Calculate moving averages
for window in ma_windows:
    btc[f'MA_{window}'] = btc['close'].rolling(window=window).mean()


btc['score'] = 0
btc['valid_counts'] = 0  # To count how many valid (non-NaN) MA pairs were evaluated

for short, long in mva_pairs:
    short_prev = btc[short].shift(1)
    long_prev = btc[long].shift(1)
    short_curr = btc[short]
    long_curr = btc[long]

    crossover_up = (short_prev < long_prev) & (short_curr > long_curr)
    crossover_down = (short_prev > long_prev) & (short_curr < long_curr)
    valid_mask = (~short_prev.isna()) & (~long_prev.isna()) & (~short_curr.isna()) & (~long_curr.isna())

    btc['score'] += np.where(crossover_up, 1, np.where(crossover_down, -1, 0))
    btc['valid_counts'] += valid_mask.astype(int)

# Normalize by the number of valid MA pairs for that row
btc['normalized_score'] = np.where(
    btc['valid_counts'] > 0,
    btc['score'] / btc['valid_counts'],
    0
)

print(btc)

cutoff = datetime.today() - rd.relativedelta(years=1)

# Subset to the last 4 years
btc_4y = btc[btc.index >= cutoff]

# Define RSI thresholds
rsi_oversold = 30
rsi_overbought = 70

# Step 1: RSI Calculation (if not done already)
delta = btc_4y['close'].diff()  # Difference between consecutive prices
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Average gains
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Average losses
rs = gain / loss  # Relative strength
btc_4y['RSI'] = 100 - (100 / (1 + rs))  # RSI Calculation

# Step 2: Identify signals using the RSI filter
buy_rsi_signal = btc_4y['RSI'] < rsi_oversold  # Buy when RSI is below 30 (oversold)
sell_rsi_signal = btc_4y['RSI'] > rsi_overbought  # Sell when RSI is above 70 (overbought)

# Add these RSI signals to the dataframe
btc_4y['rsi_buy_signal'] = buy_rsi_signal
btc_4y['rsi_sell_signal'] = sell_rsi_signal

# Modify the existing buy/sell conditions to incorporate RSI
buy_signals = btc_4y[(btc_4y['normalized_score'] >= 0.05) & (btc_4y['rsi_buy_signal'])]
sell_signals = btc_4y[(btc_4y['normalized_score'] <= -0.05) & (btc_4y['rsi_sell_signal'])]

# Create 2 stacked subplots sharing x‑axis
fig, (ax_price, ax_score) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# — Top subplot: Price + signals —
ax_price.plot(
    btc_4y.index, btc_4y['close'],
    label='BTC Price', color='navy', alpha=0.8
)
ax_price.scatter(
    buy_signals.index,  buy_signals['close'],
    marker='^', color='green', s=100, label='Buy', alpha=0.9
)
ax_price.scatter(
    sell_signals.index, sell_signals['close'],
    marker='v', color='red', s=100, label='Sell', alpha=0.9
)
ax_price.set_title(' Buy/Sell Signals (Last 4 Years)')
ax_price.set_ylabel('Price (USD)')
ax_price.legend(loc='upper left')
ax_price.grid(True)

# — Bottom subplot: raw score —
ax_score.plot(
    btc_4y.index, btc_4y['normalized_score'],
    label='Score', color='darkorange', alpha=0.8
)
ax_score.axhline( 0.1, color='green', linestyle='--', linewidth=1)
ax_score.axhline(-0.1, color='red', linestyle='--', linewidth=1)
ax_score.set_title('Crossover Score (Last 4 Years)')
ax_score.set_ylabel('Score')
ax_score.set_xlabel('Date')
ax_score.legend(loc='upper left')
ax_score.grid(True)

plt.tight_layout()
plt.show()

# # Initialize variables for backtesting
# btc['Position'] = 0  # 1 for Buy, -1 for Sell, 0 for No Position
# cash = 1000000  # Total cash (not fully used for each trade)
# allocated_cash_per_trade = 100  # Fixed amount per trade
# active_trades = []  # List to store active trades
# trade_log = []

# # Iterate through the btc to simulate trades
# for i in range(len(btc)):
#     # Check for a buy signal (Golden Cross) or another buy condition
#     if (btc['50'].iloc[i-1] < btc['200'].iloc[i-1] and btc['50'].iloc[i] > btc['200'].iloc[i] or
#         btc['close'].iloc[i] < btc['close'].iloc[i-7]*0.95 and btc['RSI'].iloc[i] < 40) and cash >= allocated_cash_per_trade:
        
#         # Calculate buy price, stop loss, and profit target for new trade
#         buy_price = btc['close'].iloc[i] * 1.005  # Account for 0.5% fee
#         stop_loss = buy_price - (2 * btc['ATR'].iloc[i])  # ATR-based stop loss
#         profit_target = buy_price * 1.1  # ATR-based profit target
        
#         # Update cash and add the new trade
#         cash -= allocated_cash_per_trade  # Subtract cash for this trade
#         active_trades.append({
#             'buy_price': buy_price,
#             'stop_loss': stop_loss,
#             'profit_target': profit_target,
#             'buy_index': i,
#             'action': 'Buy'
#         })
        
#         btc.at[btc.index[i], 'Position'] = 1  # Mark position as active
#         trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Buy', 'Price': round(float(buy_price))})
    
#     # Iterate over all active trades to check for profit-taking or stop loss conditions
#     for trade in active_trades[:]:
#         # Check if 2 weeks have passed since the buy (14 days)
#         if i - trade['buy_index'] >= 14:
#             sell_price = btc['close'].iloc[i] * 0.995  # Account for 0.5% fee
#             loss = (sell_price - trade['buy_price']) * (allocated_cash_per_trade / trade['buy_price'])
#             cash += loss  # Update cash with loss
#             active_trades.remove(trade)  # Remove trade from active trades
#             btc.at[btc.index[i], 'Position'] = -1  # Mark position as closed
#             trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Forced close (2 Weeks)', 'Price': round(float(sell_price)), 'Loss': round(float(loss))})
        
#         # Check for profit target
#         elif btc['close'].iloc[i] >= trade['profit_target']:  # If profit target is hit
#             sell_price = btc['close'].iloc[i] * 0.995  # Account for 0.5% fee
#             profit = (sell_price - trade['buy_price']) * (allocated_cash_per_trade / trade['buy_price'])
#             cash += profit  # Update cash with profit
#             active_trades.remove(trade)  # Remove trade from active trades
#             btc.at[btc.index[i], 'Position'] = -1  # Mark position as closed
#             trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Profit Target', 'Price': round(float(sell_price)), 'Profit': round(float(profit))})
        
#         # Check for stop loss
#         elif btc['close'].iloc[i] <= trade['stop_loss']:  # If stop loss is hit
#             sell_price = btc['close'].iloc[i] * 0.995  # Account for 0.5% fee
#             loss = (sell_price - trade['buy_price']) * (allocated_cash_per_trade / trade['buy_price'])
#             cash += loss  # Update cash with loss
#             active_trades.remove(trade)  # Remove trade from active trades
#             btc.at[btc.index[i], 'Position'] = -1  # Mark position as closed
#             trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Stop Loss', 'Price': round(float(sell_price)), 'Loss': round(float(loss))})

# # Summarize results
# total_profit = sum(trade['Profit'] for trade in trade_log if 'Profit' in trade)  # Sum of all profits
# total_loss = sum(trade['Loss'] for trade in trade_log if 'Loss' in trade)  # Sum of all losses
# net_profit = total_profit + total_loss  # Combine profits and losses
# num_trades = len([trade for trade in trade_log if trade['Action'] == 'Profit Target' or trade['Action'] == 'Stop Loss' or trade['Action'] == 'Forced close (2 Weeks)'])

# print(f"Total Profit: ${total_profit:.2f}")
# print(f"Total Loss: ${total_loss:.2f}")
# print(f"Net Profit: ${net_profit:.2f}")
# print(f"Number of Trades: {num_trades}")
# print("Trade Log:")
# for trade in trade_log:
#     print(trade)

# # Initialize variable to keep track of trade numbers
# trade_number = 1

# # Visualize the trades on the chart
# plt.figure(figsize=[14, 7])
# plt.plot(btc['close'].index, btc['close'], label='BTC', color='blue')
# plt.plot(btc['50'].index, btc['50'], label='50-Day Moving Average', color='orange')
# plt.plot(btc['200'].index, btc['200'], label='200-Day Moving Average', color='green')

# # Highlight buy and sell points
# buy_signals = btc[btc['Position'] == 1]
# sell_signals = btc[btc['Position'] == -1]

# # Create lists to store labels for buy and sell points
# buy_labels = []
# sell_labels = []
# rsi_values = []  # To store RSI values for the legend

# # Iterate over buy signals and match them with the corresponding sell signal
# for buy_index, buy_row in buy_signals.iterrows():
#     sell_index = None
#     for sell_index, sell_row in sell_signals.iterrows():
#         if sell_index > buy_index:  # Match sell after buy
#             buy_labels.append(f'Buy {trade_number}')
#             sell_labels.append(f'Sell {trade_number}')
#             rsi_values.append(buy_row["RSI"])  # Store RSI for the legend
#             trade_number += 1
#             break

# # Plot Buy and Sell Points
# plt.scatter(buy_signals.index, buy_signals['close'], label='Buy Signal', marker='^', color='green', s=100)
# plt.scatter(sell_signals.index, sell_signals['close'], label='Sell/Stop Loss Signal', marker='v', color='red', s=100)

# # Add legend for RSI values
# for i, rsi_value in enumerate(rsi_values):
#     plt.plot([], [], 'o', label=f'RSI Buy {i+1}: {round(rsi_value, 2)}', color='green')

# # Final chart formatting
# plt.title('BTC with Backtest - 50-Day and 200-Day Moving Averages (With Stop Loss)')
# plt.xlabel('Date')
# plt.ylabel('Price $')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.grid(True)
# plt.show()
