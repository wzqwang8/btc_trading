import pandas as pd
import eikon as ek
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Set up Eikon API
ek.set_app_key('1e01b72982374e88971ea95fe42801910d7207ef')

# Define date range
start = '2024-01-01'
end = '2025-05-21'

# Input
btc = ek.get_timeseries('BTC=', start_date=start, end_date=end)
btc['50'] = btc['CLOSE'].rolling(window=50).mean()
btc['200'] = btc['CLOSE'].rolling(window=200).mean()
btc.dropna(inplace=True)
btc['Prev_Close'] = btc['CLOSE'].shift(1)
btc.iloc[0, btc.columns.get_loc('Prev_Close')] = btc.iloc[1, btc.columns.get_loc('Prev_Close')]
btc['std'] = btc['CLOSE'].rolling(window=14).std()

# Calculate RSI (14-day by default)
delta = btc['CLOSE'].diff()  # Difference between consecutive prices
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Average gains
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Average losses
rs = gain / loss  # Relative strength
btc['RSI'] = 100 - (100 / (1 + rs))  # RSI Calculation

# Calculate True Range (TR)
btc['TR'] = btc[['HIGH', 'LOW', 'Prev_Close']].apply(
    lambda row: max(
        row['HIGH'] - row['LOW'], 
        abs(row['HIGH'] - row['Prev_Close']), 
        abs(row['LOW'] - row['Prev_Close'])
    ),
    axis=1
)
btc['ATR'] = btc['TR'].rolling(window=14).mean()
btc.dropna(inplace=True)

print(btc)

# Initialize variables for backtesting
btc['Position'] = 0  # 1 for Buy, -1 for Sell, 0 for No Position
cash = 1000000  # Total cash (not fully used for each trade)
allocated_cash_per_trade = 100  # Fixed amount per trade
active_trades = []  # List to store active trades
trade_log = []

# Iterate through the btc to simulate trades
for i in range(len(btc)):
    # Check for a buy signal (Golden Cross) or another buy condition
    if (btc['50'].iloc[i-1] < btc['200'].iloc[i-1] and btc['50'].iloc[i] > btc['200'].iloc[i] or
        btc['CLOSE'].iloc[i] < btc['CLOSE'].iloc[i-7]*0.95 and btc['RSI'].iloc[i] < 40) and cash >= allocated_cash_per_trade:
        
        # Calculate buy price, stop loss, and profit target for new trade
        buy_price = btc['CLOSE'].iloc[i] * 1.005  # Account for 0.5% fee
        stop_loss = buy_price - (2 * btc['ATR'].iloc[i])  # ATR-based stop loss
        profit_target = buy_price * 1.1  # ATR-based profit target
        
        # Update cash and add the new trade
        cash -= allocated_cash_per_trade  # Subtract cash for this trade
        active_trades.append({
            'buy_price': buy_price,
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'buy_index': i,
            'action': 'Buy'
        })
        
        btc.at[btc.index[i], 'Position'] = 1  # Mark position as active
        trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Buy', 'Price': round(float(buy_price))})
    
    # Iterate over all active trades to check for profit-taking or stop loss conditions
    for trade in active_trades[:]:
        # Check if 2 weeks have passed since the buy (14 days)
        if i - trade['buy_index'] >= 14:
            sell_price = btc['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
            loss = (sell_price - trade['buy_price']) * (allocated_cash_per_trade / trade['buy_price'])
            cash += loss  # Update cash with loss
            active_trades.remove(trade)  # Remove trade from active trades
            btc.at[btc.index[i], 'Position'] = -1  # Mark position as closed
            trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Forced Close (2 Weeks)', 'Price': round(float(sell_price)), 'Loss': round(float(loss))})
        
        # Check for profit target
        elif btc['CLOSE'].iloc[i] >= trade['profit_target']:  # If profit target is hit
            sell_price = btc['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
            profit = (sell_price - trade['buy_price']) * (allocated_cash_per_trade / trade['buy_price'])
            cash += profit  # Update cash with profit
            active_trades.remove(trade)  # Remove trade from active trades
            btc.at[btc.index[i], 'Position'] = -1  # Mark position as closed
            trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Profit Target', 'Price': round(float(sell_price)), 'Profit': round(float(profit))})
        
        # Check for stop loss
        elif btc['CLOSE'].iloc[i] <= trade['stop_loss']:  # If stop loss is hit
            sell_price = btc['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
            loss = (sell_price - trade['buy_price']) * (allocated_cash_per_trade / trade['buy_price'])
            cash += loss  # Update cash with loss
            active_trades.remove(trade)  # Remove trade from active trades
            btc.at[btc.index[i], 'Position'] = -1  # Mark position as closed
            trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Stop Loss', 'Price': round(float(sell_price)), 'Loss': round(float(loss))})

# Summarize results
total_profit = sum(trade['Profit'] for trade in trade_log if 'Profit' in trade)  # Sum of all profits
total_loss = sum(trade['Loss'] for trade in trade_log if 'Loss' in trade)  # Sum of all losses
net_profit = total_profit + total_loss  # Combine profits and losses
num_trades = len([trade for trade in trade_log if trade['Action'] == 'Profit Target' or trade['Action'] == 'Stop Loss' or trade['Action'] == 'Forced Close (2 Weeks)'])

print(f"Total Profit: ${total_profit:.2f}")
print(f"Total Loss: ${total_loss:.2f}")
print(f"Net Profit: ${net_profit:.2f}")
print(f"Number of Trades: {num_trades}")
print("Trade Log:")
for trade in trade_log:
    print(trade)

# Initialize variable to keep track of trade numbers
trade_number = 1

# Visualize the trades on the chart
plt.figure(figsize=[14, 7])
plt.plot(btc['CLOSE'].index, btc['CLOSE'], label='BTC', color='blue')
plt.plot(btc['50'].index, btc['50'], label='50-Day Moving Average', color='orange')
plt.plot(btc['200'].index, btc['200'], label='200-Day Moving Average', color='green')

# Highlight buy and sell points
buy_signals = btc[btc['Position'] == 1]
sell_signals = btc[btc['Position'] == -1]

# Create lists to store labels for buy and sell points
buy_labels = []
sell_labels = []
rsi_values = []  # To store RSI values for the legend

# Iterate over buy signals and match them with the corresponding sell signal
for buy_index, buy_row in buy_signals.iterrows():
    sell_index = None
    for sell_index, sell_row in sell_signals.iterrows():
        if sell_index > buy_index:  # Match sell after buy
            buy_labels.append(f'Buy {trade_number}')
            sell_labels.append(f'Sell {trade_number}')
            rsi_values.append(buy_row["RSI"])  # Store RSI for the legend
            trade_number += 1
            break

# Plot Buy and Sell Points
plt.scatter(buy_signals.index, buy_signals['CLOSE'], label='Buy Signal', marker='^', color='green', s=100)
plt.scatter(sell_signals.index, sell_signals['CLOSE'], label='Sell/Stop Loss Signal', marker='v', color='red', s=100)

# Add legend for RSI values
for i, rsi_value in enumerate(rsi_values):
    plt.plot([], [], 'o', label=f'RSI Buy {i+1}: {round(rsi_value, 2)}', color='green')

# Final chart formatting
plt.title('BTC with Backtest - 50-Day and 200-Day Moving Averages (With Stop Loss)')
plt.xlabel('Date')
plt.ylabel('Price $')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()
