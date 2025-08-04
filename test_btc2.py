import pandas as pd
import eikon as ek
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Set up Eikon API
ek.set_app_key('1e01b72982374e88971ea95fe42801910d7207ef')

# Define date range
start = '2024-06-01'
end = '2025-05-22'

# Input
btc = ek.get_timeseries('BTC=', start_date=start, end_date=end)
btc['50'] = btc['CLOSE'].rolling(window=7).mean()
btc['200'] = btc['CLOSE'].rolling(window=20).mean()
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

# Plot the graph
plt.figure(figsize=[14, 10])

# Price and Moving Averages
plt.subplot(3, 1, 1)
plt.plot(btc['CLOSE'], label='BTC Price', color='blue')
plt.plot(btc['50'], label='50-Day Moving Average', color='orange')
plt.plot(btc['200'], label='200-Day Moving Average', color='green')
plt.title('BTC Price and Moving Averages')
plt.legend()
plt.grid(True)

# RSI Plot
plt.subplot(3, 1, 2)
plt.plot(btc['RSI'], label='RSI', color='purple')
plt.axhline(30, color='red', linestyle='--', label='Oversold (30)')
plt.axhline(70, color='green', linestyle='--', label='Overbought (70)')
plt.title('RSI Indicator')
plt.legend()
plt.grid(True)

# ATR Plot
plt.subplot(3, 1, 3)
plt.plot(btc['ATR'], label='ATR (Average True Range)', color='brown')
plt.title('ATR Indicator')
plt.legend()
plt.grid(True)

plt.tight_layout()


# Initialize variables for backtesting
btc['Position'] = 0  # 5kt for Buy, -5kt for Sell, 0 for No Position
expo = 0  # kt
expo_upper_limit = 0.2 # btc
expo_lower_limit = -0.2
allocated_kt_per_trade = 0.001  # btc
active_trades = []  # List to store active trades
trade_log = []  # List to store trade logs

# Initialize trade counter
trade_counter = 1

# Iterate through the btc data to simulate trades
for i in range(len(btc)):
    # Check for a buy signal (Golden Cross) or another buy condition
    if (btc['50'].iloc[i-1] < btc['200'].iloc[i-1] and btc['50'].iloc[i] > btc['200'].iloc[i] or
        btc['CLOSE'].iloc[i] < btc['CLOSE'].iloc[i-5]*0.95 and btc['RSI'].iloc[i] < 40) and expo_lower_limit <= expo <= expo_upper_limit:
        
        # Calculate buy price, stop loss, and profit target for new trade
        buy_price = btc['CLOSE'].iloc[i] * 1.005  # Account for 0.5% fee
        stop_loss = buy_price - (0.8 * btc['std'].iloc[i])  # Stop loss based on volatility
        profit_target = buy_price + (1.5 * btc['std'].iloc[i])  # Profit target
        
        # Update exposure and add the new trade with trade_id
        expo += allocated_kt_per_trade  # buy so +ve
        active_trades.append({
            'trade_id': trade_counter,  # Assign a unique ID to each trade
            'buy_price': round(buy_price, 3),
            'stop_loss': round(stop_loss, 3),
            'profit_target': round(profit_target, 3),
            'buy_index': i,
            'action': 'Buy',
            'exposure': expo
        })
        trade_counter += 1  # Increment the trade_counter for the next buy
        
        btc.at[btc.index[i], 'Position'] = 5  # Mark position as active
        trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Buy', 'Price': round(float(buy_price), 2), 'Exposure': expo, 'Trade ID': trade_counter - 1})
    
    # Check for a sell signal (Golden Cross) or another sell condition
    elif (btc['50'].iloc[i-1] > btc['200'].iloc[i-1] and btc['50'].iloc[i] < btc['200'].iloc[i] or
        btc['CLOSE'].iloc[i] > btc['CLOSE'].iloc[i-5]*1.05 and btc['RSI'].iloc[i] > 70) and expo_lower_limit <= expo <= expo_upper_limit:
        
        # Calculate sell price, stop loss, and profit target for new trade
        sell_price = btc['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
        stop_loss = sell_price + (0.8 * btc['std'].iloc[i])  # Stop loss based on volatility
        profit_target = sell_price - (1.5 * btc['std'].iloc[i])  # Profit target
        
        # Update exposure and add the new trade with trade_id
        expo -= allocated_kt_per_trade  # sell so -ve
        active_trades.append({
            'trade_id': trade_counter,  # Assign a unique ID to each trade
            'sell_price': round(sell_price, 3),
            'stop_loss': round(stop_loss, 3),
            'profit_target': round(profit_target, 3),
            'sell_index': i,
            'action': 'Sell',
            'exposure': expo
        })
        trade_counter += 1  # Increment the trade_counter for the next sell
        
        btc.at[btc.index[i], 'Position'] = -5  # Mark position as active
        trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Sell', 'Price': round(float(sell_price), 2), 'Exposure': expo, 'Trade ID': trade_counter - 1})
    
    # Iterate over all active trades to check for profit-taking or stop-loss conditions
    for trade in active_trades[:]:
        # For Buy trades (Long positions)
        if trade['action'] == 'Buy':
            # Check if 2 weeks have passed since the buy (14 days)
            if i - trade['buy_index'] >= 14:
                sell_price = btc['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
                profit = (sell_price - trade['buy_price']) * allocated_kt_per_trade # Correct profit/loss calculation
                expo -= allocated_kt_per_trade  # Update exposure with loss
                active_trades.remove(trade)  # Remove trade from active trades
                btc.at[btc.index[i], 'Position'] = -5  # Mark position as closed
                trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Forced sell out (15 pricing days)', 'Price': round(float(sell_price), 2), 'Profit': round(float(profit), 2), 'Exposure': expo, 'Trade ID': trade['trade_id']})
            
            # Check for stop loss or profit target
            elif btc['CLOSE'].iloc[i] <= trade['stop_loss']:  # If stop loss is hit
                profit = (btc['CLOSE'].iloc[i] - trade['buy_price']) * allocated_kt_per_trade  # Profit calculation for Buy
                expo -= allocated_kt_per_trade  # Update exposure with profit
                active_trades.remove(trade)  # Remove trade from active trades
                btc.at[btc.index[i], 'Position'] = -5  # Mark position as closed
                trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Stop Loss sell out', 'Price': round(float(btc['CLOSE'].iloc[i]), 2), 'Profit': round(float(profit), 2), 'Exposure': expo, 'Trade ID': trade['trade_id']})
            
            elif btc['CLOSE'].iloc[i] >= trade['profit_target']:  # If profit target is hit
                profit = (btc['CLOSE'].iloc[i] - trade['buy_price']) * allocated_kt_per_trade  # Profit calculation for Buy
                expo -= allocated_kt_per_trade  # 
                active_trades.remove(trade)  # Remove trade from active trades
                btc.at[btc.index[i], 'Position'] = -5  # Mark position as closed
                trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Profit Target sell out', 'Price': round(float(btc['CLOSE'].iloc[i]), 2), 'Profit': round(float(profit), 2), 'Exposure': expo, 'Trade ID': trade['trade_id']})

        # For Sell trades (Short positions)
        elif trade['action'] == 'Sell':
            # Check if 2 weeks have passed since the sell (14 days)
            if i - trade['sell_index'] >= 14:
                buy_price = btc['CLOSE'].iloc[i] * 1.005  # Account for 0.5% fee
                profit = (trade['sell_price'] - buy_price) * allocated_kt_per_trade # Correct profit/loss calculation
                expo += allocated_kt_per_trade  # Update exposure with loss
                active_trades.remove(trade)  # Remove trade from active trades
                btc.at[btc.index[i], 'Position'] = +5  # Mark position as closed
                trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Forced buy back (15 pricing days)', 'Price': round(float(buy_price), 2), 'Profit': round(float(profit), 2), 'Exposure': expo, 'Trade ID': trade['trade_id']})
            
            # Check for stop loss or profit target
            elif btc['CLOSE'].iloc[i] >= trade['stop_loss']:  # If stop loss is hit
                profit = (trade['sell_price'] - btc['CLOSE'].iloc[i]) * allocated_kt_per_trade  # Profit calculation for Sell
                expo += allocated_kt_per_trade  # Update exposure with profit
                active_trades.remove(trade)  # Remove trade from active trades
                btc.at[btc.index[i], 'Position'] = +5  # Mark position as closed
                trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Stop Loss buy back', 'Price': round(float(btc['CLOSE'].iloc[i]), 2), 'Profit': round(float(profit), 2), 'Exposure': expo, 'Trade ID': trade['trade_id']})
            
            elif btc['CLOSE'].iloc[i] <= trade['profit_target']:  # If profit target is hit
                profit = (trade['sell_price'] - btc['CLOSE'].iloc[i]) * allocated_kt_per_trade  # Profit calculation for Sell
                expo += allocated_kt_per_trade  # Update exposure with profit
                active_trades.remove(trade)  # Remove trade from active trades
                btc.at[btc.index[i], 'Position'] = +5  # Mark position as closed
                trade_log.append({'Date': btc.index[i].strftime('%d-%m-%y'), 'Action': 'Profit Target buy back', 'Price': round(float(btc['CLOSE'].iloc[i]), 2), 'Profit': round(float(profit), 2), 'Exposure': expo, 'Trade ID': trade['trade_id']})

# Summarize results
total_profit = sum(trade['Profit'] for trade in trade_log if 'Profit' in trade)  # Sum of all profits
num_trades = len([trade for trade in trade_log if trade['Action'] in ['Profit Target buy back', 'Stop Loss buy back', 'Profit Target sell out', 'Stop Loss sell out', 'Forced sell out (15 pricing days)', 'Forced buy back (15 pricing days)']])

print(btc)
print(f"Profit: ${total_profit:.2f}")
print(f"Number of Trades: {num_trades}")

# Plot Buy and Sell Points
buy_signals = btc[btc['Position'] == 5]
sell_signals = btc[btc['Position'] == -5]

print("Buy Signals:\n", buy_signals[['CLOSE', 'Position']])
print("Sell Signals:\n", sell_signals[['CLOSE', 'Position']])

# Plot the graph
plt.figure(figsize=[14, 7])
plt.plot(btc['CLOSE'].index, btc['CLOSE'], label='btc', color='blue')
plt.plot(btc['50'].index, btc['50'], label='50-Day Moving Average', color='orange')
plt.plot(btc['200'].index, btc['200'], label='200-Day Moving Average', color='green')

# Scatter plot for buy and sell signals
plt.scatter(buy_signals.index, buy_signals['CLOSE'], label='Buy Signal', marker='^', color='green', s=100)
plt.scatter(sell_signals.index, sell_signals['CLOSE'], label='Sell/Stop Loss Signal', marker='v', color='red', s=100)


# Annotate each trade on the chart
print("Trade Log:")
trade_details = []
latest_price = round(btc['CLOSE'][-1], 2)
latest_date = pd.to_datetime(btc.index[-1]).strftime('%d-%m-%y')

for trade in trade_log:
    print(trade)
    trade_date = datetime.strptime(trade['Date'], '%d-%m-%y')
    trade_price = trade['Price']
    trade_id = trade['Trade ID']

    if trade['Action'] == 'Buy':
        buy_trade = trade
        # Look for the corresponding Sell trade for this Buy trade
        sell_trade = next((t for t in trade_log if t['Trade ID'] == trade_id and ('Sell' in t['Action'] or 'Stop Loss sell out' in t['Action'] or 'Profit Target sell out' in t['Action'] or 'Forced sell out (15 pricing days)'in t['Action'])), None)

        if sell_trade:
            # If a corresponding sell trade exists, plot both buy and sell
            sell_date = datetime.strptime(sell_trade['Date'], '%d-%m-%y')
            sell_price = sell_trade['Price']

            # Plot Buy trade (green)
            plt.text(trade_date, trade_price, f"{trade_id}", color='green', fontsize=9, ha='left', va='bottom')

            # Plot Sell trade (red)
            plt.text(sell_date, sell_price, f"{trade_id}", color='red', fontsize=9, ha='left', va='top')

            # Calculate profit for Buy trade
            profit = round((sell_price - trade_price) * allocated_kt_per_trade, 2)

            # Append trade details
            trade_details.append([
                trade_id,
                allocated_kt_per_trade,
                buy_trade['Date'],
                buy_trade['Price'],
                sell_trade['Date'],
                sell_trade['Price'],
                profit,
                sell_trade['Action']
            ])
        else:
            # If no corresponding Sell trade found (in case it's an open position)
            # Calculate Mark-to-Market (Mtm) profit (latest price - buy price)
            mtm_profit = round((latest_price - trade_price) * allocated_kt_per_trade, 2)

            # Plot Buy trade (green)
            plt.text(trade_date, trade_price, f"{trade_id} (Buy)", color='green', fontsize=9, ha='left', va='bottom')

            # Plot "Open Position" text and Mtm profit
            plt.text(trade_date, trade_price, f"Open Position", color='orange', fontsize=9, ha='left', va='top')

            # Append trade details for open position
            trade_details.append([
                trade_id,
                allocated_kt_per_trade,
                buy_trade['Date'],
                buy_trade['Price'],
                latest_date,  # Use the latest date from the btc for open positions
                latest_price,  # Use latest price from the btc as the sell price for open positions
                mtm_profit,
                'Open Position'
            ])

    elif trade['Action'] == 'Sell':
        sell_trade = trade
        # Look for the corresponding Buy trade for this Sell trade
        buy_trade = next((t for t in trade_log if t['Trade ID'] == trade_id and ('Buy' in t['Action'] or 'Stop Loss buy back' in t['Action'] or 'Profit Target buy back' in t['Action'] or 'Forced buy back (15 pricing days)' in t['Action'])), None)

        if buy_trade:
            # If a corresponding buy trade exists, plot both sell and buy
            buy_date = datetime.strptime(buy_trade['Date'], '%d-%m-%y')
            buy_price = buy_trade['Price']

            # Plot Sell trade (red)
            plt.text(trade_date, trade_price, f"{trade_id}", color='red', fontsize=9, ha='left', va='top')

            # Plot Buy trade (green)
            plt.text(buy_date, buy_price, f"{trade_id}", color='green', fontsize=9, ha='left', va='bottom')

            # Calculate profit for Sell trade
            profit = round((trade_price - buy_price) * allocated_kt_per_trade, 2)

            # Append trade details
            trade_details.append([
                trade_id,
                allocated_kt_per_trade,
                buy_trade['Date'],
                buy_trade['Price'],
                sell_trade['Date'],
                sell_trade['Price'],
                profit,
                buy_trade['Action']
            ])
        else:            
            mtm_profit = round((trade_price - latest_price) * allocated_kt_per_trade, 2)

            # Plot Buy trade (green)
            plt.text(trade_date, trade_price, f"{trade_id}", color='green', fontsize=9, ha='left', va='bottom')

            # Plot "Open Position" text and Mtm profit
            plt.text(trade_date, trade_price, f"Open Position", color='orange', fontsize=9, ha='left', va='top')

            # Append trade details for open position
            trade_details.append([
                trade_id,
                allocated_kt_per_trade,
                latest_date,
                latest_price,
                sell_trade['Date'],
                sell_trade['Price'],  
                mtm_profit,
                'Open Position'
            ])

# Add legend and title
plt.legend(loc='upper left', fontsize='small')
plt.title('BTC Backtest')
plt.xlabel('Date')
plt.ylabel('Price $')
plt.grid(True)
# Add text outside the plot area (bottom left of the figure)
plt.figtext(0.95, 0.95, f"Latest Date: {latest_date}", color='black', fontsize=8, ha='right', va='bottom')
plt.figtext(0.95, 0.93, f"Latest Price: ${latest_price:.2f}", color='black', fontsize=8, ha='right', va='bottom')
plt.figtext(0.95, 0.91, f"Total Trades: {num_trades}", color='black', fontsize=8, ha='right', va='bottom')
plt.figtext(0.95, 0.89, f"Total Profit: ${total_profit:.2f}", color='black', fontsize=8, ha='right', va='bottom')


# Prepare data for the table, including the reason for profit-taking
column_labels = ['Trade ID', 'Position (kt)', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit', 'Action']

# Only plot the table if there are trade details
if trade_details:
    # Create the table as a separate figure
    fig, ax = plt.subplots(figsize=(8, 12))
    fig.subplots_adjust(top=0.9)
    fig.text(0.5, 0.97, 'Trade Details', ha='center', va='center', fontsize=12, fontweight='bold')
    summary_text = f"Total Profit: ${total_profit:.2f}\nNumber of Trades: {num_trades}"
    fig.text(0.5, 0.95, summary_text, ha='center', va='center', fontsize=8, fontweight='bold')
    ax.axis('off')  # Turn off the axis
    table = plt.table(cellText=trade_details, colLabels=column_labels, loc='center', cellLoc='center', colColours=['#f2f2f2'] * len(column_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(column_labels))))

    # Show the plot
    plt.show()
else:
    print("No trade details available for table display.")
