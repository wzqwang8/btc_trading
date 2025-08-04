import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)

# Load and process data
file_path = f'M:/24.Naphtha/Python scripts/bot_testing/btc_gbp_hourly.csv'
btc = pd.read_csv(file_path)
btc['time'] = pd.to_datetime(btc['time'])  
btc.set_index('time', inplace=True)

# Moving Averages
btc['50'] = btc['close'].rolling(window=50).mean()
btc['200'] = btc['close'].rolling(window=200).mean()

# Shifted close
btc['h-1'] = btc['close'].shift(1)
btc.iloc[0, btc.columns.get_loc('h-1')] = btc.iloc[1, btc.columns.get_loc('h-1')]  # Fix first row

# Std Dev
btc['std'] = btc['close'].rolling(window=14).std()

# Bollinger Bands
window = 20
std_dev = 2
btc['BB_Mid'] = btc['close'].rolling(window).mean()
btc['BB_Std'] = btc['close'].rolling(window).std()
btc['BB_High'] = btc['BB_Mid'] + std_dev * btc['BB_Std']
btc['BB_Low'] = btc['BB_Mid'] - std_dev * btc['BB_Std']

# RSI
delta = btc['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
btc['RSI'] = 100 - (100 / (1 + rs))

# ATR
btc['TR'] = btc[['high', 'low', 'h-1']].apply(
    lambda row: max(
        row['high'] - row['low'], 
        abs(row['high'] - row['h-1']), 
        abs(row['low'] - row['h-1'])
    ),
    axis=1
)
btc['ATR'] = btc['TR'].rolling(window=14).mean()

# --- Volume Indicators ---
btc['obv'] = (np.sign(btc['close'].diff()) * btc['volume']).fillna(0).cumsum()
btc['vol_ma_20'] = btc['volume'].rolling(window=20).mean()
btc['vol_spike'] = btc['volume'] > (1.5 * btc['vol_ma_20'])

btc.dropna(inplace=True)
print(btc)


# # Buy Signal Condition (RSI < 30, Close < BB_Low, Volume Spike)
# btc['buy_signal'] = (
#     (btc['RSI'] < 30) &                   # RSI < 30 (oversold)
#     (btc['close'] < btc['BB_Low']) &      # Price touches or crosses the lower BB
#     (btc['vol_spike'])                     # Volume spike
# )

# btc['sell_signal'] = (
#     (btc['RSI'] > 70) &                  # RSI overbought
#     (btc['close'] > btc['BB_High']) &    # Price breaks above upper Bollinger Band
#     (btc['vol_spike'])                   # Volume spike (exit on strength)
# )


# #### ML COMPARE ####
# df_all = btc.dropna().astype(float)

# # Split data into features and target
# x = df_all.drop(columns=['Nap'])
# y = df_all['Nap']

# # Split data into train and test sets
# train_size = int(len(df_all) * 0.8)
# x_train, x_test = x[:train_size], x[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# # Define models
# models = {
#     'Linear Regression': LinearRegression(),
#     'Ridge Regression': Ridge(),
#     'Lasso Regression': Lasso(),
#     'Decision Tree': DecisionTreeRegressor(),
#     'Random Forest': RandomForestRegressor(),
#     'Gradient Boosting': GradientBoostingRegressor(),
#     'Support Vector Regression': SVR(),
#     'K-Nearest Neighbors': KNeighborsRegressor()
# }

# errors = {}
# predictions = {}
# # Evaluate models
# for name, model in models.items():
#     print(f"Evaluating model: {name}")
#     model.fit(x_train, y_train)
    
#     y_pred = model.predict(x_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
    
#     errors[name] = {'MSE': mse, 'R2': r2}
#     predictions[name] = y_pred
    
#     print(f"{name} - Mean Squared Error: {mse:.2f}")
#     print(f"{name} - R-squared: {r2:.2f}")

# # Convert results to DataFrame
# errors_df_all = pd.DataFrame(errors).T
# print(errors_df_all)

# # Find the best model based on R-squared
# best_model_name = errors_df_all['R2'].idxmax()
# best_model = models[best_model_name]

# print(f"The best model is: {best_model_name}")

# # Train the best model
# best_model.fit(x_train, y_train)

# # Prediction and evaluation
# y_pred_best = best_model.predict(x_test)
# mse_best = mean_squared_error(y_test, y_pred_best)
# r2_best = r2_score(y_test, y_pred_best)

# results_df_best = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_best})
# print(results_df_best)
# print(f'Mean Squared Error: {mse_best:.2f}')
# print(f'R-squared: {r2_best:.2f}')

# # Cross-validation using TimeSeriesSplit
# tscv = TimeSeriesSplit(n_splits=5)
# cv_scores = []
# for train_index, test_index in tscv.split(x):
#     x_train_cv, x_test_cv = x.iloc[train_index], x.iloc[test_index]
#     y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
#     best_model.fit(x_train_cv, y_train_cv)
#     y_pred_cv = best_model.predict(x_test_cv)
#     cv_scores.append(mean_squared_error(y_test_cv, y_pred_cv))

# average_mse = np.mean(cv_scores)
# print(f'Average Cross-Validated MSE: {average_mse:.2f}')

# # Plotting Actual vs Predicted for each model
# plt.figure(figsize=(14, 7))

# # Actual values
# plt.plot(y_test.index, y_test, label='Actual', color='black', linestyle='-', linewidth=2)

# # Predicted values for each model
# for name, y_pred in predictions.items():
#     plt.plot(y_test.index, y_pred, label=f'Predicted ({name})', linestyle='--')

# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.title('Actual vs Predicted Values for Each Model')
# plt.legend()
# plt.grid(True)
# plt.show()


# # SHAP values for the best model
# explainer = shap.Explainer(best_model, x_train)
# shap_values = explainer(x_test)

# plt.figure(figsize=(14, 7))
# shap.summary_plot(shap_values, x_test)

# plt.show()


##### PLOTTING ###############
# # Last 40 days of data
# last_year = btc.loc[btc.index >= btc.index.max() - pd.Timedelta(days=40)]

# # Plotting with 5 subplots
# fig, axs = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

# # Price and MAs with Buy Signal Marked
# axs[0].plot(last_year.index, last_year['close'], label='Close', color='black')
# axs[0].plot(last_year.index, last_year['50'], label='50 MA', color='blue')
# axs[0].plot(last_year.index, last_year['200'], label='200 MA', color='orange')
# axs[0].plot(last_year.index, last_year['BB_Mid'], label='BB Mid', linestyle='--')
# axs[0].plot(last_year.index, last_year['BB_High'], label='BB High', linestyle='--', color='green')
# axs[0].plot(last_year.index, last_year['BB_Low'], label='BB Low', linestyle='--', color='red')

# # Highlight Buy Signals with arrows or markers
# # Highlight Buy and Sell Signals
# buy_signals = last_year[last_year['buy_signal']]
# sell_signals = last_year[last_year['sell_signal']]

# axs[0].scatter(buy_signals.index, buy_signals['close'], color='blue', marker='^', label='Buy Signal', s=100)
# axs[0].scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', label='Sell Signal', s=100)


# axs[0].set_title('BTC Price with MAs, Bollinger Bands, and Buy Signals')
# axs[0].legend()
# axs[0].grid(True)

# # RSI
# axs[1].plot(last_year.index, last_year['RSI'], label='RSI', color='green')
# axs[1].axhline(70, color='red', linestyle='--', alpha=0.5)
# axs[1].axhline(30, color='blue', linestyle='--', alpha=0.5)
# axs[1].set_title('RSI (14)')
# axs[1].legend()
# axs[1].grid(True)

# # ATR
# axs[2].plot(last_year.index, last_year['ATR'], label='ATR', color='purple')
# axs[2].set_title('ATR (14)')
# axs[2].legend()
# axs[2].grid(True)

# # Std Dev
# axs[3].plot(last_year.index, last_year['std'], label='Rolling Std (14)', color='darkgray')
# axs[3].set_title('Standard Deviation (14)')
# axs[3].legend()
# axs[3].grid(True)

# # Volume / OBV
# axs[4].bar(last_year.index, last_year['volume'], color=['red' if spike else 'gray' for spike in last_year['vol_spike']], label='Volume', alpha=0.4)
# axs[4].plot(last_year.index, last_year['vol_ma_20'], label='Vol MA (20)', color='orange')
# axs[4].plot(last_year.index, last_year['obv'] / 1e6, label='OBV (scaled)', color='green')  # OBV scaled for visual fit

# axs[4].set_title('Volume, OBV and Volume MA (20)')
# axs[4].legend()
# axs[4].grid(True)

# plt.tight_layout()
# plt.show()

