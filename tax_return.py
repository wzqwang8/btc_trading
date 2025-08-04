###pip install forex-python

import requests
import time
import hmac
import hashlib
import pandas as pd
from datetime import datetime
from forex_python.converter import CurrencyRates

# ==== CONFIG ====
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
API_BASE = 'https://api.coinbase.com/api/v3/brokerage'
TARGET_CURRENCY = 'USD'

c = CurrencyRates()

# ==== AUTHENTICATED REQUEST ====
def coinbase_request(method, endpoint, params=None):
    timestamp = str(int(time.time()))
    request_path = f'/brokerage{endpoint}'
    body = '' if method == 'GET' else (params or '')
    if isinstance(body, dict):
        import json
        body = json.dumps(body)

    message = timestamp + method + request_path + body
    signature = hmac.new(API_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()

    headers = {
        'CB-ACCESS-KEY': API_KEY,
        'CB-ACCESS-SIGN': signature,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'Content-Type': 'application/json',
    }

    url = f'{API_BASE}{endpoint}'
    response = requests.request(method, url, headers=headers, params=params if method == 'GET' else None, data=body if method != 'GET' else None)

    if response.status_code != 200:
        raise Exception(f"Coinbase API error: {response.status_code} - {response.text}")

    return response.json()

# ==== GET FILLS ====
def get_all_fills():
    print("📥 Fetching fills...")
    fills = []
    cursor = None

    while True:
        params = {'limit': 100}
        if cursor:
            params['cursor'] = cursor
        data = coinbase_request('GET', '/orders/historical/fills', params=params)
        batch = data.get('fills', [])
        fills.extend(batch)
        if not data.get('has_next'):
            break
        cursor = data.get('cursor')

    filtered_fills = [
        {
            'trade_time': f['trade_time'],
            'side': f['side'],
            'product_id': f['product_id'],
            'price': float(f['price']),
            'size': float(f['size']),
            'fee': float(f['fee']),
            'fee_currency': f['fee_currency']
        }
        for f in fills if f['product_id'].endswith(f'-{TARGET_CURRENCY}')
    ]

    df = pd.DataFrame(filtered_fills)
    df['trade_time'] = pd.to_datetime(df['trade_time'])
    df = df.sort_values('trade_time')
    df.to_csv('coinbase_advanced_fills.csv', index=False)
    print(f"✅ Got {len(df)} USD trades")
    return df

# ==== CURRENCY CONVERSION ====
def convert_to_gbp(df):
    print("💱 Converting to GBP...")
    df['total_usd'] = df['price'] * df['size']
    df['gbp_rate'] = df['trade_time'].dt.date.apply(lambda d: c.get_rate('USD', 'GBP', d))
    df['total_gbp'] = df['total_usd'] * df['gbp_rate']
    return df

# ==== SHARE POOLING ====
def share_pooling_gains(df):
    print("🧮 Calculating gains (share pooling)...")
    pool_units = 0.0
    pool_cost = 0.0
    gains = []

    for _, row in df.iterrows():
        side = row['side'].upper()
        date = row['trade_time']
        units = row['size']
        proceeds_or_cost = row['total_gbp']

        if side == 'BUY':
            pool_units += units
            pool_cost += proceeds_or_cost
        elif side == 'SELL':
            if pool_units == 0:
                print(f"⚠️ Selling with zero pool on {date}")
                continue
            avg_cost_per_unit = pool_cost / pool_units
            cost_basis = units * avg_cost_per_unit
            gain = proceeds_or_cost - cost_basis

            pool_units -= units
            pool_cost -= cost_basis

            gains.append({
                'sell_date': date,
                'units_sold': units,
                'proceeds_gbp': proceeds_or_cost,
                'cost_basis_gbp': cost_basis,
                'gain_gbp': gain
            })

    return pd.DataFrame(gains)

# ==== RUN ALL ====
df = get_all_fills()
df = convert_to_gbp(df)
report = share_pooling_gains(df)
report.to_csv('uk_share_pool_tax_report.csv', index=False)
print("📤 Report saved to uk_share_pool_tax_report.csv")

