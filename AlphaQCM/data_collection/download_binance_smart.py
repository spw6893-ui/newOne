"""
Download and process Binance historical archive data
Strategy: Download -> Process -> Delete raw data to save space
"""
import os
import requests
import gzip
import pandas as pd
from datetime import datetime, timedelta

def download_and_extract(url, temp_path):
    """Download .gz file and extract to CSV"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(temp_path, 'wb') as f:
                f.write(response.content)

            # Extract
            csv_path = temp_path.replace('.gz', '')
            with gzip.open(temp_path, 'rb') as f_in:
                with open(csv_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            os.remove(temp_path)
            return csv_path
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_metrics(csv_path, symbol, date):
    """Process metrics file and return hourly aggregated data"""
    df = pd.read_csv(csv_path)
    # Aggregate to hourly and return
    # ... processing logic
    os.remove(csv_path)  # Delete raw file immediately
    return df

def download_symbol_date(symbol, date, data_type):
    """Download single symbol for single date"""
    base_url = "https://data.binance.vision/data/futures/um/daily"
    date_str = date.strftime('%Y-%m-%d')

    # Try different URL formats
    urls = [
        f"{base_url}/{data_type}/{symbol}/{symbol}-{data_type}-{date_str}.zip",
        f"{base_url}/{data_type}/{symbol}/{symbol}-{data_type}-{date_str}.csv.gz",
    ]

    for url in urls:
        temp_path = f"/tmp/{symbol}_{data_type}_{date_str}.tmp"
        result = download_and_extract(url, temp_path)
        if result:
            return result

    return None

# Test with one symbol and one date
symbol = "BTCUSDT"
date = datetime(2024, 2, 1)

print(f"Testing {symbol} on {date.strftime('%Y-%m-%d')}...")
result = download_symbol_date(symbol, date, "metrics")
if result:
    print(f"✓ Downloaded: {result}")
else:
    print("✗ Failed")
