"""
Download Binance historical data archive for advanced contract features
Data source: https://data.binance.vision/?prefix=data/futures/um/daily/
"""
import os
import requests
from datetime import datetime, timedelta
import gzip
import shutil

def download_file(url, output_path):
    """Download file with progress"""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

def decompress_gz(gz_path, output_path):
    """Decompress .gz file"""
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)

def download_binance_data(symbol, date, data_type, output_dir):
    """
    Download Binance historical data

    data_type options:
    - metrics: OI, mark price, index price
    - liquidationSnapshot: liquidation data
    - aggTrades: aggregate trades for CVD
    """
    base_url = "https://data.binance.vision/data/futures/um/daily"
    date_str = date.strftime('%Y-%m-%d')

    filename = f"{symbol}-{data_type}-{date_str}.csv"
    url = f"{base_url}/{data_type}/{symbol}/{filename}.gz"

    os.makedirs(output_dir, exist_ok=True)
    gz_path = os.path.join(output_dir, f"{filename}.gz")
    csv_path = os.path.join(output_dir, filename)

    if os.path.exists(csv_path):
        return True

    if download_file(url, gz_path):
        decompress_gz(gz_path, csv_path)
        return True
    return False

def main():
    # Test download for BTC
    symbol = "BTCUSDT"
    date = datetime(2024, 1, 1)

    print("Testing download for BTC on 2024-01-01...")

    # Test metrics (OI, prices)
    if download_binance_data(symbol, date, "metrics", "AlphaQCM_data/binance_archive/metrics"):
        print("✓ Metrics downloaded")
    else:
        print("✗ Metrics failed")

    # Test liquidations
    if download_binance_data(symbol, date, "liquidationSnapshot", "AlphaQCM_data/binance_archive/liquidations"):
        print("✓ Liquidations downloaded")
    else:
        print("✗ Liquidations failed")

    # Test aggTrades
    if download_binance_data(symbol, date, "aggTrades", "AlphaQCM_data/binance_archive/aggTrades"):
        print("✓ aggTrades downloaded")
    else:
        print("✗ aggTrades failed")

if __name__ == '__main__':
    main()
