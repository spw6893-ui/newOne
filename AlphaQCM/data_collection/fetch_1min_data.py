"""
Fetch 1-minute candle data for microstructure feature aggregation
"""
try:
    import ccxt
except ImportError:
    raise ImportError("ccxt not installed. Run: pip install ccxt")

import pandas as pd
from datetime import datetime
import os
import time

def fetch_1min_ohlcv(
    symbols,
    start_date='2020-01-01',
    end_date='2025-02-15',
    exchange_name='binance',
    output_dir='AlphaQCM_data/crypto_1min'
):
    """Fetch 1-minute OHLCV data for microstructure aggregation"""
    os.makedirs(output_dir, exist_ok=True)

    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'enableRateLimit': True,
        'defaultType': 'swap'
    })

    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    failed_symbols = []

    for symbol in symbols:
        print(f"Fetching 1min data for {symbol}...")
        ohlcv_data = []
        current_ts = start_ts

        while current_ts < end_ts:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, '1m', since=current_ts, limit=1000)
                if not ohlcv:
                    break
                ohlcv_data.extend(ohlcv)
                current_ts = ohlcv[-1][0] + 60000  # +1 minute
                print(f"  {datetime.fromtimestamp(current_ts/1000).strftime('%Y-%m-%d %H:%M')}")
                time.sleep(0.1)
            except Exception as e:
                print(f"  Error: {e}")
                failed_symbols.append(symbol)
                break

        if not ohlcv_data:
            print(f"  No data for {symbol}")
            failed_symbols.append(symbol)
            continue

        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('datetime')
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df[~df.index.duplicated(keep='first')].sort_index()

        output_file = os.path.join(output_dir, f"{symbol.replace('/', '_')}_1m.csv")
        df.to_csv(output_file)
        print(f"  Saved {len(df)} rows to {output_file}")

    if failed_symbols:
        print(f"\nFailed: {', '.join(failed_symbols)}")

    print(f"\n1-minute data saved to {output_dir}")

if __name__ == '__main__':
    symbol_file = 'data_collection/top100_perp_symbols.txt'
    if os.path.exists(symbol_file):
        with open(symbol_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"Loaded {len(symbols)} symbols")
    else:
        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT']
        print("Using default top 5 symbols")

    fetch_1min_ohlcv(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2025-02-15',
        exchange_name='binance'
    )
