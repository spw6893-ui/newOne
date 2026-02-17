"""
Fetch cryptocurrency data using CCXT library
"""
try:
    import ccxt
except ImportError:
    raise ImportError("ccxt not installed. Run: pip install ccxt")

import pandas as pd
from datetime import datetime
import os
import time

def fetch_crypto_ohlcv(
    symbols=['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT'],
    timeframe='1h',
    start_date='2020-01-01',
    end_date='2024-12-31',
    exchange_name='binance',
    output_dir='AlphaQCM_data/crypto_data',
    vwap_window=24,
    market_type='swap'
):
    """
    Fetch OHLCV data for crypto symbols

    Args:
        symbols: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
        timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        exchange_name: Exchange name (binance, okx, bybit, etc.)
        output_dir: Directory to save data
        vwap_window: Rolling window for VWAP calculation (default 24 periods)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize exchange
    exchange_class = getattr(ccxt, exchange_name)
    options = {'enableRateLimit': True}
    if market_type == 'swap':
        options['defaultType'] = 'swap'  # 永续合约
    exchange = exchange_class(options)

    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    all_data = {}
    failed_symbols = []

    for symbol in symbols:
        print(f"Fetching {symbol}...")
        ohlcv_data = []
        current_ts = start_ts

        while current_ts < end_ts:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
                if not ohlcv:
                    break
                ohlcv_data.extend(ohlcv)
                current_ts = ohlcv[-1][0] + 1
                print(f"  Fetched up to {datetime.fromtimestamp(current_ts/1000, tz=None).strftime('%Y-%m-%d %H:%M')}")
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")
                failed_symbols.append(symbol)
                break

        if not ohlcv_data:
            print(f"  No data for {symbol}, skipping")
            failed_symbols.append(symbol)
            continue

        # Convert to DataFrame with UTC timezone
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('datetime')
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='first')].sort_index()

        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Calculate rolling VWAP (typical price weighted by volume)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).rolling(window=vwap_window, min_periods=1).sum() / \
                     df['volume'].rolling(window=vwap_window, min_periods=1).sum()

        # Validate data
        if df.isnull().any().any():
            print(f"  Warning: {symbol} still has NaN values after filling")

        all_data[symbol.replace('/', '_')] = df

        # Save individual symbol
        symbol_file = os.path.join(output_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")
        df.to_csv(symbol_file)
        print(f"  Saved {symbol} to {symbol_file} ({len(df)} rows)")

    if failed_symbols:
        print(f"\nFailed to fetch: {', '.join(failed_symbols)}")

    return all_data

if __name__ == '__main__':
    # Load perpetual futures symbols from file
    import sys

    symbol_file = 'data_collection/top100_perp_symbols.txt'
    if os.path.exists(symbol_file):
        with open(symbol_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"Loaded {len(symbols)} perpetual futures symbols from {symbol_file}")
    else:
        # Fallback to top 20 perpetual futures
        symbols = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
            'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOGE/USDT:USDT', 'DOT/USDT:USDT', 'MATIC/USDT:USDT',
            'LINK/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'LTC/USDT:USDT', 'ETC/USDT:USDT',
            'APT/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT', 'INJ/USDT:USDT', 'SUI/USDT:USDT'
        ]
        print(f"Using default top 20 perpetual futures symbols")

    fetch_crypto_ohlcv(
        symbols=symbols,
        timeframe='1h',
        start_date='2018-01-01',
        end_date='2025-02-15',
        exchange_name='binance',
        vwap_window=24,
        market_type='swap'
    )
