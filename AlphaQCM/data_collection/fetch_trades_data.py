"""
Fetch trade-level data for taker buy ratio calculation
"""
try:
    import ccxt
except ImportError:
    raise ImportError("ccxt not installed. Run: pip install ccxt")

import pandas as pd
from datetime import datetime, timedelta
import os
import time

def fetch_trades_for_taker_ratio(
    symbols,
    start_date='2020-01-01',
    end_date='2025-02-15',
    exchange_name='binance',
    output_dir='AlphaQCM_data/crypto_trades'
):
    """
    Fetch trade data and aggregate taker buy ratio hourly

    Taker buy ratio = taker_buy_volume / total_volume
    High ratio indicates buying pressure
    """
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
        print(f"Fetching trades for {symbol}...")

        try:
            # Fetch trades in chunks (limited to recent data due to API constraints)
            # Most exchanges only provide ~7 days of trade history via REST API
            recent_start = max(start_ts, int((datetime.now() - timedelta(days=7)).timestamp() * 1000))

            trades = []
            current_ts = recent_start

            while current_ts < end_ts:
                try:
                    trade_batch = exchange.fetch_trades(symbol, since=current_ts, limit=1000)
                    if not trade_batch:
                        break
                    trades.extend(trade_batch)
                    current_ts = trade_batch[-1]['timestamp'] + 1
                    print(f"  {datetime.fromtimestamp(current_ts/1000).strftime('%Y-%m-%d %H:%M')}")
                    time.sleep(0.2)
                except Exception as e:
                    print(f"  Error fetching trades: {e}")
                    break

            if not trades:
                print(f"  No trade data for {symbol}")
                failed_symbols.append(symbol)
                continue

            # Convert to DataFrame
            df = pd.DataFrame(trades)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('datetime')

            # Calculate taker buy volume (side='buy' means taker is buyer)
            df['taker_buy_volume'] = df.apply(
                lambda x: x['amount'] if x['side'] == 'buy' else 0, axis=1
            )

            # Aggregate to hourly
            df_hourly = df.resample('1H').agg({
                'amount': 'sum',  # Total volume
                'taker_buy_volume': 'sum'
            }).rename(columns={'amount': 'total_volume'})

            # Calculate taker buy ratio
            df_hourly['taker_buy_ratio'] = df_hourly['taker_buy_volume'] / df_hourly['total_volume']
            df_hourly['taker_buy_ratio'] = df_hourly['taker_buy_ratio'].fillna(0.5)  # Neutral if no data

            # Save
            output_file = os.path.join(output_dir, f"{symbol.replace('/', '_')}_taker_ratio.csv")
            df_hourly.to_csv(output_file)
            print(f"  Saved {len(df_hourly)} rows to {output_file}")

        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            failed_symbols.append(symbol)

    if failed_symbols:
        print(f"\nFailed: {', '.join(failed_symbols)}")

    print(f"\nTrade data saved to {output_dir}")
    print("\nNote: Most exchanges only provide ~7 days of trade history via REST API")
    print("For historical taker buy ratio, consider using aggTrades endpoint or websocket collection")

if __name__ == '__main__':
    symbol_file = 'data_collection/top100_perp_symbols.txt'
    if os.path.exists(symbol_file):
        with open(symbol_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"Loaded {len(symbols)} symbols")
    else:
        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT']
        print("Using default top 5 symbols")

    fetch_trades_for_taker_ratio(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2025-02-15',
        exchange_name='binance'
    )
