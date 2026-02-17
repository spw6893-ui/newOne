"""
Fetch cryptocurrency derivatives data including funding rate, open interest, and liquidations
"""
try:
    import ccxt
except ImportError:
    raise ImportError("ccxt not installed. Run: pip install ccxt")

import pandas as pd
from datetime import datetime, timedelta
import os
import time
import numpy as np

def fetch_derivatives_data(
    symbols=['BTC/USDT:USDT', 'ETH/USDT:USDT'],
    timeframe='1h',
    start_date='2020-01-01',
    end_date='2024-12-31',
    exchange_name='binance',
    output_dir='AlphaQCM_data/crypto_derivatives',
    include_funding_rate=True,
    include_open_interest=True,
    include_liquidations=True
):
    """
    Fetch derivatives-specific data for perpetual futures

    Args:
        symbols: List of perpetual futures pairs
        timeframe: Candle timeframe
        start_date: Start date
        end_date: End date
        exchange_name: Exchange name
        output_dir: Output directory
        include_funding_rate: Fetch funding rate data
        include_open_interest: Fetch open interest data
        include_liquidations: Fetch liquidation data
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize exchange
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'enableRateLimit': True,
        'defaultType': 'swap'
    })

    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    failed_symbols = []

    for symbol in symbols:
        print(f"\nFetching derivatives data for {symbol}...")

        try:
            # 1. Fetch Funding Rate History
            if include_funding_rate:
                print(f"  Fetching funding rate...")
                funding_rates = []
                current_ts = start_ts

                while current_ts < end_ts:
                    try:
                        rates = exchange.fetch_funding_rate_history(
                            symbol,
                            since=current_ts,
                            limit=1000
                        )
                        if not rates:
                            break
                        funding_rates.extend(rates)
                        current_ts = rates[-1]['timestamp'] + 1
                        time.sleep(0.2)
                    except Exception as e:
                        print(f"    Error fetching funding rate: {e}")
                        break

                if funding_rates:
                    df_funding = pd.DataFrame(funding_rates)
                    df_funding['datetime'] = pd.to_datetime(df_funding['timestamp'], unit='ms', utc=True)
                    df_funding = df_funding.set_index('datetime')
                    df_funding = df_funding[['fundingRate']].rename(columns={'fundingRate': 'funding_rate'})

                    # Save funding rate data
                    funding_file = os.path.join(output_dir, f"{symbol.replace('/', '_')}_funding_rate.csv")
                    df_funding.to_csv(funding_file)
                    print(f"    Saved funding rate ({len(df_funding)} rows)")

            # 2. Fetch Open Interest (current snapshot only, historical data limited)
            if include_open_interest:
                print(f"  Fetching open interest...")
                try:
                    # Binance only provides current OI snapshot via REST API
                    # For historical OI, we need to collect snapshots over time
                    oi_current = exchange.fetch_open_interest(symbol)
                    if oi_current:
                        print(f"    Current OI: {oi_current.get('openInterestAmount', 'N/A')}")
                        # Note: Historical OI data requires continuous collection
                        # For now, we'll skip historical OI and focus on funding rate
                except Exception as e:
                    print(f"    Open interest not available: {e}")

            # 3. Fetch Liquidations (if available)
            if include_liquidations:
                print(f"  Fetching liquidations...")
                try:
                    # Note: Liquidation data is usually only available for recent periods
                    # Binance provides liquidation orders via REST API
                    liquidations = []
                    current_ts = max(start_ts, int((datetime.now() - timedelta(days=30)).timestamp() * 1000))

                    while current_ts < end_ts:
                        try:
                            liq = exchange.fetch_liquidations(
                                symbol,
                                since=current_ts,
                                limit=1000
                            )
                            if not liq:
                                break
                            liquidations.extend(liq)
                            if len(liq) > 0:
                                current_ts = liq[-1]['timestamp'] + 1
                            else:
                                break
                            time.sleep(0.2)
                        except Exception as e:
                            print(f"    Liquidation data not available: {e}")
                            break

                    if liquidations:
                        df_liq = pd.DataFrame(liquidations)
                        df_liq['datetime'] = pd.to_datetime(df_liq['timestamp'], unit='ms', utc=True)
                        df_liq = df_liq.set_index('datetime')

                        # Aggregate liquidations by hour
                        df_liq_agg = df_liq.resample('1H').agg({
                            'amount': 'sum',
                            'price': 'mean',
                            'side': lambda x: (x == 'sell').sum()  # Count of long liquidations
                        }).rename(columns={
                            'amount': 'liq_amount',
                            'price': 'liq_price',
                            'side': 'liq_long_count'
                        })

                        # Save liquidation data
                        liq_file = os.path.join(output_dir, f"{symbol.replace('/', '_')}_liquidations.csv")
                        df_liq_agg.to_csv(liq_file)
                        print(f"    Saved liquidations ({len(df_liq_agg)} rows)")
                except Exception as e:
                    print(f"    Liquidation data not available: {e}")

            print(f"  ✓ Completed {symbol}")

        except Exception as e:
            print(f"  ✗ Failed to fetch {symbol}: {e}")
            failed_symbols.append(symbol)

    if failed_symbols:
        print(f"\nFailed symbols: {', '.join(failed_symbols)}")

    print(f"\nDerivatives data saved to {output_dir}")

if __name__ == '__main__':
    import sys

    # Load symbols from file
    symbol_file = 'data_collection/top100_perp_symbols.txt'
    if os.path.exists(symbol_file):
        with open(symbol_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"Loaded {len(symbols)} perpetual futures symbols")
    else:
        symbols = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT'
        ]
        print(f"Using default top 5 symbols")

    fetch_derivatives_data(
        symbols=symbols,
        timeframe='1h',
        start_date='2020-01-01',
        end_date='2025-02-15',
        exchange_name='binance',
        include_funding_rate=True,
        include_open_interest=True,
        include_liquidations=True
    )
