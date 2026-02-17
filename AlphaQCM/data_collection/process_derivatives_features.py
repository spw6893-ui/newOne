"""
Process derivatives-specific features for perpetual futures
Handles leverage, cost, liquidation, and mechanism design dimensions
"""
import pandas as pd
import numpy as np
import os
import glob

def process_single_symbol(
    symbol,
    min1_dir='AlphaQCM_data/crypto_1min',
    funding_dir='AlphaQCM_data/crypto_derivatives',
    output_dir='AlphaQCM_data/crypto_derivatives_features'
):
    """
    Process derivatives features for a single symbol (memory efficient)

    Features:
    1. Price Triad: basis_pct (Last - Index) / Index
    2. OI Delta: Change in open interest (USD normalized)
    3. Funding Rate: avg, delta, annualized
    4. Liquidations: long/short liquidation sums
    5. CVD: Taker buy - taker sell volume
    6. Long/Short Ratio: Account ratio (if available)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load 1-minute data
    min1_file = os.path.join(min1_dir, f"{symbol}_1m.csv")
    if not os.path.exists(min1_file):
        return None

    df = pd.read_csv(min1_file, index_col=0, parse_dates=True)

    # Aggregate to hourly
    df_hourly = df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Load funding rate data
    funding_file = os.path.join(funding_dir, f"{symbol}_funding_rate.csv")
    if os.path.exists(funding_file):
        df_funding = pd.read_csv(funding_file, index_col=0, parse_dates=True)

        # Resample funding to hourly (forward fill)
        df_funding_hourly = df_funding.reindex(df_hourly.index, method='ffill')

        # Funding rate features
        df_hourly['funding_rate'] = df_funding_hourly['funding_rate']
        df_hourly['funding_rate_annualized'] = df_hourly['funding_rate'] * 365 * 3  # 8h intervals
        df_hourly['funding_delta'] = df_hourly['funding_rate'].diff()

        # Arbitrage pressure flag (funding > 30% annualized)
        df_hourly['funding_pressure'] = (df_hourly['funding_rate_annualized'] > 0.30).astype(int)

    # Note: Index price, Mark price, OI, Liquidations require real-time collection
    # These are placeholders for when you have the data

    # Basis (requires index price - not available via historical API)
    # df_hourly['basis_pct'] = (df_hourly['close'] - df_hourly['index_price']) / df_hourly['index_price']

    # OI Delta (requires historical OI - not available via REST API)
    # df_hourly['oi_usd_delta'] = df_hourly['oi_usd'].pct_change()

    # Liquidations (requires historical liquidation data - only last 30 days available)
    # df_hourly['liq_long_sum'] = ...
    # df_hourly['liq_short_sum'] = ...

    # CVD (requires aggTrades data with isBuyerMaker flag)
    # df_hourly['taker_delta'] = df_hourly['taker_buy_vol'] - df_hourly['taker_sell_vol']

    # Drop NaN rows
    df_hourly = df_hourly.dropna()

    # Save
    output_file = os.path.join(output_dir, f'{symbol}_derivatives.csv')
    df_hourly.to_csv(output_file)

    return len(df_hourly)

def process_all_symbols(
    min1_dir='AlphaQCM_data/crypto_1min',
    funding_dir='AlphaQCM_data/crypto_derivatives',
    output_dir='AlphaQCM_data/crypto_derivatives_features'
):
    """Process all symbols one by one (memory efficient)"""

    csv_files = glob.glob(os.path.join(min1_dir, '*_1m.csv'))

    results = []
    for csv_file in csv_files:
        symbol = os.path.basename(csv_file).replace('_1m.csv', '')
        print(f"Processing {symbol}...")

        try:
            rows = process_single_symbol(symbol, min1_dir, funding_dir, output_dir)
            if rows:
                results.append({'symbol': symbol, 'rows': rows})
                print(f"  ✓ {rows} rows")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Save summary
    df_summary = pd.DataFrame(results)
    summary_file = os.path.join(output_dir, 'processing_summary.csv')
    df_summary.to_csv(summary_file, index=False)
    print(f"\nProcessed {len(results)} symbols")
    print(f"Summary saved to {summary_file}")

if __name__ == '__main__':
    process_all_symbols(
        min1_dir='AlphaQCM_data/crypto_1min',
        funding_dir='AlphaQCM_data/crypto_derivatives',
        output_dir='AlphaQCM_data/crypto_derivatives_features'
    )
