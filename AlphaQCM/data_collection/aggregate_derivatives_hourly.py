"""
Aggregate derivatives data to hourly with contract-specific features
Memory-efficient: processes one symbol at a time
"""
import pandas as pd
import numpy as np
import os
import glob

def process_symbol(symbol, min1_dir, funding_dir, output_dir):
    """Process single symbol (memory efficient)"""

    # Load 1-minute data
    min1_file = os.path.join(min1_dir, f"{symbol}_1m.csv")
    if not os.path.exists(min1_file):
        return None

    df = pd.read_csv(min1_file, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    # Calculate typical price and VWAP components
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap_num'] = df['typical_price'] * df['volume']

    # Aggregate to hourly
    hourly = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'vwap_num': 'sum'
    })

    # VWAP
    hourly['vwap'] = hourly['vwap_num'] / hourly['volume']
    hourly = hourly.drop(columns=['vwap_num'])

    # Load funding rate
    funding_file = os.path.join(funding_dir, f"{symbol}_funding_rate.csv")
    if os.path.exists(funding_file):
        df_funding = pd.read_csv(funding_file, index_col=0, parse_dates=True)
        df_funding.index = pd.to_datetime(df_funding.index, utc=True)

        # Forward fill to hourly
        df_funding_hourly = df_funding.reindex(hourly.index, method='ffill')

        # Funding rate features
        hourly['funding_rate'] = df_funding_hourly['funding_rate']
        hourly['funding_annualized'] = hourly['funding_rate'] * 365 * 3  # 8h intervals
        hourly['funding_delta'] = hourly['funding_rate'].diff()

        # Arbitrage pressure (funding > 30% annualized)
        hourly['arb_pressure'] = (hourly['funding_annualized'].abs() > 0.30).astype(int)

    # Drop NaN
    hourly = hourly.dropna()

    # Save
    output_file = os.path.join(output_dir, f'{symbol}_hourly.csv')
    hourly.to_csv(output_file)

    return len(hourly)

def main():
    min1_dir = 'AlphaQCM_data/crypto_1min'
    funding_dir = 'AlphaQCM_data/crypto_derivatives'
    output_dir = 'AlphaQCM_data/crypto_hourly_derivatives'

    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(min1_dir, '*_1m.csv'))

    results = []
    for i, csv_file in enumerate(csv_files, 1):
        symbol = os.path.basename(csv_file).replace('_1m.csv', '')
        print(f"[{i}/{len(csv_files)}] {symbol}...", end=' ')

        try:
            rows = process_symbol(symbol, min1_dir, funding_dir, output_dir)
            if rows:
                results.append({'symbol': symbol, 'rows': rows})
                print(f"✓ {rows} rows")
            else:
                print("✗ No data")
        except Exception as e:
            print(f"✗ {e}")

    # Summary
    df_summary = pd.DataFrame(results)
    summary_file = os.path.join(output_dir, 'summary.csv')
    df_summary.to_csv(summary_file, index=False)
    print(f"\nProcessed {len(results)}/{len(csv_files)} symbols")
    print(f"Summary: {summary_file}")

if __name__ == '__main__':
    main()
