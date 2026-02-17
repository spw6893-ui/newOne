"""
Aggregate 1-minute data to hourly (simple version, no derivatives yet)
Memory-efficient: processes one symbol at a time
"""
import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def process_symbol(symbol, min1_dir, output_dir):
    """Process single symbol (memory efficient)"""

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

    # Drop NaN
    hourly = hourly.dropna()

    # Save
    output_file = os.path.join(output_dir, f'{symbol}_hourly.csv')
    hourly.to_csv(output_file)

    return len(hourly)

def main():
    min1_dir = 'AlphaQCM_data/crypto_1min'
    output_dir = 'AlphaQCM_data/crypto_hourly_basic'

    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(min1_dir, '*_1m.csv'))

    results = []
    for i, csv_file in enumerate(csv_files, 1):
        symbol = os.path.basename(csv_file).replace('_1m.csv', '')
        print(f"[{i}/{len(csv_files)}] {symbol}...", end=' ')

        try:
            rows = process_symbol(symbol, min1_dir, output_dir)
            if rows:
                results.append({'symbol': symbol, 'rows': rows})
                print(f"✓ {rows}")
            else:
                print("✗ No data")
        except Exception as e:
            print(f"✗ {e}")

    # Summary
    df_summary = pd.DataFrame(results)
    summary_file = os.path.join(output_dir, 'summary.csv')
    df_summary.to_csv(summary_file, index=False)
    print(f"\nProcessed {len(results)}/{len(csv_files)} symbols")
    print(f"Total size: ", end='')
    os.system(f"du -sh {output_dir}")

if __name__ == '__main__':
    main()
