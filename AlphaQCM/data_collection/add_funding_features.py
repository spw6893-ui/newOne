"""
Add derivatives features (funding rate) to hourly data
"""
import pandas as pd
import os
import glob

def add_funding_rate(symbol, hourly_dir, funding_dir, output_dir):
    """Add funding rate to hourly data"""

    hourly_file = os.path.join(hourly_dir, f"{symbol}_hourly.csv")
    if not os.path.exists(hourly_file):
        return None

    df = pd.read_csv(hourly_file, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    # Load funding rate
    funding_file = os.path.join(funding_dir, f"{symbol}_funding_rate.csv")
    if os.path.exists(funding_file):
        df_funding = pd.read_csv(funding_file, index_col=0)
        df_funding.index = pd.to_datetime(df_funding.index, format='mixed', utc=True)

        # Forward fill to hourly
        df_funding = df_funding.reindex(df.index, method='ffill')

        # Add funding features
        df['funding_rate'] = df_funding['funding_rate']
        df['funding_annualized'] = df['funding_rate'] * 365 * 3
        df['funding_delta'] = df['funding_rate'].diff()
        df['arb_pressure'] = (df['funding_annualized'].abs() > 0.30).astype(int)

    # Drop NaN
    df = df.dropna()

    # Save
    output_file = os.path.join(output_dir, f'{symbol}_hourly_full.csv')
    df.to_csv(output_file)

    return len(df)

def main():
    hourly_dir = 'AlphaQCM_data/crypto_hourly_basic'
    funding_dir = 'AlphaQCM_data/crypto_derivatives'
    output_dir = 'AlphaQCM_data/crypto_hourly_full'

    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(hourly_dir, '*_hourly.csv'))

    results = []
    for i, csv_file in enumerate(csv_files, 1):
        symbol = os.path.basename(csv_file).replace('_hourly.csv', '')
        print(f"[{i}/{len(csv_files)}] {symbol}...", end=' ')

        try:
            rows = add_funding_rate(symbol, hourly_dir, funding_dir, output_dir)
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
