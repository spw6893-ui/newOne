"""
Aggregate 1-minute data to hourly with microstructure features
"""
import pandas as pd
import numpy as np
import os
import glob

def aggregate_to_hourly(
    input_dir='AlphaQCM_data/crypto_1min',
    output_dir='AlphaQCM_data/crypto_hourly_micro',
    funding_dir='AlphaQCM_data/crypto_derivatives'
):
    """
    Aggregate 1-minute data to hourly with microstructure features

    Features:
    1. OHLCV (from 1-min aggregation)
    2. VWAP deviation: (close - vwap) / vwap
    3. Realized volatility: std of 60 1-min returns
    4. Taker buy ratio: Not available from OHLCV, requires trade data
    5. Volume distribution: std/mean of 1-min volumes
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, '*_1m.csv'))

    for csv_file in csv_files:
        symbol = os.path.basename(csv_file).replace('_1m.csv', '')
        print(f"Aggregating {symbol}...")

        try:
            # Load 1-minute data
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

            # Calculate 1-minute returns for realized volatility
            df['returns'] = df['close'].pct_change()

            # Calculate typical price for VWAP
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['vwap_numerator'] = df['typical_price'] * df['volume']

            # Resample to hourly
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'vwap_numerator': 'sum',
                'returns': lambda x: x.std() * np.sqrt(60)  # Realized volatility (annualized to hourly)
            }

            df_hourly = df.resample('1H').agg(agg_dict)

            # Calculate VWAP from aggregated data
            df_hourly['vwap'] = df_hourly['vwap_numerator'] / df_hourly['volume']

            # VWAP deviation
            df_hourly['vwap_dev'] = (df_hourly['close'] - df_hourly['vwap']) / df_hourly['vwap']

            # Realized volatility (already calculated in aggregation)
            df_hourly['realized_vol'] = df_hourly['returns']

            # Volume distribution (coefficient of variation)
            df_hourly['volume_cv'] = df.resample('1H')['volume'].apply(
                lambda x: x.std() / x.mean() if x.mean() > 0 else 0
            )

            # Drop temporary columns
            df_hourly = df_hourly.drop(columns=['vwap_numerator', 'returns'])

            # Load and merge funding rate data (time-aligned)
            funding_file = os.path.join(funding_dir, f"{symbol}_funding_rate.csv")
            if os.path.exists(funding_file):
                df_funding = pd.read_csv(funding_file, index_col=0, parse_dates=True)
                # Forward fill funding rate to hourly (funding rate updates every 8 hours)
                df_funding = df_funding.reindex(df_hourly.index, method='ffill')
                df_hourly = df_hourly.join(df_funding, how='left')
                print(f"  Merged funding rate data")

            # Drop NaN rows
            df_hourly = df_hourly.dropna()

            # Save aggregated data
            output_file = os.path.join(output_dir, f'{symbol}_hourly_micro.csv')
            df_hourly.to_csv(output_file)
            print(f"  Saved {len(df_hourly)} rows to {output_file}")

        except Exception as e:
            print(f"  Error processing {symbol}: {e}")

    print(f"\nAggregated data saved to {output_dir}")

if __name__ == '__main__':
    aggregate_to_hourly(
        input_dir='AlphaQCM_data/crypto_1min',
        output_dir='AlphaQCM_data/crypto_hourly_micro',
        funding_dir='AlphaQCM_data/crypto_derivatives'
    )
