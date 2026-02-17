"""
Calculate volatility and other derived indicators from OHLCV data
"""
import pandas as pd
import numpy as np
import os
import glob

def calculate_volatility_indicators(
    data_dir='AlphaQCM_data/crypto_data',
    output_dir='AlphaQCM_data/crypto_indicators',
    windows=[24, 168]  # 1 day, 1 week for hourly data
):
    """
    Calculate volatility and derived indicators from OHLCV data

    Indicators:
    - Realized Volatility (RV): Standard deviation of returns
    - Parkinson Volatility: Uses high-low range
    - Garman-Klass Volatility: More efficient estimator
    - ATR (Average True Range): Volatility measure
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(data_dir, '*_1h.csv'))

    for csv_file in csv_files:
        symbol = os.path.basename(csv_file).replace('_1h.csv', '')
        print(f"Calculating indicators for {symbol}...")

        try:
            # Load OHLCV data
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

            # Calculate returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Calculate volatility indicators for each window
            for window in windows:
                # 1. Realized Volatility (standard deviation of returns)
                df[f'rv_{window}h'] = df['returns'].rolling(window=window).std() * np.sqrt(window)

                # 2. Parkinson Volatility (high-low estimator)
                df[f'parkinson_vol_{window}h'] = np.sqrt(
                    (1 / (4 * np.log(2))) *
                    ((np.log(df['high'] / df['low'])) ** 2).rolling(window=window).mean()
                ) * np.sqrt(window)

                # 3. Garman-Klass Volatility (more efficient)
                hl = np.log(df['high'] / df['low']) ** 2
                co = np.log(df['close'] / df['open']) ** 2
                df[f'gk_vol_{window}h'] = np.sqrt(
                    (0.5 * hl - (2 * np.log(2) - 1) * co).rolling(window=window).mean()
                ) * np.sqrt(window)

            # 4. ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            df['atr_24h'] = true_range.rolling(window=24).mean()
            df['atr_168h'] = true_range.rolling(window=168).mean()

            # 5. Volume-weighted volatility
            df['volume_weighted_vol_24h'] = (
                (df['returns'] ** 2 * df['volume']).rolling(window=24).sum() /
                df['volume'].rolling(window=24).sum()
            ).apply(np.sqrt) * np.sqrt(24)

            # 6. Bollinger Band width (normalized volatility)
            for window in windows:
                sma = df['close'].rolling(window=window).mean()
                std = df['close'].rolling(window=window).std()
                df[f'bb_width_{window}h'] = (std * 2) / sma

            # Drop NaN rows
            df_indicators = df.dropna()

            # Save indicators
            output_file = os.path.join(output_dir, f'{symbol}_indicators.csv')
            df_indicators.to_csv(output_file)
            print(f"  Saved {len(df_indicators)} rows to {output_file}")

        except Exception as e:
            print(f"  Error processing {symbol}: {e}")

    print(f"\nIndicators saved to {output_dir}")

if __name__ == '__main__':
    calculate_volatility_indicators(
        data_dir='AlphaQCM_data/crypto_data',
        output_dir='AlphaQCM_data/crypto_indicators',
        windows=[24, 168]  # 1 day, 1 week
    )
