"""
Data cleaning for hourly perpetual futures data (one symbol at a time)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

from gap_detector import detect_maintenance_flags

symbol = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]

alphaqcm_root = Path(__file__).resolve().parents[1]
input_dir = str((alphaqcm_root / input_dir).resolve())
output_dir = str((alphaqcm_root / output_dir).resolve())

input_file = f"{input_dir}/{symbol}_hourly_full.csv"
df = pd.read_csv(input_file, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)

# 1. Missing value handling
df['open'] = df['open'].ffill()
df['high'] = df['high'].ffill()
df['low'] = df['low'].ffill()
df['close'] = df['close'].ffill()
df['vwap'] = df['vwap'].ffill()
df['volume'] = df['volume'].fillna(0)
if 'funding_rate' in df.columns:
    df['funding_rate'] = df['funding_rate'].ffill()
    df['funding_annualized'] = df['funding_annualized'].ffill()

# 2. Maintenance gap detection
flags = detect_maintenance_flags(df.index, max_gap_seconds=3605, cooldown_hours=2)
df = df.join(flags, how='left')
df['is_stable'] = (df['cooldown_no_trade'] == 0)

# 3. Outlier detection
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['is_spike'] = df['log_return'].abs() > 0.5

vol_mean = df['volume'].rolling(168, min_periods=24).mean()
vol_std = df['volume'].rolling(168, min_periods=24).std()
df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)
df['is_volume_spike'] = df['volume_zscore'].abs() > 5

vol_99 = df['volume'].quantile(0.99)
df['volume_clean'] = df['volume'].clip(upper=vol_99)

# 4. New listing filter
df['is_mature'] = (pd.Series(range(len(df)), index=df.index) >= 24)

# 5. Point-in-time check
expected_freq = pd.date_range(df.index.min(), df.index.max(), freq='1h')
missing_hours = len(expected_freq) - len(df)

# Drop temp columns
df = df.drop(columns=['log_return', 'volume_zscore'])

# Save
output_file = f'{output_dir}/{symbol}_cleaned.csv'
df.to_csv(output_file)

print(f"{len(df)},{df['is_stable'].sum()},{df['is_spike'].sum()},{df['is_volume_spike'].sum()},{df['is_mature'].sum()},{missing_hours}")
