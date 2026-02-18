"""
Data cleaning for hourly perpetual futures data
Implements 5 key cleaning steps (excluding normalization)
"""
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

from gap_detector import detect_maintenance_flags

def clean_symbol(
    symbol,
    input_dir,
    output_dir,
    *,
    input_suffix="_hourly_full.csv",
    output_suffix="_cleaned.csv",
    min_history_bars=24,
    skip_existing=False,
):
    """
    Clean single symbol data

    Steps:
    1. Missing value handling (ffill for prices, 0 for volume)
    2. Maintenance gap detection and flagging
    3. Outlier detection (price spikes, volume spikes)
    4. New listing filtering (min_history_bars)
    5. Point-in-time alignment check
    """

    output_file = os.path.join(output_dir, f"{symbol}{output_suffix}")
    if skip_existing and os.path.exists(output_file):
        return None

    input_file = os.path.join(input_dir, f"{symbol}{input_suffix}")
    if not os.path.exists(input_file):
        return None

    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    # 1. Missing value handling
    # Prices: forward fill
    price_cols = ['open', 'high', 'low', 'close', 'vwap']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Volume: fill with 0
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0)

    # OI and Funding: forward fill
    if 'funding_rate' in df.columns:
        df['funding_rate'] = df['funding_rate'].ffill()
        df['funding_annualized'] = df['funding_annualized'].ffill()

    # 2. Maintenance gap detection
    # 防御性编程：用“数据断流”替代“公告/维护时间表”
    # - gap > 3605s 视为维护/断流
    # - 恢复后的 2 小时（含恢复当下）强制禁止交易，避免撮合重启插针风险
    flags = detect_maintenance_flags(df.index, max_gap_seconds=3605, cooldown_hours=2)
    df = df.join(flags, how='left')
    df['is_stable'] = (df['cooldown_no_trade'] == 0)

    # 3. Outlier detection
    # Price spike detection (>50% move)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['is_spike'] = df['log_return'].abs() > 0.5

    # Volume spike detection (Z-score > 5)
    vol_mean = df['volume'].rolling(168, min_periods=24).mean()
    vol_std = df['volume'].rolling(168, min_periods=24).std()
    df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)
    df['is_volume_spike'] = df['volume_zscore'].abs() > 5

    # Winsorize volume at 99th percentile
    vol_99 = df['volume'].quantile(0.99)
    df['volume_clean'] = df['volume'].clip(upper=vol_99)

    # 4. New listing filter
    df['bars_since_start'] = range(len(df))
    df['is_mature'] = df['bars_since_start'] >= min_history_bars

    # 5. Point-in-time check (timestamp continuity)
    expected_freq = pd.date_range(df.index.min(), df.index.max(), freq='1h')
    missing_hours = len(expected_freq) - len(df)

    # Drop temporary columns
    df = df.drop(columns=['log_return', 'volume_zscore', 'bars_since_start'])

    # Save cleaned data
    df.to_csv(output_file)

    return {
        'symbol': symbol,
        'rows': len(df),
        'stable_rows': df['is_stable'].sum(),
        'spikes': df['is_spike'].sum(),
        'volume_spikes': df['is_volume_spike'].sum(),
        'mature_rows': df['is_mature'].sum(),
        'missing_hours': missing_hours
    }

def main():
    parser = argparse.ArgumentParser(description="按小时数据清洗（维护断流标记 / 异常值 / 新币成熟期）")
    parser.add_argument("--input-dir", default="AlphaQCM_data/crypto_hourly_full")
    parser.add_argument("--output-dir", default="AlphaQCM_data/crypto_hourly_cleaned")
    parser.add_argument("--input-suffix", default="_hourly_full.csv")
    parser.add_argument("--output-suffix", default="_cleaned.csv")
    parser.add_argument("--min-history-bars", type=int, default=24)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    alphaqcm_root = Path(__file__).resolve().parents[1]
    input_dir = str((alphaqcm_root / args.input_dir).resolve()) if not os.path.isabs(args.input_dir) else args.input_dir
    output_dir = str((alphaqcm_root / args.output_dir).resolve()) if not os.path.isabs(args.output_dir) else args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, f"*{args.input_suffix}"))
    if not csv_files:
        print(f"未找到输入文件：{input_dir}")
        return

    results = []
    for i, csv_file in enumerate(csv_files, 1):
        symbol = os.path.basename(csv_file).replace(args.input_suffix, '')
        print(f"[{i}/{len(csv_files)}] {symbol}...", end=' ')

        try:
            result = clean_symbol(
                symbol,
                input_dir,
                output_dir,
                input_suffix=args.input_suffix,
                output_suffix=args.output_suffix,
                min_history_bars=args.min_history_bars,
                skip_existing=args.skip_existing,
            )
            if result:
                results.append(result)
                print(f"✓ {result['rows']} rows, {result['stable_rows']} stable, {result['spikes']} spikes")
            else:
                print("↷ skipped/no data")
        except Exception as e:
            print(f"✗ {e}")

    # Summary
    df_summary = pd.DataFrame(results)
    summary_file = os.path.join(output_dir, 'cleaning_summary.csv')
    df_summary.to_csv(summary_file, index=False)

    print(f"\nCleaned {len(results)}/{len(csv_files)} symbols (written)")
    if not df_summary.empty:
        print(f"Total spikes detected: {df_summary['spikes'].sum()}")
        print(f"Total volume spikes: {df_summary['volume_spikes'].sum()}")
        print(f"Total missing hours: {df_summary['missing_hours'].sum()}")
    print(f"Summary: {summary_file}")

if __name__ == '__main__':
    main()
