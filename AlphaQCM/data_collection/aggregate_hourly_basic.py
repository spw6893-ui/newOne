"""
Aggregate 1-minute data to hourly (simple version, no derivatives yet)
Memory-efficient: processes one symbol at a time
"""
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-12


def _to_utc_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _iter_complete_chunks(
    csv_path: Path,
    *,
    chunksize: int,
    assume_sorted: bool,
):
    """
    流式读取 1m CSV，按“完整小时块”返回 DataFrame（最后一小时放入 buffer，延迟到下一 chunk）。
    该函数只做切块，不做聚合。
    """
    buffer = None
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if "datetime" not in chunk.columns:
            # 兼容早期格式：第一列为 datetime
            first = chunk.columns[0]
            chunk = chunk.rename(columns={first: "datetime"})

        chunk["datetime"] = _to_utc_datetime(chunk["datetime"])
        chunk = chunk.dropna(subset=["datetime"])
        if chunk.empty:
            continue

        if buffer is not None and not buffer.empty:
            chunk = pd.concat([buffer, chunk], ignore_index=True)

        if not assume_sorted:
            chunk = chunk.sort_values("datetime")

        hour = chunk["datetime"].dt.floor("h")
        last_hour = hour.iloc[-1]

        complete = chunk.loc[hour != last_hour].copy()
        buffer = chunk.loc[hour == last_hour].copy()

        if complete.empty:
            continue

        yield complete

    if buffer is not None and not buffer.empty:
        yield buffer


def aggregate_symbol(
    symbol: str,
    min1_dir: str,
    output_dir: str,
    *,
    chunksize: int,
    assume_sorted: bool,
    skip_existing: bool,
    force: bool,
) -> int | None:
    """
    以低内存方式聚合单币种 1m -> 1h（OHLCV + VWAP）。
    - 采用原子写入：先写 .tmp，成功后替换正式文件，避免中断留下半成品。
    """
    csv_path = Path(min1_dir) / f"{symbol}_1m.csv"
    if not csv_path.exists():
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{symbol}_hourly.csv"
    tmp_path = out_dir / f"{symbol}_hourly.csv.tmp"

    if output_path.exists() and skip_existing and not force:
        return 0

    if tmp_path.exists():
        tmp_path.unlink()

    written = 0
    for complete in _iter_complete_chunks(csv_path, chunksize=chunksize, assume_sorted=assume_sorted):
        for c in ("open", "high", "low", "close", "volume"):
            if c in complete.columns:
                complete[c] = pd.to_numeric(complete[c], errors="coerce")

        tp = (complete["high"] + complete["low"] + complete["close"]) / 3.0
        complete["vwap_num"] = tp * complete["volume"]
        complete["__hour"] = complete["datetime"].dt.floor("h")

        hourly = complete.groupby("__hour", sort=True).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            vwap_num=("vwap_num", "sum"),
        )
        hourly.index = pd.DatetimeIndex(hourly.index, name="datetime")
        hourly["vwap"] = hourly["vwap_num"] / np.maximum(hourly["volume"], EPS)
        hourly = hourly.drop(columns=["vwap_num"]).dropna()

        mode = "w" if written == 0 else "a"
        header = written == 0
        hourly.to_csv(tmp_path, mode=mode, header=header)
        written += int(hourly.shape[0])

    tmp_path.replace(output_path)
    return written

def main():
    ap = argparse.ArgumentParser(description="从 1m OHLCV 聚合生成 1h OHLCV+VWAP（基础版本）")
    ap.add_argument("--min1-dir", default="AlphaQCM_data/crypto_1min", help="输入目录（1m CSV）")
    ap.add_argument("--output-dir", default="AlphaQCM_data/crypto_hourly_basic", help="输出目录（1h CSV）")
    ap.add_argument("--chunksize", type=int, default=200_000, help="流式读取 chunksize，默认 200k")
    ap.add_argument("--assume-sorted", action="store_true", help="假设输入已按时间升序（可跳过排序，加速）")
    ap.add_argument("--skip-existing", action="store_true", help="若输出已存在则跳过（支持断点续跑）")
    ap.add_argument("--force", action="store_true", help="强制重算（覆盖已存在输出）")
    ap.add_argument("--symbols", default="", help="仅处理指定币种（逗号分隔），留空表示处理目录内所有文件")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个文件（0=不限制）")
    args = ap.parse_args()

    min1_dir = args.min1_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        csv_files = [os.path.join(min1_dir, f"{s}_1m.csv") for s in symbols]
    else:
        csv_files = glob.glob(os.path.join(min1_dir, "*_1m.csv"))
        csv_files = sorted(csv_files)
    if args.limit and args.limit > 0:
        csv_files = csv_files[: int(args.limit)]

    results = []
    for i, csv_file in enumerate(csv_files, 1):
        symbol = os.path.basename(csv_file).replace('_1m.csv', '')
        print(f"[{i}/{len(csv_files)}] {symbol}...", end=' ')

        try:
            rows = aggregate_symbol(
                symbol,
                min1_dir,
                output_dir,
                chunksize=int(args.chunksize),
                assume_sorted=bool(args.assume_sorted),
                skip_existing=bool(args.skip_existing),
                force=bool(args.force),
            )
            if rows:
                results.append({'symbol': symbol, 'rows': rows})
                print(f"✓ {rows}")
            elif rows == 0 and args.skip_existing and not args.force:
                print("↷ Skip (exists)")
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
