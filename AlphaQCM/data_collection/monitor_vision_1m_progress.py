"""
监控 Binance Vision 1m klines（落盘到 crypto_1min_vision）下载进度。

用法（在 AlphaQCM/ 目录下）：
  python3 data_collection/monitor_vision_1m_progress.py --start 2020-01-01 --end 2025-02-15
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _default_symbols_file() -> Path:
    return Path(__file__).resolve().parent / "top100_perp_symbols.txt"


def _load_ccxt_symbols(path: Path) -> list[str]:
    syms: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        syms.append(s)
    return syms


def _ccxt_to_filename_symbol(ccxt_symbol: str) -> str:
    return ccxt_symbol.replace("/", "_")


def _read_last_ts(csv_path: Path) -> Optional[pd.Timestamp]:
    try:
        with open(csv_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - 16384, 0), os.SEEK_SET)
            for raw in reversed(f.read().splitlines()):
                if not raw.strip():
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                ts_text = (line.split(",", 1)[0] or "").strip().strip('"')
                ts = pd.to_datetime(ts_text, utc=True, errors="coerce")
                if ts is not None and pd.notna(ts):
                    return ts
                break
    except Exception:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="监控 1m klines 下载进度（crypto_1min_vision）")
    ap.add_argument("--symbols-file", default=str(_default_symbols_file()))
    ap.add_argument("--data-dir", default="AlphaQCM_data/crypto_1min_vision")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-02-15")
    args = ap.parse_args()

    symbols = _load_ccxt_symbols(Path(args.symbols_file))
    data_dir = Path(args.data_dir)

    start_ts = pd.Timestamp(_parse_date(args.start), tz="UTC")
    end_ts = pd.Timestamp(_parse_date(args.end), tz="UTC") + pd.Timedelta(hours=23, minutes=59)

    rows = []
    for s in symbols:
        fp = data_dir / f"{_ccxt_to_filename_symbol(s)}_1m.csv"
        if not fp.exists():
            rows.append(
                {
                    "symbol": s,
                    "exists": False,
                    "last_ts": None,
                    "done_to_end": False,
                    "gap_to_end_hours": None,
                    "file_gb": None,
                    "mtime_utc": None,
                }
            )
            continue

        last_ts = _read_last_ts(fp)
        done = bool(last_ts is not None and pd.notna(last_ts) and last_ts >= end_ts)
        gap_hours = None
        if last_ts is not None and pd.notna(last_ts) and last_ts < end_ts:
            gap_hours = int((end_ts - last_ts).total_seconds() // 3600)

        st = fp.stat()
        rows.append(
            {
                "symbol": s,
                "exists": True,
                "last_ts": last_ts.isoformat() if last_ts is not None and pd.notna(last_ts) else None,
                "done_to_end": done,
                "gap_to_end_hours": gap_hours,
                "file_gb": round(st.st_size / (1024**3), 3),
                "mtime_utc": pd.Timestamp(st.st_mtime, unit="s", tz="UTC").isoformat(),
            }
        )

    df = pd.DataFrame(rows)
    exists_cnt = int(df["exists"].sum())
    done_cnt = int(df["done_to_end"].sum())

    print(f"Symbols: {len(symbols)}")
    print(f"Files exist: {exists_cnt}/{len(symbols)}")
    print(f"Covered to end ({end_ts.isoformat()}): {done_cnt}/{len(symbols)}")
    print()

    not_done = df[df["exists"] & (~df["done_to_end"])].copy()
    not_done = not_done.sort_values(["gap_to_end_hours", "symbol"], ascending=[False, True])
    if len(not_done) > 0:
        print("Worst gaps (top 15):")
        show = not_done[["symbol", "last_ts", "gap_to_end_hours", "file_gb", "mtime_utc"]].head(15)
        print(show.to_string(index=False))
        print()

    missing = df[~df["exists"]].copy()
    if len(missing) > 0:
        print("Missing files:")
        print(", ".join(missing["symbol"].tolist()))
        print()

    out_path = Path("AlphaQCM_data/_tmp/vision_1m_progress.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

