"""
监控 Binance Vision metrics（OI / 多空比等）下载补齐进度。

用法（在 AlphaQCM/ 目录下）：
  python3 data_collection/monitor_metrics_progress.py --start 2020-01-01 --end 2025-02-15
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _default_symbols_file() -> Path:
    return Path(__file__).resolve().parent / "top100_perp_symbols.txt"


def _load_symbols(path: Path) -> list[str]:
    syms: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        s = s.replace("/USDT:USDT", "USDT")
        s = s.replace("/USDT", "USDT")
        s = s.replace("_", "")
        syms.append(s)
    return syms


def _read_first_ts(csv_path: Path) -> Optional[pd.Timestamp]:
    try:
        with open(csv_path, "rb") as f:
            head = f.read(16384).splitlines()
        for raw in head:
            if not raw.strip():
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            ts_text = (line.split(",", 1)[0] or "").strip().strip('"')
            ts = pd.to_datetime(ts_text, utc=True, errors="coerce")
            if ts is not None and pd.notna(ts):
                return ts
    except Exception:
        return None
    return None


def _read_last_ts(csv_path: Path) -> Optional[pd.Timestamp]:
    try:
        with open(csv_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(size - 16384, 0), 0)
            lines = f.read().splitlines()
        for raw in reversed(lines):
            if not raw.strip():
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            ts_text = (line.split(",", 1)[0] or "").strip().strip('"')
            ts = pd.to_datetime(ts_text, utc=True, errors="coerce")
            if ts is not None and pd.notna(ts):
                return ts
    except Exception:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="监控 metrics 下载进度")
    parser.add_argument("--symbols-file", default=str(_default_symbols_file()))
    parser.add_argument("--data-dir", default="AlphaQCM_data/binance_metrics")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-02-15")
    args = parser.parse_args()

    symbols = _load_symbols(Path(args.symbols_file))
    data_dir = Path(args.data_dir)
    start_ts = pd.Timestamp(_parse_date(args.start), tz="UTC")
    # metrics 是 daily 归档：用“覆盖到结束日期”判断即可（不少合约只有某个固定小时的快照）
    end_ts = pd.Timestamp(_parse_date(args.end), tz="UTC")

    rows = []
    for s in symbols:
        p = data_dir / f"{s}_metrics.csv"
        if not p.exists():
            rows.append(
                {
                    "symbol": s,
                    "exists": False,
                    "first_ts": None,
                    "last_ts": None,
                    "done_to_end": False,
                    "gap_to_end_hours": None,
                    "file_mb": None,
                    "mtime_utc": None,
                }
            )
            continue

        first_ts = _read_first_ts(p)
        last_ts = _read_last_ts(p)
        done_to_end = bool(last_ts is not None and pd.notna(last_ts) and last_ts >= end_ts)
        gap_hours = None
        if last_ts is not None and pd.notna(last_ts) and last_ts < end_ts:
            gap_hours = int((end_ts - last_ts).total_seconds() // 3600)

        st = p.stat()
        rows.append(
            {
                "symbol": s,
                "exists": True,
                "first_ts": first_ts.isoformat() if first_ts is not None and pd.notna(first_ts) else None,
                "last_ts": last_ts.isoformat() if last_ts is not None and pd.notna(last_ts) else None,
                "done_to_end": done_to_end,
                "gap_to_end_hours": gap_hours,
                "file_mb": round(st.st_size / (1024 * 1024), 2),
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
        show = not_done[["symbol", "first_ts", "last_ts", "gap_to_end_hours", "file_mb", "mtime_utc"]].head(15)
        print(show.to_string(index=False))
        print()

    missing = df[~df["exists"]].copy()
    if len(missing) > 0:
        print("Missing files:")
        print(", ".join(missing["symbol"].tolist()))
        print()

    # 输出到 tmp，便于你做二次监控/可视化
    out_path = Path("AlphaQCM_data/_tmp/metrics_progress.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
