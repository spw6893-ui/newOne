"""
从 Binance Vision 下载 U 本位永续 `klines` 的 1 分钟（1m）历史数据，并落盘为 1m CSV（兼容旧管线）。

为什么要做这个：
- 之前的 1m 数据来自 CCXT，容易中途异常导致某些币种只覆盖到 2020（如 ETH）。
- Binance Vision 提供更稳定的归档源（建议用月度归档），可全量补齐到指定区间。

数据源（Vision 月度归档，避免按日下载请求量爆炸）：
  https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-YYYY-MM.zip

输出（按 CCXT 符号命名，兼容 `crypto_1min`）：
  AlphaQCM_data/crypto_1min_vision/{CCXT_SYMBOL}_1m.csv
  - index: datetime(UTC)
  - columns: open, high, low, close, volume

断点续跑：
- 支持 `--skip-existing`：若目标 CSV 已覆盖到 end_date 23:59 UTC 则跳过；
- 否则从最后时间戳 + 1 分钟继续补齐。

用法（在 AlphaQCM/ 目录下）：
  python3 data_collection/download_binance_vision_klines_1m.py --start 2020-01-01 --end 2025-02-15 --skip-existing
  python3 data_collection/download_binance_vision_klines_1m.py --symbols-offset 0 --symbols-limit 30
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from download_binance_efficient import (
    DownloadConfig,
    _find_first_available_month,
    download_month_zip_to_csv,
    _maybe_rewrite_alphaqcm_data_path,
    _head_url_exists,
    _build_monthly_kline_zip_url,
)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _default_symbols_file() -> str:
    return str(Path(__file__).resolve().parent / "top100_perp_symbols.txt")


def _load_ccxt_symbols(path: str) -> list[str]:
    symbols: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            symbols.append(s)
    return symbols


def _ccxt_to_binance_um_symbol(ccxt_symbol: str) -> str:
    # 例：BTC/USDT:USDT -> BTCUSDT
    s = str(ccxt_symbol).replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace("/", "").replace("_", "")
    return s


def _ccxt_to_filename_symbol(ccxt_symbol: str) -> str:
    # 与旧 `fetch_1min_data.py` 保持一致：仅把 / 替换为 _
    return ccxt_symbol.replace("/", "_")


def _read_csv_last_timestamp(csv_path: Path) -> Optional[pd.Timestamp]:
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


def _parse_kline_csv_1m(csv_path: Path) -> pd.DataFrame:
    """
    解析 Vision 的 klines CSV（1m），兼容：
    - 带表头
    - 不带表头（纯数值行）
    """
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = (f.readline() or "").strip()

    if first_line.lower().startswith("open_time"):
        df = pd.read_csv(csv_path)
    else:
        names = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df = pd.read_csv(csv_path, header=None, names=names)

    # Vision 的 open_time 通常是毫秒时间戳；不能直接按默认 ns 解析
    s = df["open_time"]
    if pd.api.types.is_numeric_dtype(s):
        df["open_time"] = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
    else:
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().mean() > 0.98:
            df["open_time"] = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        else:
            df["open_time"] = pd.to_datetime(s, utc=True, errors="coerce")
    df = df.dropna(subset=["open_time"]).sort_values("open_time")
    df = df.drop_duplicates(subset=["open_time"], keep="last").set_index("open_time")

    out = pd.DataFrame(index=df.index)
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(df.get(c), errors="coerce")

    out.index.name = "datetime"
    return out


def _iter_month_starts(start: datetime, end: datetime) -> list[datetime]:
    months: list[datetime] = []
    m = datetime(start.year, start.month, 1)
    last = datetime(end.year, end.month, 1)
    while m <= last:
        months.append(m)
        m = (m.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def download_symbol_1m(
    *,
    ccxt_symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: str,
    temp_dir: str,
    cfg: DownloadConfig,
    skip_existing: bool,
    session: requests.Session,
) -> bool:
    """
    返回 True：成功写入/更新了目标 1m CSV（哪怕只补了一部分）
    """
    output_dir = _maybe_rewrite_alphaqcm_data_path(output_dir)
    temp_dir = _maybe_rewrite_alphaqcm_data_path(temp_dir)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname_symbol = _ccxt_to_filename_symbol(ccxt_symbol)
    out_csv = out_dir / f"{fname_symbol}_1m.csv"
    meta_path = out_csv.with_suffix(out_csv.suffix + ".meta.json")

    expected_end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    resume_from = pd.Timestamp(start_date, tz="UTC")

    if out_csv.exists():
        last_ts = _read_csv_last_timestamp(out_csv)
        if last_ts is not None and pd.notna(last_ts):
            if skip_existing and last_ts >= expected_end:
                # 覆盖到结束：直接跳过
                return True
            resume_from = max(resume_from, last_ts + pd.Timedelta(minutes=1))

    if resume_from > expected_end:
        return True

    binance_symbol = _ccxt_to_binance_um_symbol(ccxt_symbol)

    # 快速跳过起始处连续 404 的月份（新币常见）
    first_month = _find_first_available_month(
        session=session,
        symbol=binance_symbol,
        start_date=resume_from.to_pydatetime().replace(tzinfo=None),
        end_date=end_date,
        data_type="klines",
        interval="1m",
        cfg=cfg,
    )
    if first_month is None:
        return False

    wrote_any = False
    months = _iter_month_starts(first_month, end_date)
    for m in months:
        month_str = m.strftime("%Y-%m")
        month_end = (pd.Timestamp(m, tz="UTC") + pd.offsets.MonthEnd(1)).normalize()
        month_max_ts = month_end + pd.Timedelta(hours=23, minutes=59)

        # 若本月全部早于 resume_from，跳过
        if month_max_ts < resume_from:
            continue

        # 对于明显不存在的月份，提前跳过（避免不必要下载）
        url = _build_monthly_kline_zip_url(
            symbol=binance_symbol,
            month=m,
            kline_type="klines",
            interval="1m",
            market=cfg.market,
        )
        exists = _head_url_exists(session, url, cfg)
        if exists is False:
            continue

        print(f"    {month_str} ...", end=" ")
        csv_path = download_month_zip_to_csv(
            symbol=binance_symbol,
            month=m,
            data_type="klines",
            interval="1m",
            temp_dir=temp_dir,
            cfg=cfg,
            session=session,
        )
        if csv_path is None:
            print("✗")
            continue

        try:
            df_1m = _parse_kline_csv_1m(csv_path)
        finally:
            try:
                csv_path.unlink(missing_ok=True)
            except TypeError:
                if csv_path.exists():
                    csv_path.unlink()

        # 裁剪到目标区间（resume_from..expected_end）
        df_1m = df_1m.loc[(df_1m.index >= resume_from) & (df_1m.index <= expected_end)]
        if len(df_1m) == 0:
            print("∅")
            continue

        # 追加写盘（按月 append，避免一次性大内存）
        mode = "a" if out_csv.exists() else "w"
        header = not out_csv.exists()
        df_1m.to_csv(out_csv, mode=mode, header=header)
        wrote_any = True
        print(f"✓ {len(df_1m)}")

        # 更新 resume_from（本月内可能已覆盖到末尾）
        resume_from = df_1m.index.max() + pd.Timedelta(minutes=1)
        if resume_from > expected_end:
            break

    # 写 meta（尽量基于末行时间戳，避免全量扫描）
    if out_csv.exists():
        last_ts = _read_csv_last_timestamp(out_csv)
        meta = {
            "status": "ok" if wrote_any else "noop",
            "ccxt_symbol": ccxt_symbol,
            "binance_symbol": binance_symbol,
            "requested_start": start_date.strftime("%Y-%m-%d"),
            "requested_end": end_date.strftime("%Y-%m-%d"),
            "max_ts": (last_ts.isoformat() if last_ts is not None and pd.notna(last_ts) else None),
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return wrote_any or out_csv.exists()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="从 Binance Vision 下载 1m klines 并落盘为 1m CSV（兼容旧管线）")
    ap.add_argument("--symbols-file", default=_default_symbols_file())
    ap.add_argument("--symbols-offset", type=int, default=0)
    ap.add_argument("--symbols-limit", type=int, default=0)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-02-15")
    ap.add_argument("--output-dir", default="AlphaQCM_data/crypto_1min_vision")
    ap.add_argument("--temp-dir", default="AlphaQCM_data/_tmp/binance_vision_klines_1m")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)

    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)

    symbols = _load_ccxt_symbols(args.symbols_file)
    if args.symbols_offset:
        symbols = symbols[int(args.symbols_offset) :]
    if args.symbols_limit:
        symbols = symbols[: int(args.symbols_limit)]

    cfg = DownloadConfig(market="um")

    print(f"Symbols: {len(symbols)}")
    print(f"Range: {start_date.date()} ~ {end_date.date()}")
    print(f"Output: {args.output_dir}")
    print(f"Temp: {args.temp_dir}")
    print()

    ok_cnt = 0
    with requests.Session() as session:
        for i, s in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] {s}")
            try:
                ok = download_symbol_1m(
                    ccxt_symbol=s,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=args.output_dir,
                    temp_dir=args.temp_dir,
                    cfg=cfg,
                    skip_existing=bool(args.skip_existing),
                    session=session,
                )
                if ok:
                    ok_cnt += 1
                    print("  ✓ done")
                else:
                    print("  ✗ (no data)")
            except Exception as e:
                print(f"  ✗ ({e})")

    print(f"\nCompleted: {ok_cnt}/{len(symbols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
