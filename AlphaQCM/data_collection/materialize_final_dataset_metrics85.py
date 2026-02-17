"""
将最终宽表裁剪到“metrics 覆盖完整”的 85 个币种，并补齐缺失的 EOS（仅用 Binance Vision 作为基底）。

背景：
- `final_dataset/*_final.csv` 是以 CCXT 1m 聚合与清洗作为基底，因此部分币种（例如 EOS）可能缺失；
- 你希望按 Binance Vision 的 universe（`top100_perp_symbols.txt` 过滤后 90 个）进一步剔除
  metrics 尾部缺失（持续 404）的币种，得到 85 个。

输出：
- 目录：`AlphaQCM_data/final_dataset_metrics85/`（85 个 *_final.csv + dataset_summary.csv）
- 可选：合并 Parquet：`AlphaQCM_data/final_dataset_metrics85_all.parquet`

用法（在 AlphaQCM/ 目录下）：
  python3 data_collection/materialize_final_dataset_metrics85.py
  python3 data_collection/materialize_final_dataset_metrics85.py --write-parquet
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from gap_detector import detect_maintenance_flags


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _default_symbols_file() -> Path:
    return Path(__file__).resolve().parent / "top100_perp_symbols.txt"


def load_binance_universe(symbols_file: Path) -> list[str]:
    syms: list[str] = []
    for raw in symbols_file.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        s = s.replace("/USDT:USDT", "USDT")
        s = s.replace("/USDT", "USDT")
        s = s.replace("_", "")
        syms.append(s)
    return syms


def ccxt_to_binance_um_symbol(ccxt_symbol: str) -> str:
    s = str(ccxt_symbol).split(":", 1)[0]
    return s.replace("_", "")


def binance_to_ccxt_um_symbol(binance_symbol: str) -> str:
    if not binance_symbol.endswith("USDT"):
        raise ValueError(f"暂不支持非 USDT 合约：{binance_symbol}")
    base = binance_symbol[: -len("USDT")]
    return f"{base}_USDT:USDT"


def load_metrics_covered_symbols(
    *,
    metrics_dir: Path,
    universe: set[str],
    end_date: str,
) -> set[str]:
    """
    判断“metrics 覆盖到结束日期”的 symbol 集合。
    口径：读取 `{SYMBOL}_metrics.csv.meta.json` 的 max_ts >= end_date (UTC 00:00:00)。
    """
    end_ts = pd.Timestamp(_parse_date(end_date), tz="UTC")

    covered: set[str] = set()
    for meta_path in metrics_dir.glob("*_metrics.csv.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("status") != "ok":
            continue
        symbol = meta.get("symbol")
        if not symbol or symbol not in universe:
            continue
        mx = pd.to_datetime(meta.get("max_ts"), utc=True, errors="coerce")
        if mx is not None and pd.notna(mx) and mx >= end_ts:
            covered.add(symbol)
    return covered


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except Exception:
        import shutil

        shutil.copy2(src, dst)


def _read_vision_klines_1h(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["open_time"])
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["open_time"]).drop_duplicates(subset=["open_time"], keep="last").set_index("open_time")
    # 归一到小时索引
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index.name = "datetime"
    df = df.sort_index()
    return df


def _read_vision_csv_indexed(path: Path) -> pd.DataFrame:
    """
    Binance Vision 聚合输出通常是 index 写到第一列（可能叫 Unnamed: 0）。
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index.name = "datetime"
    return df.sort_index()


def build_eos_final_from_vision(
    *,
    sample_columns: list[str],
    binance_symbol: str,
    output_path: Path,
    vision_klines_dir: Path,
    vision_metrics_dir: Path,
    vision_funding_dir: Path,
    vision_mark_dir: Path,
    vision_index_dir: Path,
    vision_premium_dir: Path,
    vision_aggtrades_dir: Path,
) -> None:
    """
    用 Binance Vision 1h K线作为 base，补齐 EOS 的最终宽表：
    - 含 OI/多空比（metrics）
    - 含 triad（mark/index/premium + basis）
    - 含 aggTrades 订单流（at_*）
    - 动量/波动率/其它基于 CCXT 1m 的字段缺失则保持 NaN
    """
    kline_path = vision_klines_dir / f"{binance_symbol}_klines.csv"
    if not kline_path.exists():
        raise FileNotFoundError(f"缺少 Vision 1h klines：{kline_path}")

    df_k = _read_vision_klines_1h(kline_path)
    out = pd.DataFrame(index=df_k.index)
    out.index.name = "datetime"

    # base OHLCV + vwap（Vision klines 已带 quote_volume）
    rename = {
        "last_open": "open",
        "last_high": "high",
        "last_low": "low",
        "last_close": "close",
        "volume": "volume",
    }
    for src, dst in rename.items():
        if src in df_k.columns:
            out[dst] = pd.to_numeric(df_k[src], errors="coerce")

    quote_volume = pd.to_numeric(df_k.get("quote_volume"), errors="coerce")
    base_volume = pd.to_numeric(df_k.get("volume"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["vwap"] = quote_volume / (base_volume.replace(0, np.nan))

    # 资金费率（Vision fundingRate：last_funding_rate）
    funding_path = vision_funding_dir / f"{binance_symbol}_fundingRate.csv"
    if funding_path.exists():
        df_f = _read_vision_csv_indexed(funding_path)
        if "last_funding_rate" in df_f.columns:
            out["funding_rate"] = pd.to_numeric(df_f["last_funding_rate"], errors="coerce").reindex(out.index)
            out["funding_rate"] = out["funding_rate"].ffill()
            out["funding_annualized"] = out["funding_rate"] * 365 * 3
            out["funding_delta"] = out["funding_rate"].diff()
            out["arb_pressure"] = (out["funding_annualized"].abs() > 0.30).astype("int8")

    # metrics（OI + 多空比）
    metrics_path = vision_metrics_dir / f"{binance_symbol}_metrics.csv"
    if metrics_path.exists():
        df_m = pd.read_csv(metrics_path, parse_dates=["create_time"])
        if "create_time" in df_m.columns:
            df_m["create_time"] = pd.to_datetime(df_m["create_time"], utc=True, errors="coerce")
            df_m = df_m.dropna(subset=["create_time"]).drop_duplicates(subset=["create_time"], keep="last").set_index(
                "create_time"
            )
            df_m.index.name = "datetime"
            rename_map = {
                "sum_open_interest": "oi_open_interest",
                "sum_open_interest_value": "oi_open_interest_usd",
                "sum_toptrader_long_short_ratio": "ls_toptrader_long_short_ratio",
                "sum_taker_long_short_vol_ratio": "ls_taker_long_short_vol_ratio",
            }
            df_m = df_m.rename(columns=rename_map)
            for c in rename_map.values():
                if c in df_m.columns:
                    out[c] = pd.to_numeric(df_m[c], errors="coerce").reindex(out.index)

    # OI 衍生
    if "oi_open_interest" in out.columns:
        out["oi_delta"] = out["oi_open_interest"].diff()
        out["oi_delta_over_volume"] = out["oi_delta"] / (out["volume"] + 1e-12)
    if "oi_open_interest_usd" in out.columns:
        out["oi_delta_usd"] = out["oi_open_interest_usd"].diff()
        out["oi_delta_usd_over_quote_volume"] = out["oi_delta_usd"] / (out["close"] * out["volume"] + 1e-12)

    # triad + basis
    mark_path = vision_mark_dir / f"{binance_symbol}_markPriceKlines.csv"
    if mark_path.exists():
        df_mark = pd.read_csv(mark_path, parse_dates=["open_time"])
        df_mark["open_time"] = pd.to_datetime(df_mark["open_time"], utc=True, errors="coerce")
        df_mark = df_mark.dropna(subset=["open_time"]).drop_duplicates(subset=["open_time"], keep="last").set_index(
            "open_time"
        )
        if "mark_close" in df_mark.columns:
            out["triad_mark_close"] = pd.to_numeric(df_mark["mark_close"], errors="coerce").reindex(out.index)

    index_path = vision_index_dir / f"{binance_symbol}_indexPriceKlines.csv"
    if index_path.exists():
        df_index = pd.read_csv(index_path, parse_dates=["open_time"])
        df_index["open_time"] = pd.to_datetime(df_index["open_time"], utc=True, errors="coerce")
        df_index = df_index.dropna(subset=["open_time"]).drop_duplicates(subset=["open_time"], keep="last").set_index(
            "open_time"
        )
        if "index_close" in df_index.columns:
            out["triad_index_close"] = pd.to_numeric(df_index["index_close"], errors="coerce").reindex(out.index)

    premium_path = vision_premium_dir / f"{binance_symbol}_premiumIndexKlines.csv"
    if premium_path.exists():
        df_p = pd.read_csv(premium_path, parse_dates=["open_time"])
        df_p["open_time"] = pd.to_datetime(df_p["open_time"], utc=True, errors="coerce")
        df_p = df_p.dropna(subset=["open_time"]).drop_duplicates(subset=["open_time"], keep="last").set_index(
            "open_time"
        )
        if "premium_close" in df_p.columns:
            out["triad_premium_close"] = pd.to_numeric(df_p["premium_close"], errors="coerce").reindex(out.index)

    if "triad_index_close" in out.columns and "close" in out.columns:
        out["triad_basis"] = (out["close"] - out["triad_index_close"]) / out["triad_index_close"]

    # aggTrades（已是 1h 聚合，统一 at_ 前缀）
    at_path = vision_aggtrades_dir / f"{binance_symbol}_aggTrades.csv"
    if at_path.exists():
        df_at = _read_vision_csv_indexed(at_path)
        df_at = df_at.add_prefix("at_")
        out = out.join(df_at, how="left")

    # gap detector + 清洗标记（尽量与 clean_hourly_data.py 一致）
    flags = detect_maintenance_flags(out.index, max_gap_seconds=3605, cooldown_hours=2)
    out = out.join(flags, how="left")
    out["is_stable"] = (out["cooldown_no_trade"] == 0)

    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["is_spike"] = out["log_return"].abs() > 0.5

    vol_mean = out["volume"].rolling(168, min_periods=24).mean()
    vol_std = out["volume"].rolling(168, min_periods=24).std()
    out["volume_zscore"] = (out["volume"] - vol_mean) / (vol_std + 1e-8)
    out["is_volume_spike"] = out["volume_zscore"].abs() > 5

    vol_99 = out["volume"].quantile(0.99)
    out["volume_clean"] = out["volume"].clip(upper=vol_99)

    out["bars_since_start"] = range(len(out))
    out["is_mature"] = out["bars_since_start"] >= 24

    # build_final_dataset 的 schema 需要这些列（若不存在则由 reindex 补 NaN）
    out["bar_end_time"] = (out.index + pd.Timedelta(hours=1))
    out["feature_time"] = out["bar_end_time"] + pd.Timedelta(milliseconds=1)

    # 临时列丢弃
    out = out.drop(columns=["log_return", "volume_zscore", "bars_since_start"], errors="ignore")

    # 统一列集合与顺序
    out = out.reindex(columns=[c for c in sample_columns if c != "datetime"])
    out.index.name = "datetime"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="裁剪最终宽表到 metrics 覆盖完整的 85 个币种")
    parser.add_argument("--symbols-file", default=str(_default_symbols_file()))
    parser.add_argument("--end-date", default="2025-02-15")
    parser.add_argument("--final-dir", default="AlphaQCM_data/final_dataset")
    parser.add_argument("--output-dir", default="AlphaQCM_data/final_dataset_metrics85")
    parser.add_argument("--metrics-dir", default="AlphaQCM_data/binance_metrics")
    parser.add_argument("--klines-dir", default="AlphaQCM_data/binance_klines_1h")
    parser.add_argument("--funding-dir", default="AlphaQCM_data/binance_fundingRate")
    parser.add_argument("--mark-dir", default="AlphaQCM_data/binance_markPriceKlines_1h")
    parser.add_argument("--index-dir", default="AlphaQCM_data/binance_indexPriceKlines_1h")
    parser.add_argument("--premium-dir", default="AlphaQCM_data/binance_premiumIndexKlines_1h")
    parser.add_argument("--aggtrades-dir", default="AlphaQCM_data/binance_aggTrades")
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--parquet-out", default="AlphaQCM_data/final_dataset_metrics85_all.parquet")
    args = parser.parse_args()

    symbols_file = Path(args.symbols_file)
    final_dir = Path(args.final_dir)
    output_dir = Path(args.output_dir)
    metrics_dir = Path(args.metrics_dir)
    klines_dir = Path(args.klines_dir)
    funding_dir = Path(args.funding_dir)
    mark_dir = Path(args.mark_dir)
    index_dir = Path(args.index_dir)
    premium_dir = Path(args.premium_dir)
    aggtrades_dir = Path(args.aggtrades_dir)

    universe_list = load_binance_universe(symbols_file)
    universe = set(universe_list)
    covered = load_metrics_covered_symbols(metrics_dir=metrics_dir, universe=universe, end_date=args.end_date)
    allow = sorted(list(covered))

    print(f"Universe (filtered top list): {len(universe)}")
    print(f"Metrics covered to {args.end_date}: {len(covered)}")
    missing = sorted(list(universe - covered))
    if missing:
        print(f"Not covered (will be excluded): {missing}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 取一个样本文件确定统一 schema（columns 顺序）
    sample_files = sorted(final_dir.glob("*_final.csv"))
    if not sample_files:
        raise FileNotFoundError(f"未找到 *_final.csv：{final_dir}")
    sample_cols = pd.read_csv(sample_files[0], nrows=0).columns.tolist()
    if not sample_cols or sample_cols[0] != "datetime":
        raise RuntimeError(f"final_dataset 列异常：{sample_files[0]}")

    # 复制满足 allow 的最终宽表（按 CCXT 文件名映射回 Binance 符号判断）
    kept = []
    for p in sorted(final_dir.glob("*_final.csv")):
        ccxt_symbol = p.name[: -len("_final.csv")]
        b = ccxt_to_binance_um_symbol(ccxt_symbol)
        if b in covered:
            dst = output_dir / p.name
            hardlink_or_copy(p, dst)
            kept.append(ccxt_symbol)

    # 补齐 EOS（若缺失）
    eos_binance = "EOSUSDT"
    if eos_binance in covered:
        eos_ccxt = binance_to_ccxt_um_symbol(eos_binance)
        eos_out = output_dir / f"{eos_ccxt}_final.csv"
        if eos_ccxt not in kept:
            print("EOS 不在 final_dataset（CCXT 缺失），改用 Binance Vision 基底补齐...")
            build_eos_final_from_vision(
                sample_columns=sample_cols,
                binance_symbol=eos_binance,
                output_path=eos_out,
                vision_klines_dir=klines_dir,
                vision_metrics_dir=metrics_dir,
                vision_funding_dir=funding_dir,
                vision_mark_dir=mark_dir,
                vision_index_dir=index_dir,
                vision_premium_dir=premium_dir,
                vision_aggtrades_dir=aggtrades_dir,
            )
            kept.append(eos_ccxt)

    kept = sorted(kept)
    print(f"\nSaved {len(kept)} symbols -> {output_dir}")

    # 生成 summary
    summary = []
    for sym in kept:
        fp = output_dir / f"{sym}_final.csv"
        df0 = pd.read_csv(fp, usecols=["datetime"])
        dt = pd.to_datetime(df0["datetime"], utc=True, errors="coerce")
        summary.append(
            {
                "symbol": sym,
                "rows": int(len(dt)),
                "start_date": dt.min(),
                "end_date": dt.max(),
            }
        )
    df_summary = pd.DataFrame(summary).sort_values("symbol")
    df_summary.to_csv(output_dir / "dataset_summary.csv", index=False)

    # 可选：写全局 Parquet
    if args.write_parquet:
        import subprocess

        print("\nBuilding global parquet...")
        subprocess.run(
            [
                "python3",
                "data_collection/build_global_final_table.py",
                "--input-dir",
                str(output_dir),
                "--output",
                args.parquet_out,
            ],
            check=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
