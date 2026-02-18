"""
从 1 分钟 OHLCV（CCXT/API）聚合生成 1 小时“高阶动量因子”特征。

目标：为 1H 策略提供三类动量特征：
1) 分段动量（Segmented Momentum）
2) 量能动量（Volume-Augmented Momentum）
3) QRS 动量（Quality / R-squared / Slope）

输入：
- `AlphaQCM_data/crypto_1min/{SYMBOL}_1m.csv`
  列：datetime, open, high, low, close, volume

输出：
- `AlphaQCM_data/crypto_hourly_momentum/{SYMBOL}_momentum.csv`
  索引列：datetime（UTC，按小时对齐）

说明：
- 该脚本不依赖 aggTrades；纯 1m OHLCV 即可。
- 为降低内存占用，按 chunk 流式处理，逐小时写盘。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings


EPS = 1e-12


@dataclass(frozen=True)
class HourFeatures:
    ts: pd.Timestamp
    values: Dict[str, float]


def _to_utc_datetime(s: pd.Series) -> pd.Series:
    """
    将 datetime 列转为 UTC 时间戳。

    说明：
    - Vision 1m CSV 通常是 `YYYY-mm-dd HH:MM:SS+00:00`，用 format 指定可显著加速并避免 pandas 的推断警告。
    - 少量历史/异构文件可能没有时区（`YYYY-mm-dd HH:MM:SS`），因此做两段式兜底解析。
    """
    s = s.astype("string")
    dt = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S%z", utc=True, errors="coerce")
    mask = dt.isna()
    if bool(mask.any()):
        dt2 = pd.to_datetime(s[mask], format="%Y-%m-%d %H:%M:%S", utc=True, errors="coerce")
        dt.loc[mask] = dt2
        mask = dt.isna()
    if bool(mask.any()):
        # 最后兜底（极少数脏格式），避免刷屏 warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format*", category=UserWarning)
            dt3 = pd.to_datetime(s[mask], utc=True, errors="coerce")
        dt.loc[mask] = dt3
    return dt


def _linreg_slope_r2(y: np.ndarray) -> Tuple[float, float]:
    """
    对 y(t) 做一元线性回归（t=0..n-1），返回 slope 与 R^2。
    - y 建议为 log(price)，提高跨币种可比性
    """
    y = np.asarray(y, dtype="float64")
    n = int(y.size)
    if n < 3:
        return (np.nan, np.nan)

    # 去除 NaN（若全部 NaN 则返回 NaN）
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return (np.nan, np.nan)
    y = y[mask]
    n = int(y.size)
    t = np.arange(n, dtype="float64")

    t_mean = (n - 1) / 2.0
    y_mean = float(y.mean())
    dt = t - t_mean
    dy = y - y_mean
    var_t = float((dt * dt).sum())
    if var_t <= 0:
        return (np.nan, np.nan)

    slope = float((dt * dy).sum() / var_t)
    intercept = y_mean - slope * t_mean
    y_hat = intercept + slope * t
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    if ss_tot <= 0:
        r2 = 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
        # 数值误差保护
        r2 = float(np.clip(r2, 0.0, 1.0))
    return slope, r2


def _compute_hour_features(
    hour_ts: pd.Timestamp,
    df_hour: pd.DataFrame,
    *,
    us_open_logret_by_hour: Dict[pd.Timestamp, float],
) -> HourFeatures:
    """
    df_hour: 单小时内的 1m 行（已按 datetime 升序）
    必需列：datetime, open, high, low, close, volume
    """
    # 数值列在读取阶段已尽量转为 float，这里直接转 numpy，避免每小时重复 to_numeric 的开销
    open_ = df_hour["open"].to_numpy(dtype="float64", copy=False)
    high_ = df_hour["high"].to_numpy(dtype="float64", copy=False)
    low_ = df_hour["low"].to_numpy(dtype="float64", copy=False)
    close_ = df_hour["close"].to_numpy(dtype="float64", copy=False)
    vol_ = df_hour["volume"].to_numpy(dtype="float64", copy=False)

    dt = df_hour["datetime"]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, utc=True, errors="coerce")
    minute = dt.dt.minute.to_numpy(dtype="int64")

    out: Dict[str, float] = {}

    # ---- 1) 分段动量：头 45min vs 尾 15min（尾部效应） ----
    # 规则：以 minute < 45 为 head，以 minute >= 45 为 tail
    head_mask = minute < 45
    if bool(head_mask.any()):
        head_end_idx = int(np.where(head_mask)[0][-1])
        o0 = float(open_[0]) if np.isfinite(open_[0]) else np.nan
        c_head = float(close_[head_end_idx]) if np.isfinite(close_[head_end_idx]) else np.nan
        c_last = float(close_[-1]) if np.isfinite(close_[-1]) else np.nan

        out["seg_head_logret"] = float(np.log(c_head + EPS) - np.log(o0 + EPS)) if np.isfinite(o0) and np.isfinite(c_head) else np.nan
        out["seg_tail_logret"] = float(np.log(c_last + EPS) - np.log(c_head + EPS)) if np.isfinite(c_head) and np.isfinite(c_last) else np.nan
        out["seg_tail_minus_head"] = out["seg_tail_logret"] - out["seg_head_logret"] if np.isfinite(out["seg_tail_logret"]) and np.isfinite(out["seg_head_logret"]) else np.nan
        denom = out["seg_head_logret"] + out["seg_tail_logret"]
        out["seg_tail_share"] = float(out["seg_tail_logret"] / denom) if np.isfinite(denom) and abs(denom) > 0 else np.nan
    else:
        out["seg_head_logret"] = np.nan
        out["seg_tail_logret"] = np.nan
        out["seg_tail_minus_head"] = np.nan
        out["seg_tail_share"] = np.nan

    # 跨市场时段：美股 09:30 ET 后 60min 动量（按“窗口结束所在小时”落盘）
    out["seg_us_open_60m_logret"] = float(us_open_logret_by_hour.get(hour_ts, np.nan))

    # ---- 2) 量能动量：高放量分钟收益 + Amihud ----
    # 计算分钟 log return（在小时内重置，避免跨小时泄漏）
    log_close = np.log(np.maximum(close_, EPS))
    logret = np.diff(log_close, prepend=log_close[0])
    logret[0] = 0.0

    if np.isfinite(vol_).any():
        try:
            q80 = float(np.nanquantile(vol_, 0.8))
        except Exception:
            q80 = np.nan
        hi_mask = np.isfinite(vol_) & (vol_ >= q80) if np.isfinite(q80) else np.zeros_like(vol_, dtype=bool)
        out["vol_top20_frac"] = float(hi_mask.mean()) if len(hi_mask) else np.nan
        out["vol_top20_logret_sum"] = float(logret[hi_mask].sum()) if bool(hi_mask.any()) else 0.0
    else:
        out["vol_top20_frac"] = np.nan
        out["vol_top20_logret_sum"] = np.nan

    # Amihud：用“报价成交额”近似（volume * close），避免不同价格水位的可比性问题
    quote_vol = vol_ * close_
    denom = np.maximum(quote_vol, EPS)
    out["amihud_signed"] = float(np.nansum(logret / denom))
    out["amihud_abs"] = float(np.nansum(np.abs(logret) / denom))

    # ---- 3) QRS：趋势斜率 * 拟合优度 ----
    # 使用 log(price) 做回归，提高跨币种可比性
    slope_c, r2_c = _linreg_slope_r2(np.log(np.maximum(close_, EPS)))
    slope_h, r2_h = _linreg_slope_r2(np.log(np.maximum(high_, EPS)))
    slope_l, r2_l = _linreg_slope_r2(np.log(np.maximum(low_, EPS)))

    # 斜率从 “每分钟” 转为 “每小时”
    out["qrs_beta_close_per_hour"] = float(slope_c * 60.0) if np.isfinite(slope_c) else np.nan
    out["qrs_r2_close"] = float(r2_c) if np.isfinite(r2_c) else np.nan
    out["qrs_close"] = float(out["qrs_beta_close_per_hour"] * out["qrs_r2_close"]) if np.isfinite(out["qrs_beta_close_per_hour"]) and np.isfinite(out["qrs_r2_close"]) else np.nan

    out["qrs_beta_high_per_hour"] = float(slope_h * 60.0) if np.isfinite(slope_h) else np.nan
    out["qrs_r2_high"] = float(r2_h) if np.isfinite(r2_h) else np.nan
    out["qrs_high"] = float(out["qrs_beta_high_per_hour"] * out["qrs_r2_high"]) if np.isfinite(out["qrs_beta_high_per_hour"]) and np.isfinite(out["qrs_r2_high"]) else np.nan

    out["qrs_beta_low_per_hour"] = float(slope_l * 60.0) if np.isfinite(slope_l) else np.nan
    out["qrs_r2_low"] = float(r2_l) if np.isfinite(r2_l) else np.nan
    out["qrs_low"] = float(out["qrs_beta_low_per_hour"] * out["qrs_r2_low"]) if np.isfinite(out["qrs_beta_low_per_hour"]) and np.isfinite(out["qrs_r2_low"]) else np.nan

    return HourFeatures(ts=hour_ts, values=out)


def _update_us_open_cache(
    df: pd.DataFrame,
    *,
    open_by_date: Dict[pd.Timestamp, float],
    close_by_date: Dict[pd.Timestamp, Tuple[pd.Timestamp, float]],
    out_by_hour: Dict[pd.Timestamp, float],
) -> None:
    """
    在一个（合并后的）chunk 上，提取美股开盘窗口 09:30~10:29（America/New_York）的 open/close。
    - open：09:30 这一分钟的 open
    - close：10:29 这一分钟的 close
    将结果写入 out_by_hour，key 为 close_minute 的 floor('h')（UTC）
    """
    if df.empty:
        return

    dt = df["datetime"]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, utc=True, errors="coerce")
    if dt.isna().all():
        return

    # 转为纽约时区（自动处理夏令时）
    dt_ny = dt.dt.tz_convert("America/New_York")
    ny_date = dt_ny.dt.normalize()  # 当地日期（00:00）

    mask_open = (dt_ny.dt.hour == 9) & (dt_ny.dt.minute == 30)
    mask_close = (dt_ny.dt.hour == 10) & (dt_ny.dt.minute == 29)

    if bool(mask_open.any()):
        sub = df.loc[mask_open, ["open"]].copy()
        sub["_ny_date"] = ny_date[mask_open].to_numpy()
        # 每天只取第一个（理论上只有一个）
        for d, g in sub.groupby("_ny_date", sort=False):
            if d not in open_by_date:
                v = pd.to_numeric(g["open"], errors="coerce").iloc[0]
                if pd.notna(v):
                    open_by_date[d] = float(v)

    if bool(mask_close.any()):
        sub = df.loc[mask_close, ["datetime", "close"]].copy()
        sub["_ny_date"] = ny_date[mask_close].to_numpy()
        # 每天只取最后一个（理论上只有一个）
        sub["datetime"] = pd.to_datetime(sub["datetime"], utc=True, errors="coerce")
        for d, g in sub.groupby("_ny_date", sort=False):
            g = g.sort_values("datetime")
            v = pd.to_numeric(g["close"], errors="coerce").iloc[-1]
            t = g["datetime"].iloc[-1]
            if pd.notna(v) and pd.notna(t):
                close_by_date[d] = (pd.Timestamp(t), float(v))

    # 合成已齐备的日期
    ready = set(open_by_date.keys()) & set(close_by_date.keys())
    for d in sorted(ready):
        o = open_by_date.get(d)
        t_close, c = close_by_date.get(d)  # type: ignore[misc]
        if o is None:
            continue
        # 将该窗口动量落到“窗口结束所在小时”（UTC）
        hour_ts = pd.Timestamp(t_close).floor("h")
        out_by_hour[hour_ts] = float(np.log(c + EPS) - np.log(o + EPS))
        # 释放缓存，避免字典无限增长
        open_by_date.pop(d, None)
        close_by_date.pop(d, None)


def _iter_full_hours(
    csv_path: Path,
    *,
    chunksize: int,
    assume_sorted: bool,
) -> Iterator[Tuple[pd.Timestamp, pd.DataFrame]]:
    """
    流式读取 1m CSV，按小时 yield 完整小时的 DataFrame（最后一小时用 buffer 延迟到下一 chunk）。
    """
    buffer: Optional[pd.DataFrame] = None
    usecols = ["datetime", "open", "high", "low", "close", "volume"]
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=lambda c: c in usecols):
        if "datetime" not in chunk.columns:
            continue
        chunk["datetime"] = _to_utc_datetime(chunk["datetime"])
        for c in ("open", "high", "low", "close", "volume"):
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
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

        complete["__hour"] = complete["datetime"].dt.floor("h")
        for h, g in complete.groupby("__hour", sort=True):
            g = g.drop(columns=["__hour"])
            yield pd.Timestamp(h), g

    if buffer is not None and not buffer.empty:
        buffer["__hour"] = buffer["datetime"].dt.floor("h")
        for h, g in buffer.groupby("__hour", sort=True):
            g = g.drop(columns=["__hour"])
            yield pd.Timestamp(h), g


def aggregate_symbol(
    csv_path: Path,
    output_path: Path,
    *,
    chunksize: int = 200_000,
    assume_sorted: bool = True,
) -> int:
    """
    返回写入的小时行数。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    # 美股开盘窗口缓存（跨 chunk）
    us_open_open_by_date: Dict[pd.Timestamp, float] = {}
    us_open_close_by_date: Dict[pd.Timestamp, Tuple[pd.Timestamp, float]] = {}
    us_open_logret_by_hour: Dict[pd.Timestamp, float] = {}

    # 写盘缓存
    rows: List[Dict[str, float]] = []
    ts_list: List[pd.Timestamp] = []
    written = 0

    def flush() -> None:
        nonlocal written, rows, ts_list
        if not rows:
            return
        df_out = pd.DataFrame(rows, index=pd.DatetimeIndex(ts_list, name="datetime")).sort_index()
        mode = "w" if written == 0 else "a"
        header = written == 0
        df_out.to_csv(tmp_path, mode=mode, header=header)
        written += int(df_out.shape[0])
        rows = []
        ts_list = []

    # 注意：为了生成 seg_us_open_60m_logret，我们需要在同一小时聚合前先更新 us_open_logret_by_hour。
    # 简化做法：在读取“完整小时 group”时，同时对该 group 内的分钟数据更新缓存（足够覆盖 09:30/10:29 关键分钟）。
    for hour_ts, df_hour in _iter_full_hours(csv_path, chunksize=chunksize, assume_sorted=assume_sorted):
        _update_us_open_cache(
            df_hour,
            open_by_date=us_open_open_by_date,
            close_by_date=us_open_close_by_date,
            out_by_hour=us_open_logret_by_hour,
        )

        feat = _compute_hour_features(hour_ts, df_hour, us_open_logret_by_hour=us_open_logret_by_hour)
        rows.append(feat.values)
        ts_list.append(feat.ts)

        if len(rows) >= 2000:
            flush()

    flush()
    tmp_path.replace(output_path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="从 1m OHLCV 聚合高阶动量因子（1h）")
    ap.add_argument("--input-dir", default="AlphaQCM_data/crypto_1min", help="输入目录（1m CSV）")
    ap.add_argument("--output-dir", default="AlphaQCM_data/crypto_hourly_momentum", help="输出目录（1h 动量因子）")
    ap.add_argument("--chunksize", type=int, default=200_000, help="流式读取 chunksize，默认 200k")
    ap.add_argument("--assume-sorted", action="store_true", help="假设输入已按时间升序（可跳过排序，加速）")
    ap.add_argument("--skip-existing", action="store_true", help="若输出已存在则跳过（支持断点续跑）")
    ap.add_argument("--force", action="store_true", help="强制重算（覆盖已存在输出）")
    ap.add_argument("--symbols", default="", help="仅处理指定币种（逗号分隔），留空表示处理目录内所有文件")
    ap.add_argument("--symbols-file", default="", help="仅处理文件中的币种（每行一个 symbol），与 --symbols 互斥")
    ap.add_argument("--shard-index", type=int, default=0, help="分片索引（从 0 开始）")
    ap.add_argument("--shard-count", type=int, default=1, help="分片总数（默认 1=不分片）")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个文件（0=不限制），用于快速验证")
    args = ap.parse_args()

    # 以代码位置锚定 AlphaQCM 根目录，避免被当前工作目录影响
    alphaqcm_root = Path(__file__).resolve().parents[1]
    input_dir = (alphaqcm_root / args.input_dir).resolve()
    output_dir = (alphaqcm_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.symbols and args.symbols_file:
        print("参数错误：--symbols 与 --symbols-file 只能二选一")
        return 2

    if args.symbols_file:
        p = Path(args.symbols_file)
        if not p.exists():
            print(f"未找到 symbols 文件：{p}")
            return 2
        wanted = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
        files = [input_dir / f"{s}_1m.csv" for s in wanted]
        files = [f for f in files if f.exists()]
    elif args.symbols.strip():
        wanted = [s.strip() for s in args.symbols.split(",") if s.strip()]
        files = [input_dir / f"{s}_1m.csv" for s in wanted]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(input_dir.glob("*_1m.csv"))

    if args.limit and args.limit > 0:
        files = files[: int(args.limit)]

    if not files:
        print(f"未找到输入文件: {input_dir}")
        return 2

    shard_count = int(args.shard_count)
    shard_index = int(args.shard_index)
    if shard_count <= 0:
        print("参数错误：--shard-count 必须 >= 1")
        return 2
    if not (0 <= shard_index < shard_count):
        print("参数错误：--shard-index 必须满足 0 <= shard-index < shard-count")
        return 2
    if shard_count > 1:
        files = [fp for j, fp in enumerate(files) if (j % shard_count) == shard_index]

    total_written = 0
    for i, fp in enumerate(files, 1):
        symbol = fp.name.replace("_1m.csv", "")
        out_path = output_dir / f"{symbol}_momentum.csv"
        print(f"[{i}/{len(files)}] {symbol}...", end=" ")
        try:
            if out_path.exists() and args.skip_existing and not args.force:
                print("↷ Skip (exists)")
                continue
            n = aggregate_symbol(
                fp,
                out_path,
                chunksize=int(args.chunksize),
                assume_sorted=bool(args.assume_sorted),
            )
            total_written += n
            print(f"✓ {n} hours -> {out_path}")
        except Exception as e:
            print(f"✗ {e}")

    print(f"\n完成：{len(files)} 个币种，总计写入 {total_written} 小时行")
    print(f"输出目录：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
