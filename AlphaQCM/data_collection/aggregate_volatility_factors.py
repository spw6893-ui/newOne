"""
从 1 分钟 OHLCV（CCXT/API）聚合生成 1 小时“波动率/微观结构”因子特征。

输入：
- `AlphaQCM_data/crypto_1min/{SYMBOL}_1m.csv`
  列：datetime, open, high, low, close, volume

输出：
- `AlphaQCM_data/crypto_hourly_volatility/{SYMBOL}_volatility.csv`
  索引列：datetime（UTC，按小时对齐）

实现要点（1H 策略常用）：
- 分钟成交量标准差（脉冲式成交 vs 均匀成交）
- 分钟极比（High/Low）序列的标准差（插针/流动性厚度）
- 上行/下行波动率占比（波动的非对称性）
- 实现使用 log return（跨币种可比性更好）

工程注意：
- 流式读取（chunksize）降低内存占用
- 对“异常分钟”（High/Low 过大）做温和过滤，避免 API/异常烛形主导统计
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class HourVolFeatures:
    ts: pd.Timestamp
    values: Dict[str, float]


def _moment_skew_kurt_excess(x: np.ndarray) -> Tuple[float, float]:
    """
    计算偏度与“超额峰度”(excess kurtosis)：
    - skew = E[(x-μ)^3] / σ^3
    - kurt_excess = E[(x-μ)^4] / σ^4 - 3

    说明：用于 doc_* 的分箱成交量向量，采用总体矩(除以 N)口径，速度更快。
    """
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan"), float("nan")
    mu = float(x.mean())
    xc = x - mu
    var = float((xc * xc).mean())
    if not np.isfinite(var) or var <= 0:
        return float("nan"), float("nan")
    std = float(np.sqrt(var))
    m3 = float((xc * xc * xc).mean())
    m4 = float((xc * xc * xc * xc).mean())
    skew = m3 / (std**3 + EPS)
    kurt_excess = m4 / (std**4 + EPS) - 3.0
    return float(skew), float(kurt_excess)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    加权分位数（权重=成交量）。

    返回：使累计权重达到 q 的 value（不做插值）。
    """
    v = np.asarray(values, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not bool(m.any()):
        return float("nan")
    v = v[m]
    w = w[m]
    total_w = float(w.sum())
    if not np.isfinite(total_w) or total_w <= 0:
        return float("nan")
    q = float(np.clip(q, 0.0, 1.0))
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    cum = np.cumsum(w)
    cutoff = q * total_w
    idx = int(np.searchsorted(cum, cutoff, side="left"))
    if idx < 0:
        idx = 0
    if idx >= v.size:
        idx = v.size - 1
    return float(v[idx])


def _pearson_corr(x: np.ndarray, y: np.ndarray, *, min_n: int = 10) -> float:
    """
    计算 Pearson 相关系数（小时内分钟序列）。
    - 有效样本 < min_n 或任一序列方差为 0 时返回 NaN
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < int(min_n):
        return float("nan")
    x = x[m]
    y = y[m]
    sx = float(np.std(x, ddof=0))
    sy = float(np.std(y, ddof=0))
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _to_utc_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _iter_full_hours(
    csv_path: Path,
    *,
    chunksize: int,
) -> Iterator[Tuple[pd.Timestamp, pd.DataFrame]]:
    """
    流式读取 1m CSV，按小时 yield 完整小时的 DataFrame（最后一小时延迟到下一 chunk）。
    """
    buffer: Optional[pd.DataFrame] = None
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if "datetime" not in chunk.columns:
            continue
        chunk["datetime"] = _to_utc_datetime(chunk["datetime"])
        chunk = chunk.dropna(subset=["datetime"])
        if chunk.empty:
            continue

        if buffer is not None and not buffer.empty:
            chunk = pd.concat([buffer, chunk], ignore_index=True)

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


def _compute_hour_vol_features(
    hour_ts: pd.Timestamp,
    df_hour: pd.DataFrame,
    *,
    max_min_range_ratio: float = 1.2,
) -> HourVolFeatures:
    """
    df_hour: 单小时内的 1m 行（已按 datetime 升序）
    必需列：datetime, open, high, low, close, volume
    """
    open_ = pd.to_numeric(df_hour["open"], errors="coerce").to_numpy(dtype="float64")
    high_ = pd.to_numeric(df_hour["high"], errors="coerce").to_numpy(dtype="float64")
    low_ = pd.to_numeric(df_hour["low"], errors="coerce").to_numpy(dtype="float64")
    close_ = pd.to_numeric(df_hour["close"], errors="coerce").to_numpy(dtype="float64")
    vol_ = pd.to_numeric(df_hour["volume"], errors="coerce").to_numpy(dtype="float64")
    dt_utc = pd.to_datetime(df_hour["datetime"], utc=True, errors="coerce")
    minute_ = dt_utc.dt.minute.to_numpy(dtype="int64")

    n_total = int(len(df_hour))

    # 异常分钟过滤：high/low 过大通常是脏数据或极端插针；默认阈值较宽（1.2=20%）
    range_ratio = high_ / np.maximum(low_, EPS)
    keep = np.isfinite(range_ratio) & (range_ratio > 0) & (range_ratio <= float(max_min_range_ratio))
    keep &= np.isfinite(close_) & (close_ > 0) & np.isfinite(vol_)

    # 最少保留 10 分钟才计算（否则统计不稳定）
    if keep.sum() < 10:
        # 降级：不做过滤，尽量给出结果
        keep = np.isfinite(close_) & (close_ > 0) & np.isfinite(vol_) & np.isfinite(high_) & np.isfinite(low_) & (low_ > 0)

    open_k = open_[keep]
    high_k = high_[keep]
    low_k = low_[keep]
    close_k = close_[keep]
    vol_k = vol_[keep]
    minute_k = minute_[keep]
    n_kept = int(keep.sum())

    out: Dict[str, float] = {
        "n_minutes": float(n_total),
        "n_minutes_kept": float(n_kept),
        "min_range_ratio_max": float(np.nanmax(range_ratio)) if np.isfinite(range_ratio).any() else np.nan,
    }

    # 1) 分钟成交量标准差
    out["vol_1m_mean"] = float(np.nanmean(vol_k)) if n_kept else np.nan
    out["vol_1m_std"] = float(np.nanstd(vol_k, ddof=0)) if n_kept else np.nan
    out["vol_1m_cv"] = float(out["vol_1m_std"] / (out["vol_1m_mean"] + EPS)) if np.isfinite(out["vol_1m_std"]) and np.isfinite(out["vol_1m_mean"]) else np.nan

    # 2) 分钟极比（high/low）标准差（以及 log 极比标准差，数值更稳定）
    rr = high_k / np.maximum(low_k, EPS)
    out["range_ratio_1m_std"] = float(np.nanstd(rr, ddof=0)) if rr.size else np.nan
    out["log_range_1m_std"] = float(np.nanstd(np.log(np.maximum(rr, EPS)), ddof=0)) if rr.size else np.nan

    # 3) 上/下行波动率占比（基于 log return）
    log_close = np.log(np.maximum(close_k, EPS))
    r = np.diff(log_close, prepend=log_close[0])
    if r.size:
        r[0] = 0.0
    up = r[r > 0]
    down = r[r < 0]
    up_ss = float((up * up).sum()) if up.size else 0.0
    down_ss = float((down * down).sum()) if down.size else 0.0
    total_ss = up_ss + down_ss
    out["up_vol_l2"] = float(np.sqrt(up_ss))
    out["down_vol_l2"] = float(np.sqrt(down_ss))
    out["up_var_share"] = float(up_ss / total_ss) if total_ss > 0 else np.nan
    out["down_var_share"] = float(down_ss / total_ss) if total_ss > 0 else np.nan

    # 4) 传统小时内 realized volatility（两种口径）
    out["rv_std_sqrt60"] = float(np.nanstd(r, ddof=0) * np.sqrt(60.0)) if r.size else np.nan
    out["rv_l2"] = float(np.sqrt(float((r * r).sum()))) if r.size else np.nan

    # 5) 单位成交量波动率（流动性 proxy）
    out["rv_per_vol"] = float(out["rv_std_sqrt60"] / (out["vol_1m_mean"] + EPS)) if np.isfinite(out["rv_std_sqrt60"]) and np.isfinite(out["vol_1m_mean"]) else np.nan

    # 6) 形态因子：分钟收益率/分钟量比 的偏度/峰度/比值
    # 说明：
    # - pandas 的 kurt() 返回“超额峰度”(excess kurtosis)，正态分布为 0。
    # - 为避免第一个分钟 return=0 人为影响，形态统计从 r[1:] 开始。
    if r.size >= 10:
        r1 = pd.Series(r[1:], dtype="float64")
        skew = float(r1.skew())
        kurt = float(r1.kurt())
        out["shape_skew"] = skew
        out["shape_kurt"] = kurt
        out["shape_skratio"] = float(skew / (abs(kurt) + EPS)) if np.isfinite(skew) and np.isfinite(kurt) else np.nan
    else:
        out["shape_skew"] = np.nan
        out["shape_kurt"] = np.nan
        out["shape_skratio"] = np.nan

    vol_sum = float(np.nansum(vol_k)) if n_kept else 0.0
    if n_kept >= 10 and vol_sum > 0:
        vol_share = pd.Series((vol_k / vol_sum).astype("float64"))
        skew_v = float(vol_share.skew())
        kurt_v = float(vol_share.kurt())
        out["shape_skewVol"] = skew_v
        out["shape_kurtVol"] = kurt_v
        out["shape_skratioVol"] = float(skew_v / (abs(kurt_v) + EPS)) if np.isfinite(skew_v) and np.isfinite(kurt_v) else np.nan
    else:
        out["shape_skewVol"] = np.nan
        out["shape_kurtVol"] = np.nan
        out["shape_skratioVol"] = np.nan

    # 7) 流动性/非流动性因子（纯 1m OHLCV 近似）
    # A) Amihud 非流动性（小时内分钟均值）
    # Illiq = Mean(|Return_1m| / Volume_1m)
    if r.size >= 2 and vol_k.size >= 2:
        abs_r1 = np.abs(r[1:])
        vol1 = np.maximum(vol_k[1:], EPS)
        out["liq_amihud"] = float(np.nanmean(abs_r1 / vol1))
    else:
        out["liq_amihud"] = np.nan

    # B) 整点前后成交量占比（替代集合竞价/结算影响 proxy）
    total_vol = float(np.nansum(vol_k)) if n_kept else 0.0
    if n_kept and total_vol > 0:
        last5 = float(np.nansum(vol_k[minute_k >= 55]))
        first5 = float(np.nansum(vol_k[minute_k < 5]))
        first10 = float(np.nansum(vol_k[minute_k < 10]))
        out["liq_last_5min_R"] = float(last5 / total_vol)
        out["liq_funding_impact"] = float((first5 + last5) / total_vol)
        out["liq_top_of_hour_ratio"] = float(first10 / total_vol)
    else:
        out["liq_last_5min_R"] = np.nan
        out["liq_funding_impact"] = np.nan
        out["liq_top_of_hour_ratio"] = np.nan

    # C) 资金费结算时刻标记（UTC 00/08/16）
    out["is_funding_hour"] = 1.0 if int(hour_ts.hour) in (0, 8, 16) else 0.0

    # D) 盘口深度/价差 proxy
    # - liq_range_vol_ratio：Mean((High-Low) / Volume)
    # - liq_spread_std：Std((High-Low) / Close)
    if n_kept:
        spread_proxy = (high_k - low_k) / np.maximum(vol_k, EPS)
        out["liq_range_vol_ratio"] = float(np.nanmean(spread_proxy))
        rel_range = (high_k - low_k) / np.maximum(close_k, EPS)
        out["liq_spread_std"] = float(np.nanstd(rel_range, ddof=0)) if rel_range.size else np.nan
    else:
        out["liq_range_vol_ratio"] = np.nan
        out["liq_spread_std"] = np.nan

    # E) 尾部风险：最后 10 分钟波动率 / 全小时波动率
    if r.size and minute_k.size == r.size:
        tail_mask = minute_k >= 50
        total_l2 = float(np.sqrt(float((r * r).sum())))
        tail_l2 = float(np.sqrt(float((r[tail_mask] * r[tail_mask]).sum()))) if bool(tail_mask.any()) else 0.0
        out["liq_tail_risk"] = float(tail_l2 / (total_l2 + EPS)) if total_l2 > 0 else np.nan
        out["liq_tail_var_share"] = float(float((r[tail_mask] * r[tail_mask]).sum()) / (float((r * r).sum()) + EPS)) if total_l2 > 0 else np.nan
    else:
        out["liq_tail_risk"] = np.nan
        out["liq_tail_var_share"] = np.nan

    # 7) 相关性因子：分钟收益率/价格 与成交量(及其变化率)的相关系数
    # - 收益率：使用分钟 log return（r），并从 r[1:] 开始避免第一分钟人为 0
    # - 成交量变化率：使用 pct_change（vol[t]/vol[t-1]-1），并轻微裁剪以降低极端值影响
    if n_kept >= 10:
        if r.size >= 2 and vol_k.size >= 2:
            r1 = r[1:]
            vol1 = vol_k[1:]
            out["corr_prv"] = _pearson_corr(r1, vol1)

            vol_ret = (vol_k[1:] / np.maximum(vol_k[:-1], EPS)) - 1.0
            vol_ret = np.clip(vol_ret, -10.0, 10.0)
            out["corr_prvr"] = _pearson_corr(r1, vol_ret)
        else:
            out["corr_prv"] = np.nan
            out["corr_prvr"] = np.nan

        out["corr_pv"] = _pearson_corr(close_k, vol_k)

        if close_k.size >= 2 and vol_k.size >= 2:
            out["corr_pvd"] = _pearson_corr(close_k[1:], vol_k[:-1])  # 滞后成交量
            out["corr_pvl"] = _pearson_corr(close_k[:-1], vol_k[1:])  # 领先成交量

            vol_ret2 = (vol_k[1:] / np.maximum(vol_k[:-1], EPS)) - 1.0
            vol_ret2 = np.clip(vol_ret2, -10.0, 10.0)
            out["corr_pvr"] = _pearson_corr(close_k[1:], vol_ret2)
        else:
            out["corr_pvd"] = np.nan
            out["corr_pvl"] = np.nan
            out["corr_pvr"] = np.nan
    else:
        out["corr_prv"] = np.nan
        out["corr_prvr"] = np.nan
        out["corr_pv"] = np.nan
        out["corr_pvd"] = np.nan
        out["corr_pvl"] = np.nan
        out["corr_pvr"] = np.nan

    # 8) 筹码分布（Volume-at-Return Distribution, doc_*）
    # 目标：描述“这一小时的成交量，主要堆积在什么样的分钟收益率水平上？”
    # - doc_* 采用分钟收益率 r（log return）与分钟成交量 vol 的组合构造
    # - 分位数口径：成交量加权分位（不插值）
    # - 分箱口径：按该小时内 r 的 min/max 动态分箱（固定 25 bins），统计每个收益率 bin 的成交量
    if n_kept >= 12 and r.size >= 3 and vol_k.size >= 3:
        r1 = r[1:]
        vol1 = vol_k[1:]

        out["doc_vol_pdf60"] = _weighted_quantile(r1, vol1, 0.60)
        out["doc_vol_pdf70"] = _weighted_quantile(r1, vol1, 0.70)
        out["doc_vol_pdf80"] = _weighted_quantile(r1, vol1, 0.80)
        out["doc_vol_pdf90"] = _weighted_quantile(r1, vol1, 0.90)
        out["doc_vol_pdf95"] = _weighted_quantile(r1, vol1, 0.95)
        # 双边 90%：abs(return) 的成交量加权 90% 分位（90% 成交量落在 [-x, +x] 的阈值 x）
        out["doc_vol_pdf90bi"] = _weighted_quantile(np.abs(r1), vol1, 0.90)

        rr_min = float(np.nanmin(r1)) if np.isfinite(r1).any() else float("nan")
        rr_max = float(np.nanmax(r1)) if np.isfinite(r1).any() else float("nan")
        if np.isfinite(rr_min) and np.isfinite(rr_max):
            if rr_max <= rr_min:
                pad = 1e-8
                rr_min -= pad
                rr_max += pad
            bin_n = 25
            edges = np.linspace(rr_min, rr_max, bin_n + 1)
            vol_bins, _ = np.histogram(r1, bins=edges, weights=np.maximum(vol1, 0.0))

            out["doc_std"] = float(np.nanstd(vol_bins, ddof=0))
            sk, ku = _moment_skew_kurt_excess(vol_bins.astype("float64"))
            out["doc_skew"] = sk
            out["doc_kurt"] = ku

            vb_sum = float(np.nansum(vol_bins))
            if np.isfinite(vb_sum) and vb_sum > 0:
                vb_sorted = np.sort(vol_bins)[::-1]
                for top_n, name in ((5, "doc_vol5_ratio"), (10, "doc_vol10_ratio"), (50, "doc_vol50_ratio")):
                    n_use = int(min(top_n, vb_sorted.size))
                    out[name] = float(vb_sorted[:n_use].sum() / vb_sum) if n_use > 0 else np.nan
            else:
                out["doc_vol5_ratio"] = np.nan
                out["doc_vol10_ratio"] = np.nan
                out["doc_vol50_ratio"] = np.nan
        else:
            out["doc_std"] = np.nan
            out["doc_skew"] = np.nan
            out["doc_kurt"] = np.nan
            out["doc_vol5_ratio"] = np.nan
            out["doc_vol10_ratio"] = np.nan
            out["doc_vol50_ratio"] = np.nan
    else:
        out["doc_kurt"] = np.nan
        out["doc_skew"] = np.nan
        out["doc_std"] = np.nan
        out["doc_vol_pdf60"] = np.nan
        out["doc_vol_pdf70"] = np.nan
        out["doc_vol_pdf80"] = np.nan
        out["doc_vol_pdf90"] = np.nan
        out["doc_vol_pdf90bi"] = np.nan
        out["doc_vol_pdf95"] = np.nan
        out["doc_vol10_ratio"] = np.nan
        out["doc_vol5_ratio"] = np.nan
        out["doc_vol50_ratio"] = np.nan

    return HourVolFeatures(ts=hour_ts, values=out)


def aggregate_symbol(
    csv_path: Path,
    output_path: Path,
    *,
    chunksize: int = 200_000,
    max_min_range_ratio: float = 1.2,
) -> int:
    """
    返回写入的小时行数。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    rows: List[Dict[str, float]] = []
    ts_list: List[pd.Timestamp] = []

    def flush(written: int) -> int:
        if not rows:
            return written
        df_out = pd.DataFrame(rows, index=pd.DatetimeIndex(ts_list, name="datetime")).sort_index()
        # 衍生：波动率稳定性比率（相对过去 24 小时的均值）
        if "rv_std_sqrt60" in df_out.columns:
            ma24 = df_out["rv_std_sqrt60"].rolling(24, min_periods=24).mean()
            df_out["rv_stability_24h"] = df_out["rv_std_sqrt60"] / (ma24 + EPS)
        mode = "w" if written == 0 else "a"
        header = written == 0
        df_out.to_csv(output_path, mode=mode, header=header)
        rows.clear()
        ts_list.clear()
        return written + int(df_out.shape[0])

    written = 0
    for hour_ts, df_hour in _iter_full_hours(csv_path, chunksize=chunksize):
        feat = _compute_hour_vol_features(hour_ts, df_hour, max_min_range_ratio=max_min_range_ratio)
        rows.append(feat.values)
        ts_list.append(feat.ts)
        if len(rows) >= 2000:
            written = flush(written)

    written = flush(written)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="从 1m OHLCV 聚合波动率因子（1h）")
    ap.add_argument("--input-dir", default="AlphaQCM_data/crypto_1min", help="输入目录（1m CSV）")
    ap.add_argument("--output-dir", default="AlphaQCM_data/crypto_hourly_volatility", help="输出目录（1h 波动率因子）")
    ap.add_argument("--chunksize", type=int, default=200_000, help="流式读取 chunksize，默认 200k")
    ap.add_argument("--max-min-range-ratio", type=float, default=1.2, help="过滤异常分钟：High/Low 超过该阈值则忽略该分钟")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个文件（0=不限制），用于快速验证")
    args = ap.parse_args()

    alphaqcm_root = Path(__file__).resolve().parents[1]
    input_dir = (alphaqcm_root / args.input_dir).resolve()
    output_dir = (alphaqcm_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*_1m.csv"))
    if args.limit and args.limit > 0:
        files = files[: int(args.limit)]

    if not files:
        print(f"未找到输入文件: {input_dir}")
        return 2

    total_written = 0
    for i, fp in enumerate(files, 1):
        symbol = fp.name.replace("_1m.csv", "")
        out_path = output_dir / f"{symbol}_volatility.csv"
        print(f"[{i}/{len(files)}] {symbol}...", end=" ")
        try:
            n = aggregate_symbol(
                fp,
                out_path,
                chunksize=int(args.chunksize),
                max_min_range_ratio=float(args.max_min_range_ratio),
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
