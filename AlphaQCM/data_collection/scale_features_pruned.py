#!/usr/bin/env python3
"""
特征清理（B）：winsorize + 标准化（z-score）

输入默认使用已做过共线性裁剪后的训练集：
- `AlphaQCM/AlphaQCM_data/final_dataset_filtered_pruned.parquet`

输出：
- `AlphaQCM/AlphaQCM_data/final_dataset_filtered_pruned_scaled.parquet`
- `AlphaQCM/data_collection/feature_scaling_report.md`

策略（默认）：
1) 逐特征、逐币种 winsorize（按分位数裁剪，默认 0.5%~99.5%）
2) 在 winsorize 之后，逐币种做均值/标准差标准化（ddof=0）

说明：
- 这是“离线训练友好”的做法：缩放参数使用全样本（可能包含未来），适合 batch 训练；
  如果你要严格点位不泄漏（walk-forward / online），需要改成“滚动窗口”拟合再应用。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


META_COLS = ["symbol", "datetime", "cs_universe_size", "cs_coverage_frac"]
EPS = 1e-12


def _winsorize_by_symbol(
    values: np.ndarray,
    symbols: pd.Series,
    *,
    q_low: float,
    q_high: float,
) -> np.ndarray:
    s = pd.Series(values)
    # groupby quantile 返回 MultiIndex（symbol, quantile）
    low = s.groupby(symbols).quantile(q_low, interpolation="linear")
    high = s.groupby(symbols).quantile(q_high, interpolation="linear")

    low_map = symbols.map(low).to_numpy(dtype="float64", copy=False)
    high_map = symbols.map(high).to_numpy(dtype="float64", copy=False)

    # 对 NaN 的 bounds：不裁剪（保持原值）
    out = values.astype("float64", copy=True)
    m = np.isfinite(out) & np.isfinite(low_map) & np.isfinite(high_map)
    out[m] = np.minimum(np.maximum(out[m], low_map[m]), high_map[m])
    return out


def _zscore_by_symbol(values: np.ndarray, symbols: pd.Series) -> tuple[np.ndarray, pd.Series, pd.Series]:
    s = pd.Series(values)
    mean = s.groupby(symbols).mean()
    std = s.groupby(symbols).std(ddof=0)

    mean_map = symbols.map(mean).to_numpy(dtype="float64", copy=False)
    std_map = symbols.map(std).to_numpy(dtype="float64", copy=False)

    out = values.astype("float64", copy=True)
    m = np.isfinite(out) & np.isfinite(mean_map) & np.isfinite(std_map) & (std_map > 0)
    out[m] = (out[m] - mean_map[m]) / (std_map[m] + EPS)
    # std==0 或 bounds 缺失：输出 NaN（避免把常数列变成全 0 误导训练）
    out[~m & np.isfinite(out)] = np.nan
    return out, mean, std


def _scale_single_symbol_df(
    df_sym: pd.DataFrame,
    *,
    q_low: float,
    q_high: float,
    float32: bool,
    meta_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], dict[str, int]]:
    """
    对单个 symbol 的 DataFrame 做 winsorize + zscore（不需要 groupby）。
    返回：
    - out_df
    - nan_before_ratio（按列）
    - nan_after_ratio（按列）
    - std_zero_flags（按列：std<=0 记 1，否则 0）
    """
    out_df = df_sym[[c for c in meta_cols if c in df_sym.columns]].copy()

    numeric_cols = df_sym.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c not in set(meta_cols)]

    n = max(1, int(len(df_sym)))
    nan_before_ratio: dict[str, float] = {}
    nan_after_ratio: dict[str, float] = {}
    std_zero_flags: dict[str, int] = {}

    for col in feat_cols:
        v = pd.to_numeric(df_sym[col], errors="coerce").to_numpy(dtype="float64", copy=False)
        nan_before_ratio[col] = float(np.isnan(v).mean())

        if np.isfinite(v).any():
            lo = float(np.nanquantile(v, q_low))
            hi = float(np.nanquantile(v, q_high))
            v_w = v.copy()
            m = np.isfinite(v_w)
            v_w[m] = np.minimum(np.maximum(v_w[m], lo), hi)

            mean = float(np.nanmean(v_w))
            std = float(np.nanstd(v_w, ddof=0))
            if np.isfinite(std) and std > 0:
                z = (v_w - mean) / (std + EPS)
                std_zero_flags[col] = 0
            else:
                z = np.full_like(v_w, np.nan, dtype="float64")
                std_zero_flags[col] = 1
        else:
            z = np.full_like(v, np.nan, dtype="float64")
            std_zero_flags[col] = 1

        nan_after_ratio[col] = float(np.isnan(z).mean())
        out_df[col] = z.astype("float32" if float32 else "float64")

    return out_df, nan_before_ratio, nan_after_ratio, std_zero_flags


def _partition_parquet_by_symbol(
    inp: Path,
    tmp_dir: Path,
    *,
    batch_size: int,
) -> list[str]:
    """
    将输入 Parquet 流式切成按 symbol 分文件的 Parquet（tmp_dir/{symbol}.parquet）。
    返回 symbols 列表。
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(inp)
    writers: dict[str, pq.ParquetWriter] = {}
    symbols_seen: set[str] = set()

    try:
        for batch in pf.iter_batches(batch_size=int(batch_size)):
            df = batch.to_pandas(self_destruct=True)
            if "symbol" not in df.columns:
                raise RuntimeError("输入缺少必要列：symbol")

            for sym, sub in df.groupby("symbol", sort=False):
                sym = str(sym)
                symbols_seen.add(sym)
                out_path = tmp_dir / f"{sym}.parquet"

                table = pa.Table.from_pandas(sub, preserve_index=False)
                if sym not in writers:
                    writers[sym] = pq.ParquetWriter(out_path, table.schema, compression="zstd", use_dictionary=True)
                writers[sym].write_table(table)
    finally:
        for w in writers.values():
            w.close()

    return sorted(symbols_seen)


def main() -> int:
    ap = argparse.ArgumentParser(description="对 pruned 数据集做 winsorize + 标准化（B）")
    ap.add_argument("--input", default="AlphaQCM/AlphaQCM_data/final_dataset_filtered_pruned.parquet")
    ap.add_argument("--output", default="AlphaQCM/AlphaQCM_data/final_dataset_filtered_pruned_scaled.parquet")
    ap.add_argument("--q-low", type=float, default=0.005)
    ap.add_argument("--q-high", type=float, default=0.995)
    ap.add_argument("--float32", action="store_true", help="将输出特征列转为 float32（省空间）")
    ap.add_argument("--report", default="AlphaQCM/data_collection/feature_scaling_report.md")
    ap.add_argument("--streaming", action="store_true", help="大文件模式：按 symbol 切分后逐币种处理，避免 OOM")
    ap.add_argument("--tmp-dir", default="AlphaQCM/AlphaQCM_data/_tmp_scale_by_symbol")
    ap.add_argument("--batch-size", type=int, default=50_000)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    report = Path(args.report)

    if not inp.exists():
        raise FileNotFoundError(f"未找到输入：{inp}")

    # 自动判断：文件过大时默认用 streaming
    use_streaming = bool(args.streaming)
    if not use_streaming:
        try:
            pf = pq.ParquetFile(inp)
            if int(pf.metadata.num_rows) > 300_000:
                use_streaming = True
        except Exception:
            pass

    report_lines: list[str] = []
    report_lines.append("# 特征清理（B）：winsorize + 标准化")
    report_lines.append("")
    report_lines.append(f"- 输入：`{inp}`")
    report_lines.append(f"- 输出：`{out}`")
    report_lines.append(f"- winsorize 分位数：[{float(args.q_low):.3f}, {float(args.q_high):.3f}]（逐币种）")
    report_lines.append(f"- 标准化：逐币种 z-score（winsorize 后，ddof=0）")
    report_lines.append(f"- 模式：`{'streaming' if use_streaming else 'in-memory'}`")
    report_lines.append("")
    report_lines.append("## 统计")
    if not use_streaming:
        df = pd.read_parquet(inp)
        for c in ("symbol", "datetime"):
            if c not in df.columns:
                raise RuntimeError(f"输入缺少必要列：{c}")

        # 特征列（数值列，排除 meta）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feat_cols = [c for c in numeric_cols if c not in set(META_COLS)]

        syms = df["symbol"]
        out_df = df[[c for c in META_COLS if c in df.columns]].copy()

        report_lines.append(f"- 行数：{len(df):,}")
        report_lines.append(f"- 币种数：{df['symbol'].nunique()}")
        report_lines.append(f"- 特征数：{len(feat_cols)}")
        report_lines.append("")

        # 逐列处理，避免一次 groupby transform 造成峰值内存
        nan_before = df[feat_cols].isna().mean().sort_values(ascending=False)

        bad_cols: list[str] = []
        for col in feat_cols:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64", copy=False)
            v_w = _winsorize_by_symbol(v, syms, q_low=float(args.q_low), q_high=float(args.q_high))
            v_z, _, std = _zscore_by_symbol(v_w, syms)
            # 若大部分 symbol std==0，则该列信息量很差
            if (std.fillna(0.0) <= 0).mean() > 0.8:
                bad_cols.append(col)
            if args.float32:
                out_df[col] = v_z.astype("float32")
            else:
                out_df[col] = v_z

        nan_after = out_df[feat_cols].isna().mean().sort_values(ascending=False)
    else:
        tmp_dir = Path(args.tmp_dir)
        if tmp_dir.exists():
            # 避免误删用户目录：只清理我们约定的目录名
            for p in tmp_dir.glob("*.parquet"):
                p.unlink()
        symbols = _partition_parquet_by_symbol(inp, tmp_dir, batch_size=int(args.batch_size))
        feat_cols = [c for c in pq.ParquetFile(inp).schema.names if c not in set(META_COLS)]
        feat_cols = [c for c in feat_cols if c not in ("symbol", "datetime")]

        report_lines.append(f"- 行数：{pq.ParquetFile(inp).metadata.num_rows:,}")
        report_lines.append(f"- 币种数：{len(symbols)}")
        report_lines.append(f"- 特征数：{len(feat_cols)}")
        report_lines.append("")

        nan_before_sum = {c: 0 for c in feat_cols}
        nan_after_sum = {c: 0 for c in feat_cols}
        std_zero_sum = {c: 0 for c in feat_cols}
        total_rows = 0

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = out.with_suffix(out.suffix + ".tmp")
        if tmp_out.exists():
            tmp_out.unlink()

        writer: pq.ParquetWriter | None = None
        try:
            for i, sym in enumerate(symbols, 1):
                df_sym = pd.read_parquet(tmp_dir / f"{sym}.parquet")
                total_rows += int(len(df_sym))

                out_sym, nb, na, sz = _scale_single_symbol_df(
                    df_sym,
                    q_low=float(args.q_low),
                    q_high=float(args.q_high),
                    float32=bool(args.float32),
                    meta_cols=META_COLS,
                )

                for c, r in nb.items():
                    nan_before_sum[c] = nan_before_sum.get(c, 0) + int(round(r * len(df_sym)))
                for c, r in na.items():
                    nan_after_sum[c] = nan_after_sum.get(c, 0) + int(round(r * len(df_sym)))
                for c, f in sz.items():
                    std_zero_sum[c] = std_zero_sum.get(c, 0) + int(f)

                table = pa.Table.from_pandas(out_sym, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_out, table.schema, compression="zstd", use_dictionary=True)
                writer.write_table(table, row_group_size=200_000)
                if i % 10 == 0 or i == len(symbols):
                    print(f"[{i}/{len(symbols)}] scaled {sym} rows={len(df_sym)}")
        finally:
            if writer is not None:
                writer.close()

        tmp_out.replace(out)

        # 缺失率估计（按总行数）
        nan_before = pd.Series({c: nan_before_sum.get(c, 0) / max(1, total_rows) for c in feat_cols}).sort_values(
            ascending=False
        )
        nan_after = pd.Series({c: nan_after_sum.get(c, 0) / max(1, total_rows) for c in feat_cols}).sort_values(
            ascending=False
        )
        bad_cols = [c for c in feat_cols if (std_zero_sum.get(c, 0) / max(1, len(symbols))) > 0.8]

    report_lines.append("## 缺失率变化（Top 20）")
    report_lines.append("")
    report_lines.append("| feature | NaN_before | NaN_after |")
    report_lines.append("| --- | ---: | ---: |")
    for c in nan_before.head(20).index.tolist():
        report_lines.append(f"| `{c}` | {float(nan_before[c]):.4f} | {float(nan_after.get(c, np.nan)):.4f} |")
    report_lines.append("")

    if bad_cols:
        report_lines.append("## 低信息量提示（std≈0 的币种占比过高）")
        report_lines.append("")
        for c in sorted(bad_cols)[:50]:
            report_lines.append(f"- `{c}`")
        if len(bad_cols) > 50:
            report_lines.append(f"- ...（共 {len(bad_cols)} 个）")
        report_lines.append("")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if not use_streaming:
        out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out, index=False)
        print(f"saved: {out} shape={out_df.shape} features={len(feat_cols)}")
    else:
        print(f"saved: {out} rows={pq.ParquetFile(out).metadata.num_rows} features={len(feat_cols)}")
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
