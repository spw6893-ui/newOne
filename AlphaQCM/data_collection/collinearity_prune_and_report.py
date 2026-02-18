#!/usr/bin/env python3
"""
对最终宽表做共线性裁剪（|corr| > 阈值）并输出报告。

默认输入：`AlphaQCM/AlphaQCM_data/final_dataset_metrics85_all.parquet`
默认输出：`AlphaQCM/AlphaQCM_data/final_dataset_filtered.parquet`

设计目标：
- 可复现：使用“按覆盖率排序 + 贪心保留”策略，避免遍历相关对时的顺序依赖
- 稳健：先做 coverage/常数列过滤，再做相关系数过滤
- 工程友好：落盘裁剪后的训练数据，并产出 markdown 报告 + 保留/删除列清单
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_EXCLUDE_COLS: list[str] = [
    # 元数据/索引
    "symbol",
    "datetime",
    "bar_end_time",
    "feature_time",
    # 质量与交易标志
    "is_valid_for_training",
    "trade_allowed",
    "under_maintenance",
    "cooldown_no_trade",
    "is_stable",
    "is_spike",
    "is_volume_spike",
    "is_mature",
    # 截面辅助
    "cs_universe_size",
    "cs_coverage_frac",
    # 过程字段（聚合脚本输出的诊断列）
    "gap_seconds",
    "n_minutes",
    "n_minutes_kept",
    "is_funding_hour",
    # 缺失率过高的 metrics 列（训练集剔除）
    "ls_toptrader_long_short_ratio",
    "ls_taker_long_short_vol_ratio",
]


@dataclass(frozen=True)
class PruneResult:
    kept: list[str]
    dropped_low_coverage: list[str]
    dropped_constant: list[str]
    dropped_high_corr: list[str]


def _iter_pairs_above_threshold(corr: pd.DataFrame, threshold: float, top_k: int = 50) -> list[tuple[str, str, float]]:
    """
    返回相关性最高的前 top_k 对（上三角），便于报告展示。
    """
    cols = list(corr.columns)
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(cols)):
        ci = cols[i]
        v = corr.iloc[i, i + 1 :]
        if v.empty:
            continue
        mx = v.max()
        if float(mx) >= float(threshold):
            j = int(np.argmax(v.to_numpy()))
            cj = cols[i + 1 + j]
            pairs.append((ci, cj, float(mx)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[: int(top_k)]


def prune_collinear_features(
    df: pd.DataFrame,
    *,
    threshold: float,
    min_coverage: float,
    exclude_cols: Iterable[str],
    valid_mask_col: str = "is_valid_for_training",
    corr_sample_rows: int = 0,
    random_seed: int = 7,
) -> PruneResult:
    """
    按覆盖率排序做贪心保留：
    - coverage < min_coverage -> 直接删
    - 常数列（std==0 / 全 NaN）-> 直接删
    - 对剩余列计算 corr（仅在 is_valid_for_training==1 的子集上）
    - 从高 coverage 到低 coverage 依次尝试加入 kept：
        如果与任一已保留列的 |corr| > threshold，则丢弃该列
    """
    exclude = set(exclude_cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    if not feature_cols:
        return PruneResult(kept=[], dropped_low_coverage=[], dropped_constant=[], dropped_high_corr=[])

    # 1) coverage 过滤（用全量 df 评估覆盖更稳定）
    coverage = df[feature_cols].notna().mean()
    kept_after_cov = coverage[coverage >= float(min_coverage)].index.tolist()
    dropped_low = [c for c in feature_cols if c not in set(kept_after_cov)]

    # 2) 常数/全 NaN 过滤（在 valid 子集上做更贴近训练）
    if valid_mask_col in df.columns:
        valid_df = df.loc[df[valid_mask_col] == 1, kept_after_cov]
    else:
        valid_df = df[kept_after_cov]

    # 可选：采样以加速（对 60w+ 行的 corr 计算能明显提速）
    if corr_sample_rows and int(corr_sample_rows) > 0 and len(valid_df) > int(corr_sample_rows):
        valid_df = valid_df.sample(n=int(corr_sample_rows), random_state=int(random_seed))

    # 先删全 NaN / 常数列
    std = valid_df.std(axis=0, ddof=0, skipna=True)
    non_all_nan = valid_df.notna().any(axis=0)
    is_constant = (~non_all_nan) | (std.fillna(0.0) <= 0.0)
    dropped_constant = sorted(valid_df.columns[is_constant].tolist())
    valid_features = [c for c in kept_after_cov if c not in set(dropped_constant)]

    if len(valid_features) <= 1:
        return PruneResult(
            kept=sorted(valid_features),
            dropped_low_coverage=sorted(dropped_low),
            dropped_constant=dropped_constant,
            dropped_high_corr=[],
        )

    # 3) corr 计算（pairwise complete obs）
    corr = valid_df[valid_features].corr().abs()

    # 4) 贪心保留：按 coverage 降序，coverage 相同按列名
    ordered = sorted(valid_features, key=lambda c: (-float(coverage.get(c, 0.0)), c))

    kept: list[str] = []
    dropped_high: list[str] = []
    for c in ordered:
        if not kept:
            kept.append(c)
            continue
        # 与已保留列最大相关性
        mx = float(corr.loc[c, kept].max())
        if np.isfinite(mx) and mx > float(threshold):
            dropped_high.append(c)
        else:
            kept.append(c)

    return PruneResult(
        kept=kept,
        dropped_low_coverage=sorted(dropped_low),
        dropped_constant=dropped_constant,
        dropped_high_corr=sorted(dropped_high),
    )


def _write_list(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")


def _sample_parquet_to_pandas(
    input_path: Path,
    *,
    sample_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    从大 Parquet 中采样到 Pandas，避免一次性 read_parquet OOM。

    采样策略：
    - 随机挑选若干 row group 读入
    - 拼接后再做一次 Pandas 层面的 sample(n)
    """
    pf = pq.ParquetFile(input_path)
    n_groups = int(pf.num_row_groups)
    if n_groups <= 0:
        raise RuntimeError(f"Parquet row groups 异常：{input_path}")

    target = int(sample_rows)
    if target <= 0:
        target = 200_000

    rng = np.random.default_rng(int(random_seed))
    order = rng.permutation(n_groups).tolist()

    tables: list[pa.Table] = []
    collected = 0
    read_until = int(target * 2)
    for rg in order:
        t = pf.read_row_group(int(rg))
        tables.append(t)
        collected += int(t.num_rows)
        if collected >= read_until:
            break

    table = pa.concat_tables(tables, promote_options="default")
    df = table.to_pandas(self_destruct=True)
    if len(df) > target:
        df = df.sample(n=target, random_state=int(random_seed))

    # 压内存：数值列尽量转 float32
    for c in df.columns:
        if c in ("symbol", "datetime", "bar_end_time", "feature_time"):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


def _write_pruned_parquet_streaming(
    input_path: Path,
    output_path: Path,
    *,
    out_cols: list[str],
    valid_mask_col: str = "is_valid_for_training",
    compression: str = "zstd",
    row_group_size: int = 200_000,
) -> tuple[int, int]:
    """
    流式写 Parquet（逐 row-group 读取/过滤/写出），避免全量加载。
    返回 (rows_written, cols_written)。
    """
    pf = pq.ParquetFile(input_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i, columns=out_cols)
            if valid_mask_col in t.column_names:
                mask = pc.equal(t[valid_mask_col], 1)
                t = t.filter(mask)
                t = t.drop([valid_mask_col])
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path,
                    schema=t.schema,
                    compression=(None if compression == "none" else compression),
                    use_dictionary=True,
                )
            writer.write_table(t, row_group_size=int(row_group_size))
            total += int(t.num_rows)
    finally:
        if writer is not None:
            writer.close()

    tmp_path.replace(output_path)
    cols = len(pq.ParquetFile(output_path).schema.names)
    return total, cols


def main() -> int:
    ap = argparse.ArgumentParser(description="共线性裁剪 + 报告生成（默认阈值 0.95）")
    ap.add_argument("--input", default="AlphaQCM/AlphaQCM_data/final_dataset_metrics85_all.parquet")
    ap.add_argument("--output", default="AlphaQCM/AlphaQCM_data/final_dataset_filtered.parquet")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--min-coverage", type=float, default=0.30)
    ap.add_argument("--corr-sample-rows", type=int, default=0, help="可选：相关矩阵计算采样行数（0=不采样）")
    ap.add_argument("--report-md", default="AlphaQCM/data_collection/collinearity_report.md")
    ap.add_argument("--kept-cols", default="AlphaQCM/data_collection/collinearity_kept_cols.txt")
    ap.add_argument("--dropped-cols", default="AlphaQCM/data_collection/collinearity_dropped_cols.txt")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report_md)
    kept_path = Path(args.kept_cols)
    dropped_path = Path(args.dropped_cols)

    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件：{input_path}")

    print(f"Sampling parquet for corr/coverage: {input_path}")
    df = _sample_parquet_to_pandas(
        input_path,
        sample_rows=int(args.corr_sample_rows),
        random_seed=7,
    )
    print(f"Sample shape: {df.shape}")

    # 计算保留列
    res = prune_collinear_features(
        df,
        threshold=float(args.threshold),
        min_coverage=float(args.min_coverage),
        exclude_cols=DEFAULT_EXCLUDE_COLS,
        corr_sample_rows=int(args.corr_sample_rows),
    )

    # 输出数据：保留元数据 + 保留特征，且仅保留 valid 行
    meta_cols = ["symbol", "datetime", "cs_universe_size", "cs_coverage_frac", "is_valid_for_training"]
    out_cols = meta_cols + res.kept
    # 只保留输入里确实存在的列（避免 schema 变更导致 KeyError）
    in_schema_cols = set(pq.ParquetFile(input_path).schema.names)
    out_cols = [c for c in out_cols if c in in_schema_cols]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written, cols_written = _write_pruned_parquet_streaming(
        input_path,
        output_path,
        out_cols=out_cols,
        valid_mask_col="is_valid_for_training",
        compression="zstd",
        row_group_size=200_000,
    )
    print(f"Saved: {output_path} rows={rows_written} cols={cols_written}")

    # 写清单
    dropped_all = sorted(set(res.dropped_low_coverage + res.dropped_constant + res.dropped_high_corr))
    _write_list(kept_path, res.kept)
    _write_list(dropped_path, dropped_all)

    # 生成报告（用 valid 子集计算 kept 内部的 top corr 对）
    valid_df = df.loc[df.get("is_valid_for_training", 1) == 1, res.kept] if res.kept else pd.DataFrame()
    top_pairs = []
    if len(res.kept) >= 2 and not valid_df.empty:
        corr_kept = valid_df.corr().abs()
        top_pairs = _iter_pairs_above_threshold(corr_kept, threshold=max(0.0, float(args.threshold) - 0.02), top_k=50)

    report_lines: list[str] = []
    report_lines.append("# 共线性裁剪报告")
    report_lines.append("")
    report_lines.append(f"- 输入：`{input_path}`")
    report_lines.append(f"- 输出：`{output_path}`")
    report_lines.append(f"- 阈值：`|corr| > {float(args.threshold):.2f}`")
    report_lines.append(f"- 最小覆盖率：`{float(args.min_coverage):.2f}`")
    report_lines.append(f"- corr 采样行数：`{int(args.corr_sample_rows)}`（0=不采样）")
    report_lines.append("")
    report_lines.append("## 统计")
    report_lines.append(f"- 保留特征：{len(res.kept)}")
    report_lines.append(f"- 删除（覆盖率过低）：{len(res.dropped_low_coverage)}")
    report_lines.append(f"- 删除（常数/全空）：{len(res.dropped_constant)}")
    report_lines.append(f"- 删除（高相关）：{len(res.dropped_high_corr)}")
    report_lines.append("")
    report_lines.append("## 清单")
    report_lines.append(f"- 保留列：`{kept_path}`")
    report_lines.append(f"- 删除列：`{dropped_path}`")
    report_lines.append("")
    if top_pairs:
        report_lines.append("## 保留特征中的高相关对（抽样/近阈值展示）")
        report_lines.append("")
        report_lines.append("| col_a | col_b | |corr| |")
        report_lines.append("| --- | --- | ---: |")
        for a, b, v in top_pairs:
            report_lines.append(f"| `{a}` | `{b}` | {v:.4f} |")
        report_lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
