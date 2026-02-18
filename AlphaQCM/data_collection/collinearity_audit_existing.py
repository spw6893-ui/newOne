#!/usr/bin/env python3
"""
审计“已生成的共线性过滤结果”是否符合预期，并输出保留/删除列清单。

动机：
- 机器内存紧张时，重新对全量宽表计算相关矩阵可能被 OOM Killer 杀掉；
- 但我们依然可以用 Parquet schema + 过滤后数据的抽样相关性，快速验证“是否仍存在 >0.95 的共线性”。

输出：
- `AlphaQCM/data_collection/collinearity_kept_cols_existing.txt`
- `AlphaQCM/data_collection/collinearity_dropped_cols_existing.txt`
- `AlphaQCM/data_collection/collinearity_audit.md`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.types as pat


EXCLUDE_COLS = {
    "symbol",
    "datetime",
    "bar_end_time",
    "feature_time",
    "is_valid_for_training",
    "trade_allowed",
    "under_maintenance",
    "cooldown_no_trade",
    "is_stable",
    "is_spike",
    "is_volume_spike",
    "is_mature",
    "cs_universe_size",
    "cs_coverage_frac",
    "gap_seconds",
    "n_minutes",
    "n_minutes_kept",
    "is_funding_hour",
}

FILTERED_META = {"symbol", "datetime", "cs_universe_size", "cs_coverage_frac"}


def _is_numeric_arrow_type(t) -> bool:
    # pyarrow 数据类型判定：整数/浮点/布尔都算数值列（布尔在相关里会变 0/1）
    return bool(pat.is_integer(t) or pat.is_floating(t) or pat.is_boolean(t))


def list_numeric_feature_cols_from_schema(schema) -> list[str]:
    cols: list[str] = []
    for field in schema:
        name = field.name
        if name in EXCLUDE_COLS:
            continue
        if _is_numeric_arrow_type(field.type):
            cols.append(name)
    return cols


def read_sample_filtered(
    parquet_path: Path,
    *,
    columns: list[str],
    sample_rows: int,
) -> pd.DataFrame:
    """
    以低内存方式从 Parquet 读取前 N 行样本（按 row group 顺序拼接，不做随机抽样）。
    """
    pf = pq.ParquetFile(parquet_path)
    batches = []
    remaining = int(sample_rows)
    for rg in range(pf.num_row_groups):
        if remaining <= 0:
            break
        tab = pf.read_row_group(rg, columns=columns)
        df = tab.to_pandas()
        if len(df) > remaining:
            df = df.iloc[:remaining].copy()
        batches.append(df)
        remaining -= len(df)
    if not batches:
        return pd.DataFrame(columns=columns)
    return pd.concat(batches, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="审计现有共线性过滤结果（低内存）")
    ap.add_argument("--source", default="AlphaQCM/AlphaQCM_data/final_dataset_metrics85_all.parquet")
    ap.add_argument("--filtered", default="AlphaQCM/AlphaQCM_data/final_dataset_filtered.parquet")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--sample-rows", type=int, default=100_000)
    ap.add_argument("--out-dir", default="AlphaQCM/data_collection")
    args = ap.parse_args()

    src = Path(args.source)
    flt = Path(args.filtered)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"未找到 source：{src}")
    if not flt.exists():
        raise FileNotFoundError(f"未找到 filtered：{flt}")

    src_schema = pq.ParquetFile(src).schema_arrow
    flt_schema = pq.ParquetFile(flt).schema_arrow

    src_features = sorted(list_numeric_feature_cols_from_schema(src_schema))
    # filtered 里除了 meta，剩下都是保留特征（它本身已经是训练可用输出）
    flt_cols = [f.name for f in flt_schema]
    kept = sorted([c for c in flt_cols if c not in FILTERED_META])
    dropped = sorted([c for c in src_features if c not in set(kept)])

    kept_path = out_dir / "collinearity_kept_cols_existing.txt"
    dropped_path = out_dir / "collinearity_dropped_cols_existing.txt"
    kept_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    dropped_path.write_text("\n".join(dropped) + ("\n" if dropped else ""), encoding="utf-8")

    # 抽样验证：kept 内是否仍有 >threshold 的强相关对
    sample_rows = int(args.sample_rows)
    df_s = read_sample_filtered(flt, columns=kept, sample_rows=sample_rows)
    corr_max = float("nan")
    worst_pair = ("", "", float("nan"))
    if len(kept) >= 2 and not df_s.empty:
        corr = df_s.corr().abs()
        m = np.triu(np.ones(corr.shape), k=1).astype(bool)
        upper = corr.where(m)
        # 找最大相关
        max_per_col = upper.max()
        corr_max = float(max_per_col.max())
        if np.isfinite(corr_max):
            col_b = str(max_per_col.idxmax())
            col_a = str(upper[col_b].idxmax())
            worst_pair = (col_a, col_b, float(corr_max))

    report = []
    report.append("# 共线性过滤审计（现有输出）")
    report.append("")
    report.append(f"- source：`{src}`")
    report.append(f"- filtered：`{flt}`")
    report.append(f"- 阈值：`|corr| > {float(args.threshold):.2f}`（本审计只做抽样核验）")
    report.append(f"- 抽样行数：`{min(sample_rows, len(df_s)):,}`（按 row group 顺序取前 N 行）")
    report.append("")
    report.append("## 列数量")
    report.append(f"- source 数值特征列数（剔除元数据/标志列）：{len(src_features)}")
    report.append(f"- filtered 保留特征列数：{len(kept)}")
    report.append(f"- 估算删除特征列数：{len(dropped)}")
    report.append("")
    report.append("## 抽样相关性核验")
    report.append(f"- max(|corr|) ≈ {corr_max:.4f}")
    if worst_pair[0] and np.isfinite(worst_pair[2]):
        report.append(f"- worst pair：`{worst_pair[0]}` vs `{worst_pair[1]}` -> {worst_pair[2]:.4f}")
    report.append("")
    report.append("## 清单文件")
    report.append(f"- 保留列：`{kept_path}`")
    report.append(f"- 删除列：`{dropped_path}`")
    report.append("")

    (out_dir / "collinearity_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"kept={len(kept)} dropped≈{len(dropped)} sample_max_corr≈{corr_max:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
