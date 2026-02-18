#!/usr/bin/env python3
"""
在“已经过滤过的训练数据”上再次做共线性裁剪（避免残留的 |corr|>0.95 对）。

为什么需要二次裁剪？
- 历史版本的共线性过滤采用“遍历相关对 + coverage 比较”策略，可能留下少量残余强相关对；
- 这里改用“按覆盖率排序 + 贪心保留”策略，保证 kept 集合内部不会再出现 >threshold 的相关对（以抽样 corr 为准）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


META_COLS = {"symbol", "datetime", "cs_universe_size", "cs_coverage_frac"}


def greedy_prune(
    df: pd.DataFrame,
    *,
    threshold: float,
    min_coverage: float,
    sample_rows: int,
    seed: int,
) -> tuple[list[str], dict[str, list[str]]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in META_COLS]

    coverage = df[features].notna().mean()
    kept_cov = coverage[coverage >= float(min_coverage)].index.tolist()
    dropped_low = sorted([c for c in features if c not in set(kept_cov)])

    # 常数/全空
    X = df[kept_cov]
    std = X.std(axis=0, ddof=0, skipna=True)
    non_all_nan = X.notna().any(axis=0)
    const_mask = (~non_all_nan) | (std.fillna(0.0) <= 0.0)
    dropped_const = sorted(X.columns[const_mask].tolist())
    kept = [c for c in kept_cov if c not in set(dropped_const)]

    if len(kept) <= 1:
        return kept, {"low_coverage": dropped_low, "constant": dropped_const, "high_corr": []}

    # 抽样算 corr
    if sample_rows and len(df) > int(sample_rows):
        Xs = df[kept].sample(n=int(sample_rows), random_state=int(seed))
    else:
        Xs = df[kept]
    corr = Xs.corr().abs()

    ordered = sorted(kept, key=lambda c: (-float(coverage.get(c, 0.0)), c))
    kept_final: list[str] = []
    dropped_high: list[str] = []
    for c in ordered:
        if not kept_final:
            kept_final.append(c)
            continue
        mx = float(corr.loc[c, kept_final].max())
        if np.isfinite(mx) and mx > float(threshold):
            dropped_high.append(c)
        else:
            kept_final.append(c)

    return kept_final, {"low_coverage": dropped_low, "constant": dropped_const, "high_corr": sorted(dropped_high)}


def main() -> int:
    ap = argparse.ArgumentParser(description="对已过滤数据再次做共线性裁剪（0.95）")
    ap.add_argument("--input", default="AlphaQCM/AlphaQCM_data/final_dataset_filtered.parquet")
    ap.add_argument("--output", default="AlphaQCM/AlphaQCM_data/final_dataset_filtered_pruned.parquet")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--min-coverage", type=float, default=0.30)
    ap.add_argument("--sample-rows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report-md", default="AlphaQCM/data_collection/collinearity_report_pruned.md")
    ap.add_argument("--kept-cols", default="AlphaQCM/data_collection/collinearity_kept_cols_pruned.txt")
    ap.add_argument("--dropped-cols", default="AlphaQCM/data_collection/collinearity_dropped_cols_pruned.txt")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise FileNotFoundError(f"未找到输入：{inp}")

    df = pd.read_parquet(inp)
    kept, drops = greedy_prune(
        df,
        threshold=float(args.threshold),
        min_coverage=float(args.min_coverage),
        sample_rows=int(args.sample_rows),
        seed=int(args.seed),
    )

    # 输出裁剪后数据
    out_cols = [c for c in ["symbol", "datetime", "cs_universe_size", "cs_coverage_frac"] if c in df.columns] + kept
    df_out = df[out_cols].copy()
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out, index=False)

    # 写清单与报告
    Path(args.kept_cols).write_text("\n".join(sorted(kept)) + "\n", encoding="utf-8")
    dropped_all = sorted(set(drops["low_coverage"] + drops["constant"] + drops["high_corr"]))
    Path(args.dropped_cols).write_text("\n".join(dropped_all) + "\n", encoding="utf-8")

    # 计算 pruned 后最大相关（抽样）
    if kept:
        if int(args.sample_rows) and len(df_out) > int(args.sample_rows):
            Xs = df_out[kept].sample(n=int(args.sample_rows), random_state=int(args.seed))
        else:
            Xs = df_out[kept]
        corr = Xs.corr().abs()
        m = np.triu(np.ones(corr.shape), k=1).astype(bool)
        mx = float(corr.where(m).max().max()) if corr.shape[0] > 1 else float("nan")
    else:
        mx = float("nan")

    report = []
    report.append("# 共线性二次裁剪报告（基于 final_dataset_filtered.parquet）")
    report.append("")
    report.append(f"- 输入：`{inp}`")
    report.append(f"- 输出：`{out}`")
    report.append(f"- 阈值：`|corr| > {float(args.threshold):.2f}`（抽样 corr）")
    report.append(f"- min coverage：`{float(args.min_coverage):.2f}`")
    report.append(f"- 抽样行数：`{int(args.sample_rows)}`")
    report.append("")
    report.append("## 结果")
    report.append(f"- 保留特征：{len(kept)}")
    report.append(f"- 删除（覆盖率过低）：{len(drops['low_coverage'])}")
    report.append(f"- 删除（常数/全空）：{len(drops['constant'])}")
    report.append(f"- 删除（高相关）：{len(drops['high_corr'])}")
    report.append(f"- pruned 后 max(|corr|) ≈ {mx:.4f}")
    report.append("")
    report.append("## 清单")
    report.append(f"- 保留列：`{args.kept_cols}`")
    report.append(f"- 删除列：`{args.dropped_cols}`")
    report.append("")

    Path(args.report_md).write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"saved {out} shape={df_out.shape} kept={len(kept)} max_corr≈{mx:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

