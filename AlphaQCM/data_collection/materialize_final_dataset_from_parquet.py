#!/usr/bin/env python3
"""
将“单一 Parquet 宽表”拆分回 `final_dataset/{SYMBOL}_final.csv` 形式。

适用场景：
- 你已经有了全局 Parquet（例如做过共线性/标准化/去极值的训练版数据）
- 但下游流程/工具仍希望读取 per-symbol 的 *_final.csv

注意：
- 该脚本不会修改原始 `AlphaQCM_data/final_dataset/`；
  建议输出到新目录（例如 `final_dataset_filtered_pruned_scaled/`）。
- 默认假设 Parquet 内部已经按 symbol、datetime 分组排序（本仓库的构建方式通常满足）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser(description="Parquet -> per-symbol *_final.csv")
    ap.add_argument("--input", default="AlphaQCM/AlphaQCM_data/final_dataset_filtered_pruned_scaled.parquet")
    ap.add_argument("--output-dir", default="AlphaQCM/AlphaQCM_data/final_dataset_filtered_pruned_scaled")
    ap.add_argument("--row-groups", type=int, default=0, help="仅处理前 N 个 row group（0=不限制），用于快速验证")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(f"未找到输入：{inp}")

    pf = pq.ParquetFile(inp)
    schema_names = pf.schema.names
    if "symbol" not in schema_names or "datetime" not in schema_names:
        raise RuntimeError("输入 Parquet 必须包含 `symbol` 与 `datetime` 列")

    seen: set[str] = set()
    total_rows = 0

    n_rg = pf.num_row_groups
    if args.row_groups and int(args.row_groups) > 0:
        n_rg = min(n_rg, int(args.row_groups))

    for rg in range(n_rg):
        tab = pf.read_row_group(rg)
        df = tab.to_pandas()
        if df.empty:
            continue

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime", "symbol"])
        if df.empty:
            continue

        # per-symbol 输出不包含 symbol 列（文件名即 symbol）
        for sym, g in df.groupby("symbol", sort=False):
            sym = str(sym)
            tmp_path = out_dir / f"{sym}_final.csv.tmp"
            out_path = out_dir / f"{sym}_final.csv"
            g2 = g.drop(columns=["symbol"])
            # 保证 datetime 列排第一
            cols = ["datetime"] + [c for c in g2.columns if c != "datetime"]
            g2 = g2[cols]

            mode = "a" if sym in seen else "w"
            header = sym not in seen
            g2.to_csv(tmp_path, mode=mode, header=header, index=False)
            seen.add(sym)
            total_rows += int(len(g2))

        if (rg + 1) % 10 == 0 or (rg + 1) == n_rg:
            print(f"[row_group {rg+1}/{n_rg}] symbols={len(seen)} total_rows={total_rows}")

    # 原子替换：.tmp -> 正式文件
    for sym in sorted(seen):
        tmp_path = out_dir / f"{sym}_final.csv.tmp"
        out_path = out_dir / f"{sym}_final.csv"
        if tmp_path.exists():
            if out_path.exists():
                out_path.unlink()
            tmp_path.replace(out_path)

    # summary
    summary = []
    for sym in sorted(seen):
        fp = out_dir / f"{sym}_final.csv"
        d = pd.read_csv(fp, usecols=["datetime"])
        dt = pd.to_datetime(d["datetime"], utc=True, errors="coerce")
        summary.append(
            {
                "symbol": sym,
                "rows": int(len(dt)),
                "start_date": dt.min(),
                "end_date": dt.max(),
            }
        )
    pd.DataFrame(summary).to_csv(out_dir / "dataset_summary.csv", index=False)

    print(f"Saved {len(seen)} symbols -> {out_dir} (rows={total_rows})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

