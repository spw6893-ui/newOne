#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Parquet 文件中“流式”删除指定列（不会把全表读入内存）。

典型用法：
  python3 AlphaQCM/data_collection/drop_columns_from_parquet.py \\
    --input AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet \\
    --output AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet \\
    --drop-cols ls_toptrader_long_short_ratio,ls_taker_long_short_vol_ratio

说明：
- 输出支持“覆盖写”（会先写到 .tmp 再 replace）。
- 逐 row-group 读取/写入，适合 1~10GB 级文件。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_drop_cols(s: str) -> set[str]:
    out: set[str] = set()
    for part in str(s).split(","):
        p = part.strip()
        if p:
            out.add(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="流式删除 Parquet 列")
    ap.add_argument("--input", required=True, help="输入 Parquet 路径")
    ap.add_argument("--output", required=True, help="输出 Parquet 路径（可与 input 相同）")
    ap.add_argument("--drop-cols", default="", help="要删除的列（逗号分隔）")
    ap.add_argument("--compression", default="zstd", help="输出压缩：zstd/snappy/gzip/none")
    ap.add_argument("--row-group-size", type=int, default=200_000)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise FileNotFoundError(f"未找到输入：{inp}")

    drop_cols = parse_drop_cols(args.drop_cols)
    if not drop_cols:
        print("未指定 drop-cols，直接退出（无操作）")
        return 0

    pf = pq.ParquetFile(inp)
    in_cols = pf.schema.names
    keep_cols = [c for c in in_cols if c not in drop_cols]

    missing = sorted(list(drop_cols - set(in_cols)))
    if missing:
        print(f"注意：以下列在输入中不存在，将忽略：{missing}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i, columns=keep_cols)
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp,
                    schema=t.schema,
                    compression=(None if args.compression == "none" else args.compression),
                    use_dictionary=True,
                )
            writer.write_table(t, row_group_size=int(args.row_group_size))
            total += int(t.num_rows)
    finally:
        if writer is not None:
            writer.close()

    tmp.replace(out)
    print(f"saved: {out} rows={total} cols={len(keep_cols)} dropped={sorted(list(drop_cols & set(in_cols)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

