"""
将 `final_dataset/{SYMBOL}_final.csv` 合并为“一张大表”并落盘为 Parquet。

设计目标：
- 不需要一次性把所有 symbol 全部读进内存
- 输出 schema 与 `data_view.md` 的最终宽表一致，只额外增加 `symbol` 列

用法（在 AlphaQCM/ 目录下）：
  python3 data_collection/build_global_final_table.py
  python3 data_collection/build_global_final_table.py --output AlphaQCM_data/final_dataset_all.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def iter_final_csv_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*_final.csv"))
    if not files:
        raise FileNotFoundError(f"未找到 *_final.csv：{input_dir}")
    return files


def symbol_from_filename(p: Path) -> str:
    name = p.name
    if not name.endswith("_final.csv"):
        raise ValueError(f"文件名不符合约定：{name}")
    return name[: -len("_final.csv")]


def build_arrow_schema(sample_df: pd.DataFrame) -> pa.Schema:
    table = pa.Table.from_pandas(sample_df, preserve_index=False)
    return table.schema


def _read_csv(p: Path, parse_dates: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(p)
    for c in parse_dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


def _coerce_numeric_to_float(df: pd.DataFrame, *, exclude: set[str]) -> pd.DataFrame:
    """
    统一数值列 dtype，避免某个样本文件把 `volume` 推断成 int64，
    但其它币种出现小数导致写 Parquet 时触发“截断”错误。

    策略：除 symbol/时间列外，其余列尽量转为 float64（NaN 允许）。
    """
    for c in df.columns:
        if c in exclude:
            continue
        if c in ("symbol",):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="合并 final_dataset 为单一 Parquet 大表")
    parser.add_argument("--input-dir", default="AlphaQCM_data/final_dataset")
    parser.add_argument("--output", default="AlphaQCM_data/final_dataset_all.parquet")
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"])
    parser.add_argument("--row-group-size", type=int, default=200_000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parse_dates = ("datetime", "bar_end_time", "feature_time")

    files = iter_final_csv_files(input_dir)

    # 用首个文件推断“统一列集合与顺序”（本项目目前所有 final.csv 列一致）
    sample = _read_csv(files[0], parse_dates=parse_dates)
    sample.insert(0, "symbol", symbol_from_filename(files[0]))
    sample = _coerce_numeric_to_float(sample, exclude=set(parse_dates))

    schema = build_arrow_schema(sample)

    # 原子写：先写到 tmp，再 replace
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer = pq.ParquetWriter(
        tmp_path,
        schema=schema,
        compression=(None if args.compression == "none" else args.compression),
        use_dictionary=True,
    )

    total_rows = 0
    try:
        for i, p in enumerate(files, 1):
            sym = symbol_from_filename(p)
            df = _read_csv(p, parse_dates=parse_dates)
            df.insert(0, "symbol", sym)
            df = _coerce_numeric_to_float(df, exclude=set(parse_dates))

            table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            writer.write_table(table, row_group_size=args.row_group_size)
            total_rows += table.num_rows

            if i % 10 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] {sym} rows={table.num_rows} total={total_rows}")
    finally:
        writer.close()

    tmp_path.replace(output_path)
    print(f"Saved: {output_path} (rows={total_rows}, cols={len(schema.names)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
