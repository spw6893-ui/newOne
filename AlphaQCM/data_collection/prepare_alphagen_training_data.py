#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 AlphaGen/时序模型准备“无前视偏差”的训练数据（按币种逐文件输出）。

设计目标：
- 明确时间语义：将 feature_time 视为“该小时 bar 收盘后可用的特征时间点”
- 目标（label）使用未来收益：默认预测下一小时（1H forward）的 log return
- 提供一致的质量过滤与缺失值策略（可选）

输入：
- `AlphaQCM_data/final_dataset/{SYMBOL}_final.csv`

输出：
- `AlphaQCM_data/alphagen_ready/{SYMBOL}_train.csv`

典型用法：
  python3 AlphaQCM/data_collection/prepare_alphagen_training_data.py \
    --input-dir AlphaQCM_data/final_dataset \
    --output-dir AlphaQCM_data/alphagen_ready \
    --horizon-hours 1 \
    --filter-quality \
    --impute ffill --ffill-limit 24
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


def _read_header_cols(fp: Path) -> list[str]:
    """
    仅读取 CSV 表头，避免全量载入。

    约定：输入文件是 `*_final.csv`（index=第一列），因此返回的 cols 不包含 index 列名。
    """
    with fp.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    if not header:
        return []
    # 第一列是 index（datetime），不纳入 columns
    return [c.strip() for c in header[1:] if c is not None and str(c).strip()]


def _canonical_order(cols: Iterable[str]) -> list[str]:
    """
    给输出列一个稳定顺序，便于 diff/调试，也便于训练端做一致的 FeatureType 映射。
    """
    cols = list(dict.fromkeys([str(c).strip() for c in cols if str(c).strip()]))
    colset = set(cols)

    core = [c for c in ("symbol", "feature_time", "split", "open", "high", "low", "close", "volume", "volume_clean", "vwap") if c in colset]
    labels = sorted([c for c in cols if c.startswith("y_")])
    rest = [c for c in cols if (c not in set(core)) and (c not in set(labels))]
    rest_sorted = sorted(rest)
    return core + rest_sorted + labels


def _compute_global_schema(files: Sequence[Path], mode: str) -> list[str]:
    mode = str(mode).lower().strip()
    if mode not in {"union", "intersection"}:
        raise ValueError(f"schema mode 仅支持 union/intersection：{mode}")

    col_sets: list[set[str]] = []
    for fp in files:
        cols = set(_read_header_cols(fp))
        col_sets.append(cols)

    if not col_sets:
        return []
    if mode == "union":
        merged: set[str] = set().union(*col_sets)
        return _canonical_order(merged)
    merged = set.intersection(*col_sets)
    return _canonical_order(merged)


def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.loc[df.index.notna()].copy()
    df.index.name = "datetime"
    return df


def _add_forward_targets(df: pd.DataFrame, *, horizon_hours: int) -> pd.DataFrame:
    if "close" not in df.columns:
        raise ValueError("缺少 close 列，无法构造目标")
    h = int(horizon_hours)
    if h <= 0:
        raise ValueError("horizon_hours 必须为正整数")
    logc = np.log(np.maximum(pd.to_numeric(df["close"], errors="coerce").astype("float64"), EPS))
    df[f"y_logret_fwd_{h}h"] = (logc.shift(-h) - logc)
    df[f"y_ret_fwd_{h}h"] = (df["close"].shift(-h) / df["close"] - 1.0)
    return df


def _apply_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    质量过滤（保守）：只保留“允许交易 + 已成熟 + 非明显异常”的样本。
    """
    if "is_valid_for_training" in df.columns:
        s = df["is_valid_for_training"]
        return df.loc[(s.fillna(0).astype("int64") == 1)].copy()

    m = pd.Series(True, index=df.index)
    for col, rule in (
        ("trade_allowed", lambda s: s == 1),
        ("is_mature", lambda s: s.astype(bool)),
        ("is_spike", lambda s: ~s.astype(bool)),
        ("is_volume_spike", lambda s: ~s.astype(bool)),
        ("cooldown_no_trade", lambda s: s == 0),
    ):
        if col in df.columns:
            s = df[col]
            m &= rule(s.fillna(0))
    return df.loc[m].copy()


def _impute(df: pd.DataFrame, *, policy: str, ffill_limit: Optional[int]) -> pd.DataFrame:
    policy = str(policy).lower().strip()
    if policy == "none":
        return df

    # 只对数值列做填充，避免把 datetime/object 搞脏
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if policy == "ffill":
        df[num_cols] = df[num_cols].ffill(limit=ffill_limit)
        return df
    if policy == "zero":
        df[num_cols] = df[num_cols].fillna(0.0)
        return df

    raise ValueError(f"未知 impute policy: {policy}（可选：none/ffill/zero）")


def prepare_one(
    fp: Path,
    *,
    horizon_hours: int,
    filter_quality: bool,
    impute_policy: str,
    ffill_limit: Optional[int],
    schema_cols: Optional[Sequence[str]] = None,
    train_end: Optional[pd.Timestamp] = None,
    val_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df = _to_utc_index(df)

    df = _add_forward_targets(df, horizon_hours=horizon_hours)
    # 去掉无法构造未来目标的尾部行
    df = df.dropna(subset=[f"y_logret_fwd_{int(horizon_hours)}h"]).copy()

    if filter_quality:
        df = _apply_quality_filter(df)

    if schema_cols is not None:
        # 统一输出 schema：缺失列补 NaN（后续由 impute 策略决定是否填充）
        df = df.reindex(columns=list(schema_cols))

    df = _impute(df, policy=impute_policy, ffill_limit=ffill_limit)
    # 避免 DataFrame 高度碎片化导致的性能警告（后续可能还会添加少量列）
    df = df.copy()

    # 可选：按时间打 split（便于 walk-forward / 时间切分）
    # 口径：优先使用 feature_time（若存在），否则使用 index(datetime)
    if train_end is not None or val_end is not None:
        if "feature_time" in df.columns:
            t = pd.to_datetime(df["feature_time"], utc=True, errors="coerce")
        else:
            t = df.index.to_series()

        split = pd.Series("test", index=df.index, dtype="object")
        if val_end is not None:
            split.loc[t <= val_end] = "val"
        if train_end is not None:
            split.loc[t <= train_end] = "train"
        df["split"] = split

    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="准备 AlphaGen 训练数据（无前视偏差）")
    ap.add_argument("--input-dir", default="AlphaQCM_data/final_dataset", help="输入目录（*_final.csv）")
    ap.add_argument("--output-dir", default="AlphaQCM_data/alphagen_ready", help="输出目录（*_train.csv）")
    ap.add_argument("--horizon-hours", type=int, default=1, help="预测视野（小时），默认 1")
    ap.add_argument("--filter-quality", action="store_true", help="过滤维护/异常/新币等低质量样本")
    ap.add_argument("--impute", default="none", help="缺失值处理：none/ffill/zero")
    ap.add_argument("--ffill-limit", type=int, default=24, help="ffill 的最大连续填充步数（小时），默认 24")
    ap.add_argument("--train-end", default="", help="训练集结束时间（UTC，例：2022-12-31 23:00:00+00:00）")
    ap.add_argument("--val-end", default="", help="验证集结束时间（UTC，例：2023-06-30 23:00:00+00:00）")
    ap.add_argument(
        "--schema",
        default="per-file",
        help="输出列 schema：per-file（默认，按币种原样）/ union（全币种并集）/ intersection（全币种交集）",
    )
    ap.add_argument(
        "--schema-out",
        default="",
        help="当 --schema=union|intersection 时，保存 schema 的 JSON 路径（默认写到输出目录：schema.json）",
    )
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个文件（0=不限制）")
    args = ap.parse_args()

    alphaqcm_root = Path(__file__).resolve().parents[1]
    input_dir = (alphaqcm_root / args.input_dir).resolve()
    output_dir = (alphaqcm_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*_final.csv"))
    if args.limit and args.limit > 0:
        files = files[: int(args.limit)]
    if not files:
        print(f"未找到输入文件：{input_dir}")
        return 2

    schema_mode = str(args.schema).lower().strip()
    schema_cols: Optional[list[str]] = None
    if schema_mode in {"union", "intersection"}:
        schema_cols = _compute_global_schema(files, schema_mode)
        if not schema_cols:
            raise RuntimeError(f"无法计算全局 schema：{schema_mode}")
        schema_out = str(args.schema_out).strip()
        if not schema_out:
            schema_out = str((output_dir / "schema.json").resolve())
        Path(schema_out).parent.mkdir(parents=True, exist_ok=True)
        with open(schema_out, "w", encoding="utf-8") as f:
            json.dump({"mode": schema_mode, "columns": schema_cols}, f, ensure_ascii=False, indent=2)
        print(f"全局 schema({schema_mode}) 列数：{len(schema_cols)} -> {schema_out}")

    train_end = pd.to_datetime(args.train_end, utc=True, errors="coerce") if str(args.train_end).strip() else None
    val_end = pd.to_datetime(args.val_end, utc=True, errors="coerce") if str(args.val_end).strip() else None

    for i, fp in enumerate(files, 1):
        symbol = fp.name.replace("_final.csv", "")
        out_fp = output_dir / f"{symbol}_train.csv"
        print(f"[{i}/{len(files)}] {symbol}...", end=" ")
        try:
            df = prepare_one(
                fp,
                horizon_hours=int(args.horizon_hours),
                filter_quality=bool(args.filter_quality),
                impute_policy=str(args.impute),
                ffill_limit=int(args.ffill_limit) if args.ffill_limit is not None else None,
                schema_cols=schema_cols,
                train_end=train_end,
                val_end=val_end,
            )
            df.to_csv(out_fp)
            print(f"✓ {len(df)} rows -> {out_fp}")
        except Exception as e:
            print(f"✗ {e}")

    print(f"\n完成：{len(files)} 个币种")
    print(f"输出目录：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
