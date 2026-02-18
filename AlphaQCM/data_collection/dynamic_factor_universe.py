#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态因子池（Dynamic Factor Universe）工具：当某些因子在早期历史中不可用（NaN）时，
按“可用因子集合”动态重归一化权重，保证不同时期的分数可比。

典型场景（你当前最关心的）：
- OI / OI_delta 类因子从较晚日期才开始有值（Binance Vision metrics 起始较晚）
- 你希望在 OI 不可用的时期，模型/打分只使用其它因子，并把权重重新归一化

输出：
- 对每个 *_final.csv 生成一个带 score 的新文件（默认不覆盖原文件）

权重归一化口径（默认）：
- score = sum(w_i * x_i) / sum(|w_i|)（仅在 x_i 非 NaN 的因子上求和）
  这样在某些因子缺失时，分母会自动变小，保证分数尺度在不同历史阶段更可比。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_weights(path: Path) -> dict[str, float]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("weights 文件必须是 JSON object：{col: weight, ...}")
    out: dict[str, float] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = float(v)
        except Exception as e:
            raise ValueError(f"非法权重：{k}={v} ({e})")
    if not out:
        raise ValueError("weights 为空")
    return out


def compute_dynamic_score(
    df: pd.DataFrame,
    *,
    weights: dict[str, float],
    normalize: str = "sum_abs",
    score_col: str = "score_dyn",
    denom_col: str = "score_dyn_wsum",
    used_col: str = "score_dyn_n",
) -> pd.DataFrame:
    """
    对 df 计算动态因子池分数：
    - 对每行，只有非 NaN 的因子参与 sum
    - 权重按 normalize 口径在“可用因子集合”上归一化
    """
    cols = [c for c in weights.keys() if c in df.columns]
    if not cols:
        raise ValueError("weights 中没有任何列存在于输入 df")

    w = np.array([weights[c] for c in cols], dtype="float64")
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    avail = x.notna().to_numpy(dtype=bool, copy=False)
    x0 = x.fillna(0.0).to_numpy(dtype="float64", copy=False)

    numer = (x0 * w.reshape(1, -1)).sum(axis=1)

    normalize = str(normalize).strip().lower()
    if normalize == "sum_abs":
        denom = (avail * np.abs(w).reshape(1, -1)).sum(axis=1)
    elif normalize == "sum":
        # 仅建议在所有权重均为正、且你希望“总权重=1”的语义时使用
        denom = (avail * w.reshape(1, -1)).sum(axis=1)
    else:
        raise ValueError("normalize 仅支持 sum_abs / sum")

    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(denom > 0, numer / denom, np.nan)

    out = df.copy()
    out[score_col] = score.astype("float64")
    out[denom_col] = denom.astype("float64")
    out[used_col] = avail.sum(axis=1).astype("int16")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="动态因子池（缺失因子权重自动重归一化）")
    ap.add_argument("--input-dir", required=True, help="输入目录：包含 *_final.csv")
    ap.add_argument("--output-dir", required=True, help="输出目录：写 *_final_dyn.csv")
    ap.add_argument("--weights", required=True, help="权重 JSON 文件：{col: weight, ...}")
    ap.add_argument("--normalize", default="sum_abs", choices=["sum_abs", "sum"])
    ap.add_argument("--suffix", default="_final_dyn.csv", help="输出文件后缀（默认 _final_dyn.csv）")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个文件（0=不限制）")
    args = ap.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    w_path = Path(args.weights)
    if not inp.exists():
        raise FileNotFoundError(f"未找到 input-dir：{inp}")
    if not w_path.exists():
        raise FileNotFoundError(f"未找到 weights：{w_path}")
    out.mkdir(parents=True, exist_ok=True)

    weights = load_weights(w_path)

    files = sorted(inp.glob("*_final.csv"))
    if args.limit and int(args.limit) > 0:
        files = files[: int(args.limit)]
    if not files:
        print(f"未找到 *_final.csv：{inp}")
        return 2

    for i, fp in enumerate(files, 1):
        sym = fp.name[: -len("_final.csv")]
        dst = out / f"{sym}{args.suffix}"
        print(f"[{i}/{len(files)}] {sym}...", end=" ")
        df = pd.read_csv(fp)
        df2 = compute_dynamic_score(df, weights=weights, normalize=args.normalize)
        df2.to_csv(dst, index=False)
        print(f"✓ -> {dst.name}")

    print(f"\n完成：{len(files)} 个币种")
    print(f"输出目录：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

