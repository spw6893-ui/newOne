#!/usr/bin/env python3
"""
从 AlphaGen 训练过程导出的 expr_trials.jsonl 训练一个轻量代理模型（线性回归 + Adagrad），
用于预测 abs(single_ic)，从而在训练时做候选筛选/重排。

设计目标：
- 纯 numpy 实现（尽量不引入额外依赖）
- 训练速度快，可在服务器上随时增量重训
- 特征工程简单但稳定：基于表达式字符串提取 op/feature/window/dt/长度等离散特征
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


_RE_OP = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(")
_RE_FEAT = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_RE_INT = re.compile(r"(?:,|\()\s*(\d{1,5})\s*(?:\)|,)")


def _stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def expr_to_feature_ids(expr: str, dim: int) -> List[int]:
    """
    将表达式字符串转为稀疏离散特征（hash trick），返回索引列表（可能含重复，会去重）。
    """
    ops = _RE_OP.findall(expr)
    feats = _RE_FEAT.findall(expr)
    ints = _RE_INT.findall(expr)

    # 基础特征
    f: List[str] = []
    f.append(f"len:{min(64, max(1, len(expr)//8))}")  # 粗粒度长度桶
    f.append(f"ops_cnt:{min(32, len(ops))}")
    f.append(f"feats_cnt:{min(32, len(feats))}")

    # 具体 op / feature
    for op in ops[:64]:
        f.append(f"op:{op}")
    for ft in feats[:64]:
        f.append(f"feat:{ft}")

    # 常见窗口/滞后参数（截断）
    for x in ints[:64]:
        f.append(f"int:{x}")

    # 二元交叉（低成本、提高表达力）
    for op in ops[:8]:
        for ft in feats[:8]:
            f.append(f"x:{op}|{ft}")

    idx = [int(_stable_hash(s) % dim) for s in f]
    # 去重
    return sorted(set(idx))


def iter_trials(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


@dataclass
class SurrogateModel:
    dim: int
    w: np.ndarray
    g2: np.ndarray
    bias: float

    def score(self, expr: str) -> float:
        ids = expr_to_feature_ids(expr, self.dim)
        if not ids:
            return float(self.bias)
        return float(self.bias + float(self.w[ids].sum()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, dim=np.array([self.dim], dtype=np.int32), w=self.w, g2=self.g2, bias=np.array([self.bias], dtype=np.float32))


def train(
    items: Iterable[Tuple[List[int], float]],
    dim: int,
    epochs: int,
    lr: float,
    l2: float,
) -> SurrogateModel:
    w = np.zeros((dim,), dtype=np.float32)
    g2 = np.zeros((dim,), dtype=np.float32)
    bias = np.float32(0.0)
    bias_g2 = np.float32(0.0)

    data = list(items)
    if not data:
        raise RuntimeError("没有可用样本：请确认 jsonl 中存在 single_ic 字段且为有限数。")

    for ep in range(int(epochs)):
        np.random.shuffle(data)
        for ids, y in data:
            # 预测：线性和
            pred = float(bias + (float(w[ids].sum()) if ids else 0.0))
            err = pred - float(y)

            # bias 更新（Adagrad）
            g = err
            bias_g2 = np.float32(bias_g2 + g * g)
            bias = np.float32(bias - (lr / (math.sqrt(float(bias_g2)) + 1e-8)) * g)

            if not ids:
                continue

            # 权重更新（稀疏 Adagrad + L2）
            for j in ids:
                gj = err + l2 * float(w[j])
                g2[j] = np.float32(g2[j] + gj * gj)
                w[j] = np.float32(w[j] - (lr / (math.sqrt(float(g2[j])) + 1e-8)) * gj)

    return SurrogateModel(dim=dim, w=w, g2=g2, bias=float(bias))


def main() -> int:
    ap = argparse.ArgumentParser(description="Train surrogate model for abs(single_ic) from expr_trials.jsonl")
    ap.add_argument("--input", required=True, help="expr_trials.jsonl 路径")
    ap.add_argument("--output", required=True, help="输出 .npz（例如 alphagen_output/surrogate_model.npz）")
    ap.add_argument("--dim", type=int, default=65536, help="hash 向量维度（默认 65536）")
    ap.add_argument("--epochs", type=int, default=3, help="训练轮数（默认 3）")
    ap.add_argument("--lr", type=float, default=0.3, help="Adagrad 学习率（默认 0.3）")
    ap.add_argument("--l2", type=float, default=1e-6, help="L2 正则（默认 1e-6）")
    ap.add_argument("--min_abs_ic", type=float, default=0.0, help="过滤 abs(single_ic) 过小样本（默认不过滤）")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise FileNotFoundError(f"未找到输入：{inp}")

    dim = int(args.dim)
    min_abs = float(args.min_abs_ic)

    samples: List[Tuple[List[int], float]] = []
    n_total = 0
    n_kept = 0
    for row in iter_trials(inp):
        n_total += 1
        expr = str(row.get("expr", "") or "")
        single_ic = row.get("single_ic", None)
        if single_ic is None:
            continue
        try:
            v = float(single_ic)
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        y = abs(v)
        if y < min_abs:
            continue
        ids = expr_to_feature_ids(expr, dim)
        samples.append((ids, float(y)))
        n_kept += 1

    print(f"Loaded trials: total={n_total}, kept={n_kept}, dim={dim}")
    m = train(samples, dim=dim, epochs=int(args.epochs), lr=float(args.lr), l2=float(args.l2))
    m.save(out)
    print(f"Saved surrogate model: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
