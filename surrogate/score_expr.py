#!/usr/bin/env python3
"""
用训练好的 surrogate_model.npz 给表达式打分（预测 abs(single_ic)），并输出 TopK。

输入支持：
- 纯文本：每行一个表达式
- pool json：alphagen 的 alpha_pool.json（读取 exprs）
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


_RE_OP = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(")
_RE_FEAT = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_RE_INT = re.compile(r"(?:,|\()\s*(\d{1,5})\s*(?:\)|,)")


def _stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def expr_to_feature_ids(expr: str, dim: int) -> List[int]:
    ops = _RE_OP.findall(expr)
    feats = _RE_FEAT.findall(expr)
    ints = _RE_INT.findall(expr)

    f: List[str] = []
    f.append(f"len:{min(64, max(1, len(expr)//8))}")
    f.append(f"ops_cnt:{min(32, len(ops))}")
    f.append(f"feats_cnt:{min(32, len(feats))}")
    for op in ops[:64]:
        f.append(f"op:{op}")
    for ft in feats[:64]:
        f.append(f"feat:{ft}")
    for x in ints[:64]:
        f.append(f"int:{x}")
    for op in ops[:8]:
        for ft in feats[:8]:
            f.append(f"x:{op}|{ft}")
    idx = [int(_stable_hash(s) % dim) for s in f]
    return sorted(set(idx))


def load_model(path: Path):
    d = np.load(path, allow_pickle=False)
    dim = int(d["dim"][0])
    w = d["w"].astype(np.float32, copy=False)
    bias = float(d["bias"][0])
    return dim, w, bias


def score_exprs(exprs: Iterable[str], dim: int, w: np.ndarray, bias: float) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    for e in exprs:
        e = (e or "").strip()
        if not e:
            continue
        ids = expr_to_feature_ids(e, dim)
        s = float(bias + (float(w[ids].sum()) if ids else 0.0))
        out.append((s, e))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def read_exprs(path: Path) -> List[str]:
    if path.suffix.lower() in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        exprs = data.get("exprs") or []
        # alphagen 的 to_json_dict 存的是字符串
        return [str(x) for x in exprs if x]
    # 默认按文本行读取
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Score expressions with surrogate model")
    ap.add_argument("--model", required=True, help="surrogate_model.npz")
    ap.add_argument("--input", required=True, help="表达式文本/alpha_pool.json")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    dim, w, bias = load_model(Path(args.model))
    exprs = read_exprs(Path(args.input))
    scored = score_exprs(exprs, dim, w, bias)[: int(args.topk)]

    for s, e in scored:
        print(f"{s:.6f}\t{e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
