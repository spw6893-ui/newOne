#!/usr/bin/env python3
"""
离线读取 TensorBoard event 文件并打印最新标量。

用途：
- 无法访问 TensorBoard 网页时，用这个脚本在服务器命令行查看训练进度。

示例：
  python3 AlphaQCM/tools/tb_tail.py
  python3 AlphaQCM/tools/tb_tail.py --log-root AlphaQCM/AlphaQCM_data/crypto_logs
  python3 AlphaQCM/tools/tb_tail.py --tags ic/train ic/valid ic/test
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class ScalarPoint:
    tag: str
    step: int
    value: float


def _find_latest_event_dir(log_root: str) -> str:
    pattern = os.path.join(log_root, "**", "summary", "events.out.tfevents.*")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"未找到 event 文件：{pattern}")
    latest = max(files, key=os.path.getmtime)
    return os.path.dirname(latest)


def _load_latest_scalars(event_dir: str, tags: Optional[Iterable[str]]) -> List[ScalarPoint]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "无法导入 tensorboard 的 EventAccumulator。请先安装/修复 tensorboard："
            "python3 -m pip install -U tensorboard setuptools"
        ) from e

    ea = EventAccumulator(event_dir)
    ea.Reload()
    scalar_tags = set(ea.Tags().get("scalars", []))

    wanted = list(tags) if tags else sorted(scalar_tags)
    out: List[ScalarPoint] = []
    for tag in wanted:
        if tag not in scalar_tags:
            continue
        points = ea.Scalars(tag)
        if not points:
            continue
        p = points[-1]
        out.append(ScalarPoint(tag=tag, step=int(p.step), value=float(p.value)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        default="AlphaQCM/AlphaQCM_data/crypto_logs",
        help="AlphaQCM 训练日志根目录（默认：AlphaQCM/AlphaQCM_data/crypto_logs）",
    )
    parser.add_argument(
        "--event-dir",
        default="",
        help="直接指定包含 events.out.tfevents.* 的 summary 目录；指定后会忽略 --log-root 的自动查找。",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=["ic/train", "ic/valid", "ic/test", "loss/quantile_loss", "loss/mse_loss", "stats/mean_Q"],
        help="要打印的标量 tag 列表；为空则打印全部标量 tag 的最新值。",
    )
    args = parser.parse_args()

    if args.event_dir:
        event_dir = args.event_dir
    else:
        event_dir = _find_latest_event_dir(args.log_root)

    print(f"event_dir: {event_dir}")
    pts = _load_latest_scalars(event_dir, tags=args.tags)
    if not pts:
        print("未找到可用标量（可能还没写入 summary，或 tags 不存在）。")
        return 0

    for p in pts:
        print(f"{p.tag}: step={p.step} value={p.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

