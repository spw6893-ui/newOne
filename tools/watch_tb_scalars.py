#!/usr/bin/env python3
"""
在“无法打开 TensorBoard 网页”的服务器环境下，用终端实时查看关键标量。

特性：
- 既支持直接指定 event 文件，也支持指定目录（自动递归找最新的 events.out.tfevents.*）
- 默认只看两个指标：eval/val_ic 与 eval/test_ic（可用 --tags 自定义）
- 仅当 step 更新时才打印，避免刷屏

示例：
  python3 tools/watch_tb_scalars.py --path alphagen_output/tensorboard
  python3 tools/watch_tb_scalars.py --path . --tags eval/val_ic eval/test_ic --every 5
  python3 tools/watch_tb_scalars.py --path /home/ppw/CryptoQuant/events.out.tfevents.xxx --once
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _iter_event_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for p in root.rglob("events.out.tfevents.*"):
        if p.is_file():
            yield p


def _pick_latest_event_file(root: Path) -> Optional[Path]:
    files = list(_iter_event_files(root))
    if not files:
        return None
    files.sort(key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)
    return files[0]


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    value: float


def _load_last_scalars(event_fp: Path, tags: list[str]) -> dict[str, ScalarPoint]:
    # 延迟导入，避免在没有 tensorboard 的环境里直接崩
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(
        str(event_fp),
        size_guidance={"scalars": 0},
    )
    ea.Reload()
    avail = set(ea.Tags().get("scalars", []))
    out: dict[str, ScalarPoint] = {}
    for t in tags:
        if t not in avail:
            continue
        items = ea.Scalars(t)
        if not items:
            continue
        last = items[-1]
        out[t] = ScalarPoint(step=int(last.step), value=float(last.value))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        required=True,
        help="event 文件路径，或包含 event 文件的目录（会递归查找最新）。",
    )
    ap.add_argument(
        "--tags",
        nargs="+",
        default=["eval/val_ic", "eval/test_ic"],
        help="要显示的标量 tag 列表（默认：eval/val_ic eval/test_ic）。",
    )
    ap.add_argument(
        "--every",
        type=float,
        default=10.0,
        help="轮询间隔（秒）。默认 10s。",
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="只读取一次并退出。",
    )
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.path)).resolve()
    tags = [str(t).strip() for t in args.tags if str(t).strip()]
    if not tags:
        print("❌ --tags 不能为空", file=sys.stderr)
        return 2

    last_print_key: Optional[tuple[int, ...]] = None
    current_event: Optional[Path] = None

    while True:
        event_fp = _pick_latest_event_file(root)
        if event_fp is None:
            print(f"[{_now_str()}] ❌ 未找到 event 文件：{root}", file=sys.stderr)
            return 2

        if current_event != event_fp:
            current_event = event_fp
            last_print_key = None
            print(f"[{_now_str()}] ✅ 使用 event 文件：{current_event}")

        try:
            scalars = _load_last_scalars(current_event, tags)
        except Exception as e:
            print(f"[{_now_str()}] ⚠ 读取失败：{e}", file=sys.stderr)
            if args.once:
                return 1
            time.sleep(max(1.0, float(args.every)))
            continue

        # 用 tags 对应的 step 组合做“是否更新”的判据
        key = tuple(scalars.get(t, ScalarPoint(step=-1, value=float("nan"))).step for t in tags)
        if key != last_print_key:
            last_print_key = key
            parts = []
            for t in tags:
                p = scalars.get(t)
                if p is None:
                    parts.append(f"{t}=NA")
                else:
                    parts.append(f"{t}={p.value:.6g}@{p.step}")
            print(f"[{_now_str()}] " + "  ".join(parts))

        if args.once:
            return 0
        time.sleep(max(0.5, float(args.every)))


if __name__ == "__main__":
    raise SystemExit(main())

