#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控 1m->1h 波动率因子聚合进度（aggregate_volatility_factors.py）。

典型用法：
  python3 AlphaQCM/data_collection/monitor_volatility_progress.py --interval 10
  python3 AlphaQCM/data_collection/monitor_volatility_progress.py --once

输出（低开销）：
- 日志最后更新时间、当前处理到的 symbol
- 输出目录中 *_volatility.csv 的数量
- 已包含 doc_* 新字段的文件数量（通过表头是否含 doc_kurt 判断）
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LogStatus:
    log_path: Path
    log_mtime: Optional[float]
    marker_text: str
    last_line: str


_RE_MARKER = re.compile("^\\[(\\d+)/(\\d+)\\]\\s+(.+?)\\.\\.\\.")


def _now_local_str() -> str:
    dt = datetime.now().astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def _fmt_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "未知"
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    return f"{seconds / 3600:.1f}h"


def _safe_stat_mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _tail_lines(path: Path, max_lines: int = 200, max_bytes: int = 128 * 1024) -> list[str]:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - max_bytes, 0), os.SEEK_SET)
            data = f.read()
    except FileNotFoundError:
        return []
    except Exception:
        return []

    text = data.decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _parse_log_status(log_path: Path) -> LogStatus:
    lines = _tail_lines(log_path, max_lines=200)
    marker_text = "未发现"
    last_line = lines[-1].strip() if lines else ""
    for ln in reversed(lines):
        m = _RE_MARKER.match(ln.strip())
        if m:
            idx, total, sym = m.group(1), m.group(2), m.group(3).strip()
            marker_text = f"{idx}/{total} {sym}"
            break
    return LogStatus(
        log_path=log_path,
        log_mtime=_safe_stat_mtime(log_path),
        marker_text=marker_text,
        last_line=last_line,
    )


def _count_outputs(output_dir: Path) -> tuple[int, int, Optional[Path], Optional[float]]:
    """
    返回：总文件数、包含 doc_kurt 的文件数、最新文件路径、最新 mtime
    """
    total = 0
    has_doc = 0
    newest_path: Optional[Path] = None
    newest_mtime: Optional[float] = None

    if not output_dir.exists():
        return 0, 0, None, None

    for fp in output_dir.glob("*_volatility.csv"):
        if not fp.is_file():
            continue
        total += 1
        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                header = f.readline()
            if "doc_kurt" in header:
                has_doc += 1
        except Exception:
            pass

        try:
            mt = fp.stat().st_mtime
            if newest_mtime is None or mt > newest_mtime:
                newest_mtime = mt
                newest_path = fp
        except FileNotFoundError:
            pass

    return total, has_doc, newest_path, newest_mtime


def _print_once(log_path: Path, output_dir: Path) -> None:
    st = _parse_log_status(log_path)
    now = time.time()
    age = _fmt_age(now - st.log_mtime) if st.log_mtime is not None else "未知"

    total, has_doc, newest_path, newest_mtime = _count_outputs(output_dir)
    newest_age = _fmt_age(now - newest_mtime) if newest_mtime is not None else "未知"
    newest_name = newest_path.name if newest_path is not None else "无"

    print(f"[{_now_local_str()}]")
    print(f"- 日志：{st.log_path} (更新: {age} 前) | 当前：{st.marker_text}")
    if st.last_line:
        print(f"- 日志末行：{st.last_line[:120]}")
    print(f"- 输出：{output_dir} | 文件: {total} | 已含 doc_*: {has_doc} | 最新: {newest_name} ({newest_age} 前)")


def main() -> int:
    ap = argparse.ArgumentParser(description="监控波动率因子聚合进度（1m->1h）")
    ap.add_argument("--log", default="AlphaQCM_data/_tmp/logs/volatility_agg_doc.log", help="聚合日志路径")
    ap.add_argument("--output-dir", default="AlphaQCM_data/crypto_hourly_volatility", help="输出目录")
    ap.add_argument("--interval", type=int, default=15, help="轮询间隔（秒）")
    ap.add_argument("--once", action="store_true", help="只输出一次后退出")
    args = ap.parse_args()

    alphaqcm_root = Path(__file__).resolve().parents[1]
    log_path = (alphaqcm_root / args.log).resolve()
    output_dir = (alphaqcm_root / args.output_dir).resolve()

    if args.once:
        _print_once(log_path, output_dir)
        return 0

    while True:
        _print_once(log_path, output_dir)
        time.sleep(max(1, int(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
