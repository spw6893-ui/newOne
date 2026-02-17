#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控 Binance Vision aggTrades 全量下载/聚合进度（适配本仓库的目录结构）。

典型用法：
  python3 AlphaQCM/data_collection/monitor_aggtrades_progress.py --interval 15
  python3 AlphaQCM/data_collection/monitor_aggtrades_progress.py --once

输出内容（尽量低开销）：
- 3 个分片日志（partA/B/C）的：当前 symbol、当前月份、日志最后更新时间
- 产物统计：1m parquet 文件数/币种目录数、1h CSV 完成数、目录体积
- 临时下载目录中最近更新的 .zip / .zip.part 文件（用于判断是否“在下载”）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class PartStatus:
    name: str
    log_path: Path
    log_mtime: Optional[float]
    marker_text: str
    month_text: str
    last_saved_text: str


_RE_MARKER = re.compile(r"^\[(\d+)/(\d+)\]\s+([A-Z0-9_]+)\s*$")
_RE_QUEUE_CLAIM = re.compile(r"^\[[A-Za-z0-9_-]+\]\s+claim\s+([A-Z0-9_]+)\b")
_RE_MONTH = re.compile(r"^\s{2}(\d{4}-\d{2})\.\.\.\s*(.*)$")
_RE_SAVED = re.compile(r"^\s{2}Saved:\s+(.+)$")


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


def _tail_lines(path: Path, max_lines: int = 200, max_bytes: int = 256 * 1024) -> list[str]:
    """
    读取文件末尾若干行（避免整文件读入）。
    """
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - max_bytes, 0), os.SEEK_SET)
            data = f.read()
    except FileNotFoundError:
        return []
    except Exception:
        # 兜底：极端情况下不让监控脚本崩
        return []

    # 兼容可能的非 UTF-8 垃圾字符
    text = data.decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _parse_part_status(name: str, log_path: Path) -> PartStatus:
    lines = _tail_lines(log_path, max_lines=300)
    marker_text = "未发现"
    month_text = "未发现"
    last_saved_text = "未发现"

    for ln in reversed(lines):
        m = _RE_MARKER.match(ln.strip())
        if m:
            idx, total, sym = m.group(1), m.group(2), m.group(3)
            marker_text = f"{idx}/{total} {sym}"
            break

    # 队列 worker 日志（run_aggtrades_queue.py）：形如 “[A] claim ADAUSDT | ...”
    if marker_text == "未发现":
        for ln in reversed(lines):
            m = _RE_QUEUE_CLAIM.match(ln.strip())
            if m:
                marker_text = m.group(1)
                break

    for ln in reversed(lines):
        m = _RE_MONTH.match(ln)
        if m:
            month = m.group(1)
            tail = m.group(2).strip()
            if tail == "":
                month_text = f"{month} (处理中)"
            elif "✓" in tail or "hours" in tail:
                month_text = f"{month} (完成)"
            elif "✗" in tail:
                month_text = f"{month} (缺失/失败)"
            else:
                month_text = f"{month} ({tail[:60]})"
            break

    for ln in reversed(lines):
        m = _RE_SAVED.match(ln)
        if m:
            last_saved_text = Path(m.group(1).strip()).name
            break

    mtime = _safe_stat_mtime(log_path)
    return PartStatus(
        name=name,
        log_path=log_path,
        log_mtime=mtime,
        marker_text=marker_text,
        month_text=month_text,
        last_saved_text=last_saved_text,
    )


@dataclass(frozen=True)
class OutputStats:
    parquet_files: int
    parquet_symbol_dirs: int
    hourly_csv: int
    parquet_size_bytes: int
    hourly_size_bytes: int
    newest_parquet_mtime: Optional[float]


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for root, _, files in os.walk(path):
            for fn in files:
                try:
                    total += (Path(root) / fn).stat().st_size
                except FileNotFoundError:
                    pass
    except FileNotFoundError:
        return 0
    return total


def _human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(v)}{u}"
            return f"{v:.1f}{u}"
        v /= 1024.0
    return f"{v:.1f}TiB"


def _scan_outputs(parquet_root: Path, hourly_root: Path) -> OutputStats:
    parquet_files = 0
    parquet_symbol_dirs = 0
    newest_mtime: Optional[float] = None

    if parquet_root.exists():
        for ent in parquet_root.iterdir():
            if not ent.is_dir():
                continue
            parquet_symbol_dirs += 1
            try:
                for f in ent.iterdir():
                    if f.is_file() and f.name.endswith(".parquet"):
                        parquet_files += 1
                        try:
                            mt = f.stat().st_mtime
                            if newest_mtime is None or mt > newest_mtime:
                                newest_mtime = mt
                        except FileNotFoundError:
                            pass
            except FileNotFoundError:
                pass

    hourly_csv = 0
    if hourly_root.exists():
        try:
            for f in hourly_root.iterdir():
                if f.is_file() and f.name.endswith("_aggTrades.csv"):
                    hourly_csv += 1
        except FileNotFoundError:
            pass

    parquet_size = _dir_size_bytes(parquet_root)
    hourly_size = _dir_size_bytes(hourly_root)
    return OutputStats(
        parquet_files=parquet_files,
        parquet_symbol_dirs=parquet_symbol_dirs,
        hourly_csv=hourly_csv,
        parquet_size_bytes=parquet_size,
        hourly_size_bytes=hourly_size,
        newest_parquet_mtime=newest_mtime,
    )


def _list_recent_tmp_files(tmp_dir: Path, limit: int = 10) -> list[tuple[float, Path]]:
    items: list[tuple[float, Path]] = []
    if not tmp_dir.exists():
        return items
    try:
        for f in tmp_dir.iterdir():
            if not f.is_file():
                continue
            if not (f.name.endswith(".zip") or f.name.endswith(".zip.part")):
                continue
            try:
                items.append((f.stat().st_mtime, f))
            except FileNotFoundError:
                continue
    except FileNotFoundError:
        return items
    items.sort(key=lambda x: x[0], reverse=True)
    return items[:limit]


def _fmt_time_local(ts: Optional[float]) -> str:
    if ts is None:
        return "未知"
    dt = datetime.fromtimestamp(ts).astimezone()
    return dt.strftime("%m-%d %H:%M:%S")


def _clear_screen() -> None:
    # 尽量兼容：支持无 tty 的场景
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")


def _print_snapshot(
    parts: Iterable[PartStatus],
    out: OutputStats,
    tmp_files: list[tuple[float, Path]],
    prev: Optional[OutputStats],
    interval_sec: int,
    stall_warn_sec: int,
    queue_state_path: Optional[Path],
) -> None:
    now = time.time()
    _clear_screen()

    print(f"[{_now_local_str()}] aggTrades 下载/聚合监控")
    print("-" * 72)
    for p in parts:
        age = None if p.log_mtime is None else max(now - p.log_mtime, 0.0)
        print(
            f"{p.name}: {p.marker_text:>14} | {p.month_text:<16} | "
            f"log更新时间 {_fmt_time_local(p.log_mtime)} (距今 {_fmt_age(age)}) | 最近Saved {p.last_saved_text}"
        )

    print("-" * 72)
    delta_parquet = None
    delta_csv = None
    if prev is not None:
        delta_parquet = out.parquet_files - prev.parquet_files
        delta_csv = out.hourly_csv - prev.hourly_csv
    rate_text = ""
    if delta_parquet is not None and interval_sec > 0:
        rate = delta_parquet / (interval_sec / 60.0)
        rate_text = f" | parquet增速 {rate:.1f}/min"

    newest_age = None if out.newest_parquet_mtime is None else max(now - out.newest_parquet_mtime, 0.0)
    print(
        f"产物: 1m parquet={out.parquet_files} (symbol_dir={out.parquet_symbol_dirs}, { _human_bytes(out.parquet_size_bytes) })"
        f" | 1h CSV={out.hourly_csv} ({ _human_bytes(out.hourly_size_bytes) })"
        f" | 最新parquet {_fmt_time_local(out.newest_parquet_mtime)} (距今 {_fmt_age(newest_age)})"
        f"{rate_text}"
    )

    # 简单卡顿告警：产物无增长 + parquet 最新写入时间太久
    stalled = False
    if newest_age is not None and newest_age >= stall_warn_sec:
        if prev is not None and prev.parquet_files == out.parquet_files and prev.hourly_csv == out.hourly_csv:
            stalled = True
    if stalled:
        print(f"告警: {stall_warn_sec}s 内无新 parquet/CSV 写入，可能卡在下载/解压/聚合或网络抖动。")

    print("-" * 72)
    if not tmp_files:
        print("临时目录: 未发现 .zip/.zip.part")
    else:
        print("临时目录(最近更新):")
        for mt, p in tmp_files:
            try:
                size = p.stat().st_size
            except FileNotFoundError:
                continue
            age = max(now - mt, 0.0)
            print(f"  {_fmt_time_local(mt)} (距今 {_fmt_age(age)})  { _human_bytes(size):>8}  {p.name}")

    if queue_state_path and queue_state_path.exists():
        try:
            state = json.loads(queue_state_path.read_text(encoding="utf-8"))
            items = state.get("items") or {}
            cnt = {"pending": 0, "in_progress": 0, "done": 0, "failed": 0}
            for it in items.values():
                st = (it.get("status") or "pending").strip()
                cnt[st] = cnt.get(st, 0) + 1
            print("-" * 72)
            print(
                f"队列: pending={cnt.get('pending', 0)} in_progress={cnt.get('in_progress', 0)} "
                f"done={cnt.get('done', 0)} failed={cnt.get('failed', 0)} | state={queue_state_path}"
            )
        except Exception:
            pass

    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description="监控 Binance Vision aggTrades 下载/聚合进度（本仓库专用）")
    ap.add_argument("--interval", type=int, default=15, help="刷新间隔（秒），默认 15")
    ap.add_argument("--once", action="store_true", help="只输出一次然后退出")
    ap.add_argument("--stall-warn-sec", type=int, default=15 * 60, help="多少秒无新 parquet/CSV 产出则提示告警")
    ap.add_argument(
        "--root",
        type=str,
        default=".",
        help="仓库根目录（默认当前目录）。一般不需要改。",
    )
    ap.add_argument(
        "--state",
        type=str,
        default="",
        help="可选：队列模式 state 文件（run_aggtrades_queue.py）。例如 AlphaQCM_data/_tmp/aggTrades_queue_state.json",
    )
    args = ap.parse_args()

    repo_root = Path(args.root).resolve()
    alphaqcm_data = repo_root / "AlphaQCM" / "AlphaQCM_data"

    logs_dir = alphaqcm_data / "_tmp" / "logs"
    tmp_dir = alphaqcm_data / "_tmp" / "binance_vision"
    parquet_root = alphaqcm_data / "binance_aggTrades_1m_parquet"
    hourly_root = alphaqcm_data / "binance_aggTrades"

    queue_state_path: Optional[Path] = None
    if args.state:
        s = args.state
        if s.startswith("AlphaQCM_data/") or s == "AlphaQCM_data":
            # 允许用 AlphaQCM_data/... 这种短路径
            queue_state_path = alphaqcm_data / s.replace("AlphaQCM_data/", "")
        else:
            pth = Path(s)
            queue_state_path = (pth if pth.is_absolute() else (repo_root / pth)).resolve()

    # 日志来源：默认读老的 partA/B/C；若启用队列模式（传了 --state），优先读 queue worker 的日志
    if args.state:
        parts = [
            ("workerA", logs_dir / "aggTrades_queue_A.log"),
            ("workerB", logs_dir / "aggTrades_queue_B.log"),
            ("workerC", logs_dir / "aggTrades_queue_C.log"),
        ]
    else:
        parts = [
            ("partA", logs_dir / "aggTrades_partA.log"),
            ("partB", logs_dir / "aggTrades_partB.log"),
            ("partC", logs_dir / "aggTrades_partC.log"),
        ]

    prev: Optional[OutputStats] = None
    interval = max(int(args.interval), 1)

    while True:
        part_status = [_parse_part_status(name, p) for name, p in parts]
        out = _scan_outputs(parquet_root, hourly_root)
        tmp_files = _list_recent_tmp_files(tmp_dir, limit=10)
        _print_snapshot(
            parts=part_status,
            out=out,
            tmp_files=tmp_files,
            prev=prev,
            interval_sec=interval,
            stall_warn_sec=int(args.stall_warn_sec),
            queue_state_path=queue_state_path,
        )
        if args.once:
            return 0
        prev = out
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
