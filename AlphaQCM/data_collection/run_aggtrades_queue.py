"""
多 worker（多进程 / 多 tmux）共享队列：并发跑 aggTrades 全量下载 + 1m Parquet 中间层 + 1h 因子输出。

为什么需要它？
- 过去用 --symbols-offset/--symbols-limit 手动分片：某个 part 跑完后会闲置，且很难动态“分担”卡住的部分。
- 这个脚本用一个 state 文件做队列，多个 worker 竞争领取下一个 pending symbol，谁空闲谁继续干活。
- 断点可恢复：机器挂了/进程被 kill 后，超过一定时间仍处于 in_progress 的任务会自动回到 pending。
- 不重复下载：底层 download_symbol_range 已支持 --skip-existing（meta 完成标记）+ 1m parquet 复用。

建议用法（3 个 worker 并发，日志分开写）：
  stdbuf -oL -eL python3 -u AlphaQCM/data_collection/run_aggtrades_queue.py \\
    --worker-id A \\
    --symbols-file AlphaQCM/data_collection/top100_perp_symbols.txt \\
    --start 2020-01-01 --end 2025-02-15 \\
    --state AlphaQCM_data/_tmp/aggTrades_queue_state.json \\
    --aggtrades-1m-dir AlphaQCM_data/binance_aggTrades_1m_parquet \\
    --output-dir AlphaQCM_data/binance_aggTrades \\
    --skip-existing |& tee -a AlphaQCM/AlphaQCM_data/_tmp/logs/aggTrades_queue_A.log
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import fcntl

from download_all_binance_archive import load_symbols
from download_binance_efficient import download_symbol_range


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _maybe_rewrite_alphaqcm_data_path(p: str) -> str:
    """
    与 download_binance_efficient.py 的路径重写语义保持一致：
    - 允许传 AlphaQCM_data/...（相对 repo 根）；
    - 实际落到 AlphaQCM/AlphaQCM_data/...。
    """
    if p.startswith("AlphaQCM_data/") or p == "AlphaQCM_data":
        return str(Path("AlphaQCM") / p)
    return p


@dataclass(frozen=True)
class QueueConfig:
    symbols_file: str
    market: str
    start: str
    end: str
    output_dir: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _reconcile_done_by_meta(state: dict[str, Any], cfg: QueueConfig, skip_existing: bool) -> None:
    """
    若 output 的 meta 标记已完成，则把 state 标记为 done。
    用途：state 丢失/重建时，避免重复 claim 已完成 symbols。
    """
    if not skip_existing:
        return

    out_dir = Path(_maybe_rewrite_alphaqcm_data_path(cfg.output_dir))
    for s in state.get("symbols", []):
        meta_path = out_dir / f"{s}_aggTrades.csv.meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("status") == "ok"
                and meta.get("symbol") == s
                and meta.get("data_type") == "aggTrades"
                and meta.get("market") == cfg.market
                and meta.get("requested_start") == cfg.start
                and meta.get("requested_end") == cfg.end
            ):
                item = state["items"].get(s, {})
                item.update({"status": "done", "updated_at_utc": _utc_now_iso(), "worker": "reconcile", "note": "meta已完成"})
                state["items"][s] = item
        except Exception:
            continue


def _ensure_state(path: Path, cfg: QueueConfig, symbols: list[str], skip_existing: bool) -> dict[str, Any]:
    """
    初始化/复用 state：
    - 若文件不存在：创建。
    - 若存在且配置一致：复用（并做一次 reconcile）。
    - 若存在但配置不一致：备份为 .bak.* 再创建新 state。
    """
    if path.exists():
        try:
            state = _load_json(path)
            if state.get("config") == cfg.__dict__ and state.get("symbols") == symbols:
                state.setdefault("items", {})
                for s in symbols:
                    state["items"].setdefault(
                        s,
                        {"status": "pending", "updated_at_utc": None, "worker": None, "note": None},
                    )
                _reconcile_done_by_meta(state, cfg, skip_existing=skip_existing)
                return state
        except Exception:
            pass

        bak = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        try:
            os.replace(path, bak)
        except Exception:
            pass

    state = {
        "created_at_utc": _utc_now_iso(),
        "config": cfg.__dict__,
        "symbols": symbols,
        "items": {
            s: {"status": "pending", "updated_at_utc": None, "worker": None, "note": None}
            for s in symbols
        },
    }
    _reconcile_done_by_meta(state, cfg, skip_existing=skip_existing)
    return state


def _reset_stale_in_progress(state: dict[str, Any], stale_hours: float) -> int:
    now = datetime.now(tz=timezone.utc)
    reset = 0
    for _, item in (state.get("items") or {}).items():
        if item.get("status") != "in_progress":
            continue
        ts = item.get("updated_at_utc")
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(ts)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if (now - t).total_seconds() >= stale_hours * 3600:
            item.update({"status": "pending", "worker": None, "note": f"超时重置(>{stale_hours}h)"})
            reset += 1
    return reset


def _claim_next_symbol(state: dict[str, Any], worker_id: str) -> Optional[str]:
    for s in state.get("symbols", []):
        item = state["items"].get(s, {})
        if item.get("status") == "pending":
            item.update({"status": "in_progress", "updated_at_utc": _utc_now_iso(), "worker": worker_id, "note": None})
            state["items"][s] = item
            return s
    return None


def _mark_symbol(state: dict[str, Any], symbol: str, status: str, worker_id: str, note: Optional[str]) -> None:
    item = state["items"].get(symbol, {})
    item.update({"status": status, "updated_at_utc": _utc_now_iso(), "worker": worker_id, "note": note})
    state["items"][symbol] = item


def _count_status(state: dict[str, Any]) -> dict[str, int]:
    c: dict[str, int] = {"pending": 0, "in_progress": 0, "done": 0, "failed": 0}
    for item in (state.get("items") or {}).values():
        st = (item.get("status") or "pending").strip()
        c[st] = c.get(st, 0) + 1
    return c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="aggTrades 并发队列 worker（共享 state 文件）")
    ap.add_argument("--worker-id", required=True, help="worker 标识（例如 A/B/C）")
    ap.add_argument("--symbols-file", default=str(Path(__file__).resolve().parent / "top100_perp_symbols.txt"))
    ap.add_argument("--market", default="um", choices=["um", "cm"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--state", default="AlphaQCM_data/_tmp/aggTrades_queue_state.json")
    ap.add_argument(
        "--stale-hours",
        type=float,
        default=48.0,
        help="超过该时长未更新的 in_progress 重置为 pending（默认 48h，避免大币种单次跑太久被误判超时）",
    )
    ap.add_argument("--output-dir", default="AlphaQCM_data/binance_aggTrades")
    ap.add_argument("--temp-dir", default="AlphaQCM_data/_tmp/binance_vision")
    ap.add_argument("--aggtrades-1m-dir", default="AlphaQCM_data/binance_aggTrades_1m_parquet")
    ap.add_argument("--skip-existing", action="store_true", help="已完成输出（meta 标记完成）直接跳过")
    ap.add_argument("--max-symbols", type=int, default=0, help="最多处理多少个 symbol（0=不限制）")
    args = ap.parse_args(argv)

    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    symbols = load_symbols(args.symbols_file, market=args.market)

    state_path = Path(_maybe_rewrite_alphaqcm_data_path(args.state))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = QueueConfig(
        symbols_file=str(Path(args.symbols_file)),
        market=args.market,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
    )

    processed = 0
    while True:
        with open(lock_path, "a+") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            state = _ensure_state(state_path, cfg, symbols, skip_existing=bool(args.skip_existing))
            reset = _reset_stale_in_progress(state, stale_hours=float(args.stale_hours))
            if reset:
                print(f"[{args.worker_id}] reset stale in_progress: {reset}")

            sym = _claim_next_symbol(state, args.worker_id)
            counts = _count_status(state)
            _write_json_atomic(state_path, state)
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)

        if sym is None:
            print(f"[{args.worker_id}] 队列已空：pending=0 done={counts.get('done', 0)} failed={counts.get('failed', 0)}")
            return 0

        print(
            f"[{args.worker_id}] claim {sym} | pending={counts.get('pending', 0)} in_progress={counts.get('in_progress', 0)} done={counts.get('done', 0)} failed={counts.get('failed', 0)}"
        )

        try:
            ok = download_symbol_range(
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
                data_type="aggTrades",
                output_dir=args.output_dir,
                temp_dir=args.temp_dir,
                market=args.market,
                interval=None,
                skip_existing=bool(args.skip_existing),
                aggtrades_1m_dir=(args.aggtrades_1m_dir or None),
                aggtrades_keep_zip=False,
            )
            status = "done" if ok else "failed"
            note = None if ok else "download_symbol_range 返回 False（可能无数据/全 404）"
        except KeyboardInterrupt:
            # 先回滚到 pending，避免队列卡住
            with open(lock_path, "a+") as lockf:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
                state = _ensure_state(state_path, cfg, symbols, skip_existing=bool(args.skip_existing))
                _mark_symbol(state, sym, "pending", worker_id=args.worker_id, note="被中断，回滚到 pending")
                _write_json_atomic(state_path, state)
                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            raise
        except Exception as e:
            status = "failed"
            note = f"异常: {type(e).__name__}: {e}"

        with open(lock_path, "a+") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            state = _ensure_state(state_path, cfg, symbols, skip_existing=bool(args.skip_existing))
            _mark_symbol(state, sym, status, worker_id=args.worker_id, note=note)
            counts2 = _count_status(state)
            _write_json_atomic(state_path, state)
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)

        print(
            f"[{args.worker_id}] done {sym} -> {status} | pending={counts2.get('pending', 0)} in_progress={counts2.get('in_progress', 0)} done={counts2.get('done', 0)} failed={counts2.get('failed', 0)}"
        )

        processed += 1
        if args.max_symbols and processed >= args.max_symbols:
            print(f"[{args.worker_id}] 达到 max-symbols={args.max_symbols}，退出")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
