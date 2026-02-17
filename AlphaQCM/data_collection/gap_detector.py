"""
数据断流/维护窗口检测（防御性编程）。

核心思想：
- 维护期本身往往“没有数据”（断流），无法直接在缺失区间打标；
- 真正高风险发生在“恢复后的瞬间”（撮合刚启动、流动性脆弱、插针），因此需要对恢复后的若干小时强制禁交易。

本模块提供一个轻量的 Gap Detector：
- 若相邻两条记录时间差 > max_gap_seconds（默认 3605 秒），则认为发生了断流/维护；
- 将“恢复的第一根 bar”标记为 under_maintenance=1；
- 并对恢复后 cooldown_hours（默认 2 小时）内的 bar 标记 cooldown_no_trade=1。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def detect_maintenance_flags(
    index: pd.DatetimeIndex,
    *,
    max_gap_seconds: int = 3605,
    cooldown_hours: int = 2,
) -> pd.DataFrame:
    """
    输入：时间索引（建议 UTC，且按小时 bar）
    输出：与 index 对齐的 DataFrame，包含：
    - gap_seconds: 与上一条记录的时间差（秒）
    - under_maintenance: 是否为“断流后恢复的第一根 bar”（0/1）
    - cooldown_no_trade: 是否处于恢复后冷静期（0/1，包含恢复当下这根 bar）
    - trade_allowed: 是否允许交易（0/1）
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index, utc=True, errors="coerce")
        index = pd.DatetimeIndex(index)

    if len(index) == 0:
        return pd.DataFrame(
            {
                "gap_seconds": pd.Series(dtype="float64"),
                "under_maintenance": pd.Series(dtype="int8"),
                "cooldown_no_trade": pd.Series(dtype="int8"),
                "trade_allowed": pd.Series(dtype="int8"),
            },
            index=index,
        )

    # 保留原始顺序，内部用排序后的副本计算，再 reindex 回来
    idx_sorted = pd.DatetimeIndex(index).sort_values()
    gap_seconds = idx_sorted.to_series().diff().dt.total_seconds()

    # 断流：当前 bar 与上一个 bar 的间隔超过阈值
    under = (gap_seconds > float(max_gap_seconds)).fillna(False)

    cooldown = pd.Series(0, index=idx_sorted, dtype="int8")
    if cooldown_hours > 0 and bool(under.any()):
        for t in idx_sorted[under.to_numpy()]:
            end_t = t + pd.Timedelta(hours=int(cooldown_hours))
            cooldown.loc[(idx_sorted >= t) & (idx_sorted <= end_t)] = 1

    out = pd.DataFrame(
        {
            "gap_seconds": gap_seconds.reindex(idx_sorted),
            "under_maintenance": under.astype("int8").reindex(idx_sorted),
            "cooldown_no_trade": cooldown.reindex(idx_sorted),
        },
        index=idx_sorted,
    )
    out["trade_allowed"] = (out["cooldown_no_trade"] == 0).astype("int8")

    # 对齐回原始 index（含可能的未排序情况）
    return out.reindex(index)

