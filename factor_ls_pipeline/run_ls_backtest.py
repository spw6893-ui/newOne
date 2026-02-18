#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


RE_FEATURE = re.compile(r"\$([A-Za-z0-9_]+)")


def _parse_timeframe(timeframe: str) -> pd.Timedelta:
    tf = str(timeframe).strip().lower()
    if tf == "1h":
        return pd.Timedelta(hours=1)
    if tf == "4h":
        return pd.Timedelta(hours=4)
    if tf == "1d":
        return pd.Timedelta(days=1)
    raise ValueError(f"暂不支持的 timeframe: {timeframe}（目前支持 1h/4h/1d）")


def _read_csv_header(fp: Path) -> List[str]:
    with fp.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [h.strip() for h in header if h is not None]


def _list_symbol_files(data_dir: Path) -> List[Path]:
    files = sorted(data_dir.glob("*_train.csv"))
    if files:
        return files
    # 兼容 crypto_data：{symbol}_{timeframe}.csv（这里不强制 timeframe，由调用者提前过滤）
    return sorted(data_dir.glob("*.csv"))


def _symbol_from_filename(fp: Path) -> str:
    name = fp.name
    if name.endswith("_train.csv"):
        return name[: -len("_train.csv")]
    if name.endswith(".csv"):
        return name[: -len(".csv")]
    return name


def _load_factor_file(fp: Path) -> Tuple[List[str], List[float]]:
    """
    返回 (expr_strings, weights)。
    支持：
    - alpha_pool.json: {"exprs": [...], "weights": [...]}
    - validation_results.json: {"factors": [...], "weights": [...]}
    - *_best_table.csv: 列 exprs/weight
    """
    if fp.suffix.lower() == ".json":
        obj = json.loads(fp.read_text(encoding="utf-8"))
        if "factors" in obj and "weights" in obj:
            return [str(x) for x in obj["factors"]], [float(x) for x in obj["weights"]]
        if "exprs" in obj and "weights" in obj:
            return [str(x) for x in obj["exprs"]], [float(x) for x in obj["weights"]]
        raise ValueError(f"不支持的 JSON 格式：{fp}（需要 factors/weights 或 exprs/weights）")

    if fp.suffix.lower() == ".csv":
        df = pd.read_csv(fp)
        if "exprs" not in df.columns or "weight" not in df.columns:
            raise ValueError(f"CSV 需要包含列 exprs/weight：{fp}")
        df = df[df["exprs"].astype(str).str.lower() != "ensemble"].copy()
        exprs = [str(x) for x in df["exprs"].tolist()]
        weights = [float(x) for x in df["weight"].tolist()]
        return exprs, weights

    raise ValueError(f"不支持的因子文件类型：{fp}")


def _extract_feature_names(exprs: Iterable[str]) -> List[str]:
    feats: List[str] = []
    seen = set()
    for s in exprs:
        for m in RE_FEATURE.finditer(str(s)):
            f = m.group(1)
            if f and f not in seen:
                seen.add(f)
                feats.append(f)
    # close 必须存在（用于 forward return）
    if "close" not in seen:
        feats = ["close"] + feats
    return feats


def _install_dynamic_feature_type(feature_cols: Sequence[str]) -> Tuple[object, Dict[str, int]]:
    """
    给 alphagen_qlib.stock_data 安装动态 FeatureType，并返回 (sd_module, col_to_idx)。
    """
    import alphagen_qlib.stock_data as sd

    col_to_idx = {str(c): i for i, c in enumerate(feature_cols)}
    members: Dict[str, int] = {}
    for c, i in col_to_idx.items():
        name = str(c).upper()
        name = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
        if not name or name[0].isdigit():
            name = f"F_{name}"
        base = name
        k = 2
        while name in members:
            name = f"{base}_{k}"
            k += 1
        members[name] = int(i)

    sd.FeatureType = IntEnum("FeatureType", members)  # type: ignore[attr-defined]
    sd.FEATURE_COLUMNS = list(feature_cols)  # type: ignore[attr-defined]
    return sd, col_to_idx


@dataclass(frozen=True)
class ParsedExpr:
    expr: "Expression"
    need_backtrack: int
    need_future: int


class _ExprParser:
    def __init__(self, s: str, col_to_idx: Dict[str, int], sd_module: object):
        self.s = s.strip()
        self.i = 0
        self.col_to_idx = col_to_idx
        self.sd = sd_module
        self.need_backtrack = 0
        self.need_future = 0

        from alphagen.data.expression import (  # noqa: WPS433
            Abs,
            Add,
            Constant,
            Corr,
            Cov,
            Delta,
            Div,
            EMA,
            Feature,
            Greater,
            Less,
            Log,
            Mad,
            Max,
            Mean,
            Med,
            Min,
            Mul,
            Pow,
            Ref,
            Std,
            Sub,
            Sum,
            Var,
            WMA,
        )

        self.Feature = Feature
        self.Constant = Constant

        self.op_map = {
            "Abs": Abs,
            "Log": Log,
            "Add": Add,
            "Sub": Sub,
            "Mul": Mul,
            "Div": Div,
            "Pow": Pow,
            "Greater": Greater,
            "Less": Less,
            "Ref": Ref,
            "Mean": Mean,
            "Sum": Sum,
            "Std": Std,
            "Var": Var,
            "Max": Max,
            "Min": Min,
            "Med": Med,
            "Mad": Mad,
            "Delta": Delta,
            "WMA": WMA,
            "EMA": EMA,
            "Cov": Cov,
            "Corr": Corr,
            "Constant": Constant,
        }

        self.rolling_ops = {
            "Ref",
            "Mean",
            "Sum",
            "Std",
            "Var",
            "Max",
            "Min",
            "Med",
            "Mad",
            "Delta",
            "WMA",
            "EMA",
            "Cov",
            "Corr",
        }

    def _peek(self) -> str:
        return self.s[self.i : self.i + 1]

    def _eat_ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def _eat(self, ch: str) -> None:
        self._eat_ws()
        if self._peek() != ch:
            raise ValueError(f"解析失败：期望 '{ch}'，但得到 '{self._peek()}'，原串={self.s!r}")
        self.i += 1

    def _read_ident(self) -> str:
        self._eat_ws()
        j = self.i
        while self.i < len(self.s) and (self.s[self.i].isalnum() or self.s[self.i] == "_"):
            self.i += 1
        if self.i == j:
            raise ValueError(f"解析失败：期望标识符，原串={self.s!r}")
        return self.s[j : self.i]

    def _read_number_token(self) -> str:
        self._eat_ws()
        j = self.i
        if self._peek() in {"+", "-"}:
            self.i += 1
        while self.i < len(self.s) and (self.s[self.i].isdigit() or self.s[self.i] in {".", "e", "E", "+" , "-"}):
            # 简单放宽，交给 float/int 处理
            self.i += 1
        if self.i == j:
            raise ValueError(f"解析失败：期望数字，原串={self.s!r}")
        # 支持 10d 这种
        if self.i < len(self.s) and self.s[self.i] == "d":
            self.i += 1
        return self.s[j : self.i]

    def _parse_atom(self):
        self._eat_ws()
        ch = self._peek()

        # feature: $name
        if ch == "$":
            self.i += 1
            name = self._read_ident()
            if name not in self.col_to_idx:
                raise KeyError(f"特征列不存在：{name}（表达式={self.s!r}）")
            idx = self.col_to_idx[name]
            return self.Feature(self.sd.FeatureType(idx))

        # number literal (AlphaGen 风格常量)
        if ch.isdigit() or ch in {"+", "-"}:
            tok = self._read_number_token()
            if tok.endswith("d"):
                return int(tok[:-1])
            if re.fullmatch(r"[+-]?\d+", tok):
                return int(tok)
            return float(tok)

        # identifier: OpName(...)
        ident = self._read_ident()
        if ident not in self.op_map:
            raise KeyError(f"未知算子/构造器：{ident}（原串={self.s!r}）")

        # function call
        self._eat("(")
        args = []
        if self._peek() != ")":
            while True:
                args.append(self._parse_expr())
                self._eat_ws()
                if self._peek() == ",":
                    self.i += 1
                    continue
                break
        self._eat(")")

        # Constant(x) 兼容
        if ident == "Constant":
            if len(args) != 1:
                raise ValueError(f"Constant 参数错误：{args}")
            v = args[0]
            if isinstance(v, (int, float)):
                return self.Constant(float(v))
            raise ValueError(f"Constant 只支持数字参数：{v}")

        # rolling/pair rolling 需要处理 dt 的 backtrack/future 需求
        if ident in self.rolling_ops:
            # Ref(op, dt)
            if ident == "Ref":
                if len(args) != 2:
                    raise ValueError(f"Ref 参数错误：{args}")
                dt = args[1]
                if not isinstance(dt, int):
                    if isinstance(dt, float) and float(dt).is_integer():
                        dt = int(dt)
                    else:
                        raise ValueError(f"Ref 的 dt 必须是 int：{dt}")
                # dt>0 => lookback，需要 max_backtrack>=dt；dt<0 => forward，需要 max_future>=-dt
                if dt > 0:
                    self.need_backtrack = max(self.need_backtrack, int(dt))
                elif dt < 0:
                    self.need_future = max(self.need_future, int(-dt))
                return self.op_map[ident](args[0], dt)

            # PairRolling: Cov/Corr(lhs, rhs, dt)
            if ident in {"Cov", "Corr"}:
                if len(args) != 3:
                    raise ValueError(f"{ident} 参数错误：{args}")
                dt = args[2]
                if not isinstance(dt, int):
                    if isinstance(dt, float) and float(dt).is_integer():
                        dt = int(dt)
                    else:
                        raise ValueError(f"{ident} 的 dt 必须是 int：{dt}")
                self.need_backtrack = max(self.need_backtrack, int(dt) - 1)
                return self.op_map[ident](args[0], args[1], dt)

            # Rolling: Op(expr, dt)
            if len(args) != 2:
                raise ValueError(f"{ident} 参数错误：{args}")
            dt = args[1]
            if not isinstance(dt, int):
                if isinstance(dt, float) and float(dt).is_integer():
                    dt = int(dt)
                else:
                    raise ValueError(f"{ident} 的 dt 必须是 int：{dt}")
            self.need_backtrack = max(self.need_backtrack, int(dt) - 1)
            return self.op_map[ident](args[0], dt)

        # unary/binary
        op = self.op_map[ident]
        if len(args) == 1:
            return op(args[0])
        if len(args) == 2:
            return op(args[0], args[1])
        raise ValueError(f"{ident} 参数个数不支持：{len(args)}")

    def _parse_expr(self):
        return self._parse_atom()


def parse_expression_string(s: str, col_to_idx: Dict[str, int], sd_module: object) -> ParsedExpr:
    """
    把字符串表达式解析成 alphagen.data.expression.Expression，并返回 backtrack/future 需求。
    """
    # 为避免命名空间冲突，延迟导入类型
    from alphagen.data.expression import Expression  # noqa: WPS433

    p = _ExprParser(s, col_to_idx=col_to_idx, sd_module=sd_module)
    expr = p._parse_expr()
    if not isinstance(expr, Expression):
        # 允许纯 feature/常量等被解析成 Expression；数字不应成为“最终表达式”
        raise ValueError(f"表达式不是 Expression：{s!r} -> {expr!r}")
    return ParsedExpr(expr=expr, need_backtrack=p.need_backtrack, need_future=p.need_future)


@dataclass
class StockDataLike:
    data: torch.Tensor  # (T, F, S)
    dates: pd.DatetimeIndex
    symbols: List[str]
    max_backtrack_days: int
    max_future_days: int

    @property
    def n_stocks(self) -> int:
        return int(self.data.shape[-1])

    @property
    def n_features(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_days(self) -> int:
        return int(self.data.shape[0] - self.max_backtrack_days - self.max_future_days)


def _load_wide_csv_data(
    data_dir: Path,
    symbols: Sequence[str],
    feature_cols: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeframe: str,
    max_backtrack: int,
    max_future: int,
    min_date_coverage_frac: float,
    factor_min_coverage: float,
) -> StockDataLike:
    """
    读取 alphagen_ready 宽表（每币一 CSV），对齐到同一时间轴，并构造 (T,F,S) tensor。
    """
    tf = _parse_timeframe(timeframe)
    start2 = start - max_backtrack * tf
    end2 = end + max_future * tf

    feat_set = list(feature_cols)
    # 保证 close 在第一位不要求，但必须存在
    if "close" not in feat_set:
        feat_set = ["close"] + feat_set

    all_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        fp = data_dir / f"{sym}_train.csv"
        if not fp.exists():
            # 兼容 crypto_data 命名
            fp2 = data_dir / f"{sym}_{timeframe}.csv"
            fp = fp2 if fp2.exists() else fp
        if not fp.exists():
            continue

        usecols = ["datetime"] + [c for c in feat_set if c]
        try:
            df = pd.read_csv(fp, usecols=usecols, parse_dates=["datetime"])
            df = df.set_index("datetime")
        except Exception:
            # 回退全量读取（防止列缺失导致崩）
            df = pd.read_csv(fp, index_col=0, parse_dates=True)

        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        if idx.isna().all() and "datetime" in df.columns:
            idx = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.drop(columns=["datetime"])
        df.index = idx
        df = df.loc[df.index.notna()].copy()

        df = df[(df.index >= start2) & (df.index <= end2)]
        if len(df) == 0:
            continue
        all_dfs[sym] = df

    if not all_dfs:
        raise RuntimeError(f"没有加载到任何币种数据：{data_dir}")

    # union dates + coverage filter（对齐 AlphaQCM CryptoData 的策略）
    all_dates = pd.DatetimeIndex([])
    for df in all_dfs.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()

    date_counts = pd.Series(0, index=all_dates)
    for df in all_dfs.values():
        date_counts[df.index] += 1

    min_cov_n = max(1, int(np.ceil(len(all_dfs) * float(min_date_coverage_frac))))
    valid_dates = date_counts[date_counts >= min_cov_n].index

    syms = list(all_dfs.keys())
    n_dates = len(valid_dates)
    n_features = len(feat_set)
    n_symbols = len(syms)

    data = np.full((n_dates, n_features, n_symbols), np.nan, dtype=np.float32)
    for i, sym in enumerate(syms):
        df = all_dfs[sym].reindex(valid_dates).ffill()
        for j, col in enumerate(feat_set):
            if col in df.columns:
                data[:, j, i] = pd.to_numeric(df[col], errors="coerce").astype("float32").values

    # 可选：因子可用性门控（按时点跨币种覆盖率阈值）
    min_cov = float(factor_min_coverage)
    if min_cov > 0:
        min_cov = max(0.0, min(1.0, min_cov))
        for j in range(n_features):
            cov = np.mean(~np.isnan(data[:, j, :]), axis=1)
            low = cov < min_cov
            if np.any(low):
                data[low, j, :] = np.nan

    inf_mask = ~np.isfinite(data)
    if np.any(inf_mask):
        data[inf_mask] = np.nan

    tensor = torch.from_numpy(data)

    return StockDataLike(
        data=tensor,
        dates=valid_dates,
        symbols=syms,
        max_backtrack_days=int(max_backtrack),
        max_future_days=int(max_future),
    )


def _zscore_cs_keep_nan(x: torch.Tensor) -> torch.Tensor:
    """
    截面 z-score（按 time 维度逐行），保留 NaN 语义。
    x: (T, S)
    """
    mask = torch.isnan(x) | (~torch.isfinite(x))
    n = (~mask).sum(dim=1)
    n_safe = torch.clamp(n, min=1)
    x0 = x.clone()
    x0[mask] = 0.0
    mean = x0.sum(dim=1) / n_safe
    xc = (x0 - mean[:, None]) * (~mask)
    var = (xc * xc).sum(dim=1) / n_safe
    std = torch.sqrt(var)
    std_safe = torch.where(std > 0, std, torch.ones_like(std))
    out = (x0 - mean[:, None]) / std_safe[:, None]
    out[mask] = torch.nan
    return out


def _combine_factors(expr_values: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    """
    NaN 友好组合：单个因子缺失视为 0，只有全缺失才是 NaN。
    """
    stacked = torch.stack([expr_values[i] * float(weights[i]) for i in range(len(weights))], dim=0)
    all_nan = torch.isnan(stacked).all(dim=0)
    out = torch.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).sum(dim=0)
    out[all_nan] = torch.nan
    return out


def _build_ls_weights(scores: np.ndarray, long_frac: float, short_frac: float, gross: float) -> np.ndarray:
    """
    给定单期截面分数，构建多空权重（sum w = 0，sum |w| = gross）。
    """
    mask = np.isfinite(scores)
    idx = np.where(mask)[0]
    if idx.size < 2:
        return np.zeros_like(scores, dtype=np.float32)

    n = idx.size
    n_long = max(1, int(np.floor(n * long_frac)))
    n_short = max(1, int(np.floor(n * short_frac)))
    if n_long + n_short > n:
        # 太挤了就按可用数量缩
        n_long = max(1, n // 2)
        n_short = max(1, n - n_long)

    order = idx[np.argsort(scores[idx])]
    short_idx = order[:n_short]
    long_idx = order[-n_long:]

    w = np.zeros_like(scores, dtype=np.float32)
    half = float(gross) / 2.0
    w[long_idx] = half / float(n_long)
    w[short_idx] = -half / float(n_short)
    return w


def backtest_long_short(
    data: StockDataLike,
    close_col_idx: int,
    score: torch.Tensor,
    horizon: int,
    long_frac: float,
    short_frac: float,
    gross: float,
    cost_bps: float,
) -> pd.DataFrame:
    """
    score: (n_days, n_symbols)，用于在 t 时刻排序，收益使用 t->t+h 的 forward return。
    """
    T = data.n_days
    S = data.n_stocks
    if score.shape != (T, S):
        raise ValueError(f"score 形状不匹配：{tuple(score.shape)} vs {(T, S)}")

    bt = data.max_backtrack_days
    h = int(horizon)
    close0 = data.data[bt : bt + T, close_col_idx, :].cpu().numpy()
    close1 = data.data[bt + h : bt + h + T, close_col_idx, :].cpu().numpy()
    fwd_ret = close1 / close0 - 1.0

    dates = data.dates[bt : bt + T]
    score_np = score.detach().cpu().numpy().astype(np.float32)

    w_prev = np.zeros(S, dtype=np.float32)
    out = {
        "datetime": [],
        "gross_ret": [],
        "net_ret": [],
        "turnover": [],
        "n_long": [],
        "n_short": [],
    }

    cost_rate = float(cost_bps) * 1e-4

    for t in range(T):
        sc = score_np[t, :]
        # 只用 forward return 有效的币
        valid = np.isfinite(sc) & np.isfinite(fwd_ret[t, :])
        sc2 = np.where(valid, sc, np.nan)
        w = _build_ls_weights(sc2, long_frac=long_frac, short_frac=short_frac, gross=gross)
        turnover = 0.5 * float(np.nansum(np.abs(w - w_prev)))
        gross_ret = float(np.nansum(w * fwd_ret[t, :]))
        net_ret = gross_ret - cost_rate * turnover

        out["datetime"].append(dates[t])
        out["gross_ret"].append(gross_ret)
        out["net_ret"].append(net_ret)
        out["turnover"].append(turnover)
        out["n_long"].append(int(np.sum(w > 0)))
        out["n_short"].append(int(np.sum(w < 0)))
        w_prev = w

    df = pd.DataFrame(out)
    df["equity"] = (1.0 + df["net_ret"]).cumprod()
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["peak"] - 1.0
    return df


def summarize_curve(curve: pd.DataFrame, periods_per_year: float) -> Dict[str, float]:
    r = curve["net_ret"].to_numpy(dtype=float)
    eq = curve["equity"].to_numpy(dtype=float)
    dd = curve["drawdown"].to_numpy(dtype=float)
    if len(r) == 0:
        return {}
    mean = float(np.nanmean(r))
    vol = float(np.nanstd(r, ddof=0))
    ann_ret = (1.0 + mean) ** periods_per_year - 1.0
    ann_vol = vol * np.sqrt(periods_per_year)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else float("nan")
    mdd = float(np.nanmin(dd))
    avg_turnover = float(np.nanmean(curve["turnover"].to_numpy(dtype=float)))
    return {
        "n_periods": float(len(r)),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "avg_turnover": float(avg_turnover),
        "final_equity": float(eq[-1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor-file", required=True, help="alpha_pool.json / validation_results.json / *_best_table.csv")
    ap.add_argument("--data-dir", default="AlphaQCM/AlphaQCM_data/alphagen_ready", help="alphagen_ready 宽表目录")
    ap.add_argument("--symbols", default="all", help="all / top10 / top20 / top100（默认 all）")
    ap.add_argument("--timeframe", default="1h", help="1h/4h/1d（默认 1h）")
    ap.add_argument("--start", default="2024-07-01", help="回测开始日期（UTC 推荐）")
    ap.add_argument("--end", default="2025-02-15", help="回测结束日期（UTC 推荐）")
    ap.add_argument("--horizon", type=int, default=1, help="forward return 期数（1h 下 1=1小时）")
    ap.add_argument("--long-frac", type=float, default=0.2, help="做多分位比例（默认 0.2）")
    ap.add_argument("--short-frac", type=float, default=0.2, help="做空分位比例（默认 0.2）")
    ap.add_argument("--gross", type=float, default=1.0, help="总杠杆（sum|w|，默认 1.0）")
    ap.add_argument("--cost-bps", type=float, default=0.0, help="单边交易成本（bps），按换手扣减（默认 0）")
    ap.add_argument("--min-date-coverage", type=float, default=0.5, help="日期覆盖率阈值（默认 0.5）")
    ap.add_argument("--factor-min-coverage", type=float, default=0.0, help="因子可用性门控阈值（默认 0，关闭）")
    ap.add_argument("--output-dir", default="factor_ls_output", help="输出目录")
    args = ap.parse_args()

    # 让 alphagen submodule 可导入（独立工程但复用 alphagen 表达式执行）
    repo_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(repo_root / "alphagen"))

    factor_file = Path(args.factor_file)
    expr_strs, weights = _load_factor_file(factor_file)
    if len(expr_strs) == 0:
        raise RuntimeError("因子列表为空")
    if len(expr_strs) != len(weights):
        raise RuntimeError(f"exprs/weights 长度不一致：{len(expr_strs)} vs {len(weights)}")

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (repo_root / data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    # 解析需要的特征列
    needed = _extract_feature_names(expr_strs)
    # 宽表里列名是 snake_case，小写；确保一致
    needed = [str(x).strip() for x in needed if str(x).strip()]

    # 安装动态 FeatureType（只包含“表达式用到的列”）
    sd, col_to_idx = _install_dynamic_feature_type(needed)

    # 解析表达式，同时统计 backtrack/future 需求
    parsed: List[ParsedExpr] = []
    max_bt = 0
    max_fu = int(args.horizon)
    for s in expr_strs:
        pe = parse_expression_string(str(s), col_to_idx=col_to_idx, sd_module=sd)
        parsed.append(pe)
        max_bt = max(max_bt, pe.need_backtrack)
        max_fu = max(max_fu, pe.need_future)

    # symbols 解析：默认全量扫描
    sym_arg = str(args.symbols).strip().lower()
    if sym_arg == "all":
        sym_files = _list_symbol_files(data_dir)
        symbols = [_symbol_from_filename(fp) for fp in sym_files]
    elif sym_arg == "top10":
        symbols = [
            "BTC_USDT",
            "ETH_USDT",
            "BNB_USDT",
            "SOL_USDT",
            "XRP_USDT",
            "ADA_USDT",
            "AVAX_USDT",
            "DOGE_USDT",
            "DOT_USDT",
            "MATIC_USDT",
        ]
    elif sym_arg == "top20":
        symbols = [
            "BTC_USDT",
            "ETH_USDT",
            "BNB_USDT",
            "SOL_USDT",
            "XRP_USDT",
            "ADA_USDT",
            "AVAX_USDT",
            "DOGE_USDT",
            "DOT_USDT",
            "MATIC_USDT",
            "LINK_USDT",
            "UNI_USDT",
            "ATOM_USDT",
            "LTC_USDT",
            "ETC_USDT",
            "APT_USDT",
            "ARB_USDT",
            "OP_USDT",
            "INJ_USDT",
            "SUI_USDT",
        ]
    elif sym_arg == "top100":
        sym_files = sorted(data_dir.glob("*_train.csv"))
        symbols = [_symbol_from_filename(fp) for fp in sym_files][:100]
    else:
        # 允许逗号分隔
        symbols = [x.strip() for x in sym_arg.split(",") if x.strip()]

    start = pd.to_datetime(args.start, utc=True)
    end = pd.to_datetime(args.end, utc=True)

    data = _load_wide_csv_data(
        data_dir=data_dir,
        symbols=symbols,
        feature_cols=needed,
        start=start,
        end=end,
        timeframe=args.timeframe,
        max_backtrack=max_bt,
        max_future=max_fu,
        min_date_coverage_frac=float(args.min_date_coverage),
        factor_min_coverage=float(args.factor_min_coverage),
    )

    # 评估每个因子并归一化
    values: List[torch.Tensor] = []
    for pe in parsed:
        v = pe.expr.evaluate(data)  # (n_days, n_symbols)
        v = _zscore_cs_keep_nan(v)
        values.append(v)
    score = _combine_factors(values, weights)

    close_idx = col_to_idx["close"]
    curve = backtest_long_short(
        data=data,
        close_col_idx=int(close_idx),
        score=score,
        horizon=int(args.horizon),
        long_frac=float(args.long_frac),
        short_frac=float(args.short_frac),
        gross=float(args.gross),
        cost_bps=float(args.cost_bps),
    )

    # 汇总指标
    tf = _parse_timeframe(args.timeframe)
    periods_per_year = float(pd.Timedelta(days=365) / tf)
    summary = summarize_curve(curve, periods_per_year=periods_per_year)
    summary.update(
        {
            "timeframe": str(args.timeframe),
            "horizon": int(args.horizon),
            "long_frac": float(args.long_frac),
            "short_frac": float(args.short_frac),
            "gross": float(args.gross),
            "cost_bps": float(args.cost_bps),
            "start": str(start),
            "end": str(end),
            "n_symbols_loaded": int(len(data.symbols)),
            "n_features": int(len(needed)),
            "n_factors": int(len(expr_strs)),
            "factor_file": str(factor_file),
            "data_dir": str(data_dir),
        }
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out_dir / "curve.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========================================")
    print("多空回测完成")
    print("========================================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"曲线: {out_dir / 'curve.csv'}")
    print(f"汇总: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

