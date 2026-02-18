import os
import csv
import json
import yaml
import argparse
import sys
import torch
from datetime import datetime
from pathlib import Path
from enum import IntEnum
from typing import List, Optional, Sequence

import numpy as np

# 重要：强制让 `alphagen` / `alphagen_qlib` 优先从 AlphaQCM 目录导入，
# 避免与仓库根目录的 git submodule `alphagen/`（同名目录）发生命名空间包冲突。
_base_dir_for_path = Path(__file__).resolve().parent
sys.path.insert(0, str(_base_dir_for_path))

from fqf_iqn_qrdqn.agent import QRQCMAgent, IQCMAgent, FQCMAgent


def _read_csv_header(fp: Path) -> List[str]:
    with fp.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [h.strip() for h in header if h is not None]


def _detect_feature_columns(data_dir: Path, timeframe: str) -> List[str]:
    """
    从数据目录里的 CSV 表头推断特征列全集。

    兼容两种布局：
    - alphagen_ready: {symbol}_train.csv
    - crypto_data: {symbol}_{timeframe}.csv
    """
    train_files = sorted(data_dir.glob("*_train.csv"))
    files = train_files if train_files else sorted(data_dir.glob(f"*_{timeframe}.csv"))
    if not files:
        raise FileNotFoundError(f"未找到任何 CSV：{data_dir}（期望 *_train.csv 或 *_{timeframe}.csv）")

    exclude = {"datetime", "symbol", "split"}
    exclude_env = os.environ.get("ALPHAGEN_EXCLUDE_COLS", "").strip()
    exclude_cols = {c.strip() for c in exclude_env.split(",") if c.strip()} if exclude_env else set()

    mode = os.environ.get("ALPHAGEN_FEATURE_SCHEMA_MODE", "union").strip().lower()
    ordered: List[str] = []
    seen = set()
    per_file_sets: List[set[str]] = []
    for fp in files:
        h = _read_csv_header(fp)
        cols = [c for c in h if c and (c not in exclude) and (not c.startswith("y_")) and (c not in exclude_cols)]
        per_file_sets.append(set(cols))
        for c in cols:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
    if mode == "intersection" and per_file_sets:
        inter = set.intersection(*per_file_sets)
        out = [c for c in ordered if c in inter]
    else:
        out = ordered

    if not out:
        raise RuntimeError(f"未能从表头推断任何特征列：{data_dir}")
    return out


def _install_dynamic_feature_type(feature_cols: Sequence[str]) -> None:
    """
    动态构造 alphagen_qlib.stock_data.FeatureType，使 AlphaQCM 的 alphagen token 能覆盖“宽表全部因子列”。

    必须在 import `alphagen.data.tokens` / `alphagen.rl.env.wrapper` 之前调用（它们 import 时会读取 len(FeatureType)）。
    """
    import alphagen_qlib.stock_data as sd

    members: dict[str, int] = {}
    for i, col in enumerate(feature_cols):
        name = str(col).upper()
        name = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
        if not name or name[0].isdigit():
            name = f"F_{name}"
        base = name
        k = 2
        while name in members:
            name = f"{base}_{k}"
            k += 1
        members[name] = i

    sd.FeatureType = IntEnum("FeatureType", members)  # type: ignore[attr-defined]
    sd.FEATURE_COLUMNS = list(feature_cols)  # type: ignore[attr-defined]


def _cs_mean_pearson_ic(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    计算“按时点的截面皮尔逊相关(IC)”，并对所有时点取均值。
    x/y: shape=(time, symbols)，允许 NaN/inf，自动忽略无效值。
    """
    if x.shape != y.shape:
        raise ValueError(f"x/y 形状不一致: {tuple(x.shape)} vs {tuple(y.shape)}")
    mask = (~torch.isfinite(x)) | (~torch.isfinite(y))
    n = (~mask).sum(dim=1)
    valid = n >= 2
    if not bool(valid.any()):
        return float("nan")

    n_safe = torch.clamp(n, min=1)
    x0 = x.clone()
    y0 = y.clone()
    x0[mask] = 0.0
    y0[mask] = 0.0
    mean_x = x0.sum(dim=1) / n_safe
    mean_y = y0.sum(dim=1) / n_safe
    xc = (x0 - mean_x[:, None]) * (~mask)
    yc = (y0 - mean_y[:, None]) * (~mask)
    var_x = (xc * xc).sum(dim=1) / n_safe
    var_y = (yc * yc).sum(dim=1) / n_safe
    std_x = torch.sqrt(var_x)
    std_y = torch.sqrt(var_y)
    valid = valid & (std_x > 0) & (std_y > 0)
    if not bool(valid.any()):
        return float("nan")

    corr = (xc * yc).sum(dim=1) / (n_safe * std_x * std_y)
    return float(corr[valid].mean().item())


def _select_top_features_by_ic(
    data: torch.Tensor,
    feature_cols: Sequence[str],
    horizon: int,
    k: int,
    corr_threshold: float,
    ensure_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    用训练集做单变量 IC 打分并选 Top-K，同时用 mutual-IC 做去冗余（剔除 |corr|>=阈值 的特征）。
    """
    if k <= 0:
        return list(feature_cols)
    if data.ndim != 3:
        raise ValueError(f"data 期望 shape=(time, features, symbols)，但得到: {tuple(data.shape)}")
    if len(feature_cols) != int(data.shape[1]):
        raise ValueError(f"feature_cols 数量与 data features 维不一致: {len(feature_cols)} vs {int(data.shape[1])}")

    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"horizon 必须>0，但得到 {horizon}")

    ensure = [c for c in (ensure_cols or []) if c]
    ensure_set = set(ensure)

    if "close" not in feature_cols:
        raise RuntimeError("特征列里缺少 close，无法计算 forward return 作为 target")
    close_idx = list(feature_cols).index("close")
    close = data[:, close_idx, :]
    if close.shape[0] <= horizon:
        raise RuntimeError(f"样本长度不足以计算 {horizon} 期 forward return：time={close.shape[0]}")
    y = close[horizon:, :] / close[:-horizon, :] - 1.0
    x_time = slice(0, -horizon)

    scores: List[tuple[str, float]] = []
    for j, col in enumerate(feature_cols):
        x = data[x_time, j, :]
        ic = _cs_mean_pearson_ic(x, y)
        if not np.isfinite(ic):
            ic = 0.0
        scores.append((str(col), float(ic)))
    scores.sort(key=lambda t: abs(t[1]), reverse=True)

    corr_threshold = max(0.0, min(1.0, float(corr_threshold)))
    kept: List[str] = []
    for c in ensure:
        if c in feature_cols and c not in kept:
            kept.append(c)

    for col, _ic in scores:
        if col in ensure_set or col in kept:
            continue
        if len(kept) >= k:
            break
        j = list(feature_cols).index(col)
        x = data[x_time, j, :]
        redundant = False
        for exist in kept:
            jj = list(feature_cols).index(exist)
            xx = data[x_time, jj, :]
            mic = _cs_mean_pearson_ic(x, xx)
            if np.isfinite(mic) and abs(float(mic)) >= corr_threshold:
                redundant = True
                break
        if not redundant:
            kept.append(col)

    if "close" in feature_cols and "close" not in kept:
        kept = ["close"] + kept
    out = [c for c in kept if c in feature_cols]
    return out if out else ["close"]


def run(args):
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "qcm_config" / f"{args.model}.yaml"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    symbols = args.symbols

    # 数据源：对齐 AlphaGen 的“宽表因子”（alphagen_ready），同时兼容旧的 crypto_data（OHLCV+VWAP）
    # data_dir 既支持：
    # - 相对 AlphaQCM/ 的路径（推荐，例如 AlphaQCM_data/alphagen_ready）
    # - 相对仓库根目录的路径（例如 AlphaQCM/AlphaQCM_data/alphagen_ready）
    # - 绝对路径
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        cwd_candidate = (Path.cwd() / data_dir).resolve()
        if cwd_candidate.exists():
            data_dir = cwd_candidate
        else:
            data_dir = (base_dir / data_dir).resolve()
    else:
        data_dir = data_dir.resolve()

    feature_cols_all = _detect_feature_columns(data_dir, args.timeframe)

    # 可选：特征预筛选（与 AlphaGen 同一套环境变量）
    features_max_raw = os.environ.get("QCM_FEATURES_MAX", "").strip()
    if features_max_raw:
        features_max = int(features_max_raw)
    else:
        features_max = int(os.environ.get("ALPHAGEN_FEATURES_MAX", "0").strip() or 0)
    prune_corr_raw = os.environ.get("QCM_FEATURES_PRUNE_CORR", "").strip()
    if prune_corr_raw:
        prune_corr = float(prune_corr_raw)
    else:
        prune_corr = float(os.environ.get("ALPHAGEN_FEATURES_PRUNE_CORR", "0.95").strip() or 0.95)

    # 先装全量 FeatureType（用于加载一次数据算 IC）
    _install_dynamic_feature_type(feature_cols_all)

    # action space 上限：AlphaEnvWrapper observation dtype=uint8 => SIZE_ALL<=256
    # SIZE_ALL = 1(null)+len(OPS)+len(FT)+len(DELTA)+len(CONSTS)+1(SEP)
    # 对 AlphaQCM 的 alphagen/config.py 来说，固定开销=43，因此 features<=213 才安全。
    if len(feature_cols_all) > 213:
        raise RuntimeError(
            f"特征列过多（{len(feature_cols_all)}），会导致 action/observation space 超过 uint8 上限。"
            f"请先设置 QCM_FEATURES_MAX/ALPHAGEN_FEATURES_MAX 做预筛选。"
        )

    from alphagen_qlib.crypto_data import CryptoData

    if features_max > 0:
        print(f"计算特征 IC 以做预筛选: topK={features_max}, prune_corr={prune_corr}")
        score_data = CryptoData(
            symbols=symbols,
            start_time=args.train_start,
            end_time=args.train_end,
            timeframe=args.timeframe,
            data_dir=str(data_dir),
            max_backtrack_periods=100,
            max_future_periods=max(30, int(args.target_periods) + 1),
            features=None,
            device=torch.device("cpu"),
            feature_columns=feature_cols_all,
        )
        selected_cols = _select_top_features_by_ic(
            data=score_data.data.detach().cpu(),
            feature_cols=feature_cols_all,
            horizon=int(args.target_periods),
            k=features_max,
            corr_threshold=prune_corr,
            ensure_cols=["close"],
        )
        feature_cols = selected_cols
        print(f"预筛选后特征数: {len(feature_cols)}")
        print(f"预筛选特征列表: {feature_cols}")
        try:
            out_dir = base_dir / "AlphaQCM_data" / "_tmp"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "qcm_selected_features.json").write_text(
                json.dumps({"features": feature_cols}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"✓ 预筛选结果已保存: {out_dir / 'qcm_selected_features.json'}")
        except Exception as e:
            print(f"⚠ 保存预筛选结果失败: {e}")
    else:
        feature_cols = feature_cols_all

    # 用最终特征集合重建 FeatureType（必须在 import alphagen token/env 之前）
    _install_dynamic_feature_type(feature_cols)

    if len(feature_cols) > 213:
        raise RuntimeError(f"预筛选后特征仍过多（{len(feature_cols)}），请进一步降低 FEATURES_MAX。")

    # 现在再 import alphagen（确保 action space 读到的是动态 FeatureType）
    from alphagen.data.expression import Feature, Ref
    import alphagen_qlib.stock_data as sd
    from alphagen_qlib.calculator import QLibStockDataCalculator
    from alphagen.models.alpha_pool import AlphaPool
    from alphagen.rl.env.wrapper import AlphaEnv

    # Target: predict future return (align with AlphaGen when target_periods=1)
    if "close" not in feature_cols:
        raise RuntimeError("特征列中未找到 close，无法构造 target")
    close_idx = feature_cols.index("close")
    close = Feature(sd.FeatureType(close_idx))
    target = Ref(close, -int(args.target_periods)) / close - 1

    # Load crypto data（对齐 AlphaGen：同样的时间切分、同样的 union+coverage+ffill+NaN 语义）
    data_train = CryptoData(
        symbols=symbols,
        start_time=args.train_start,
        end_time=args.train_end,
        timeframe=args.timeframe,
        data_dir=str(data_dir),
        max_backtrack_periods=100,
        max_future_periods=max(30, int(args.target_periods) + 1),
        features=None,
        feature_columns=feature_cols,
        device=device
    )
    data_valid = CryptoData(
        symbols=symbols,
        start_time=args.valid_start,
        end_time=args.valid_end,
        timeframe=args.timeframe,
        data_dir=str(data_dir),
        max_backtrack_periods=100,
        max_future_periods=max(30, int(args.target_periods) + 1),
        features=None,
        feature_columns=feature_cols,
        device=device
    )
    data_test = CryptoData(
        symbols=symbols,
        start_time=args.test_start,
        end_time=args.test_end,
        timeframe=args.timeframe,
        data_dir=str(data_dir),
        max_backtrack_periods=100,
        max_future_periods=max(30, int(args.target_periods) + 1),
        features=None,
        feature_columns=feature_cols,
        device=device
    )

    train_calculator = QLibStockDataCalculator(data_train, target)
    valid_calculator = QLibStockDataCalculator(data_valid, target)
    test_calculator = QLibStockDataCalculator(data_test, target)

    train_pool = AlphaPool(
        capacity=args.pool,
        calculator=train_calculator,
        ic_lower_bound=None,
        l1_alpha=5e-3
    )
    train_env = AlphaEnv(pool=train_pool, device=device, print_expr=True)

    # Specify the directory to log
    name = args.model
    time = datetime.now().strftime("%Y%m%d-%H%M")

    if name in ['qrdqn', 'iqn']:
        log_dir = os.path.join(
            str((base_dir / 'AlphaQCM_data' / 'crypto_logs').resolve()),
            f'{symbols}_{args.timeframe}',
            f'pool_{args.pool}_QCM_{args.std_lam}',
            f"{name}-seed{args.seed}-{time}-N{config['N']}-lr{config['lr']}-per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}"
        )
    elif name == 'fqf':
        log_dir = os.path.join(
            str((base_dir / 'AlphaQCM_data' / 'crypto_logs').resolve()),
            f'{symbols}_{args.timeframe}',
            f'pool_{args.pool}_QCM_{args.std_lam}',
            f"{name}-seed{args.seed}-{time}-N{config['N']}-lr{config['quantile_lr']}-per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}"
        )

    # Create the agent and run
    if name == 'qrdqn':
        agent = QRQCMAgent(
            env=train_env,
            valid_calculator=valid_calculator,
            test_calculator=test_calculator,
            log_dir=log_dir,
            seed=args.seed,
            std_lam=args.std_lam,
            cuda=True,
            **config
        )
    elif name == 'iqn':
        agent = IQCMAgent(
            env=train_env,
            valid_calculator=valid_calculator,
            test_calculator=test_calculator,
            log_dir=log_dir,
            seed=args.seed,
            std_lam=args.std_lam,
            cuda=True,
            **config
        )
    elif name == 'fqf':
        agent = FQCMAgent(
            env=train_env,
            valid_calculator=valid_calculator,
            test_calculator=test_calculator,
            log_dir=log_dir,
            seed=args.seed,
            std_lam=args.std_lam,
            cuda=True,
            **config
        )

    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qrdqn',
                        choices=['qrdqn', 'iqn', 'fqf'],
                        help='Model type')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--pool', type=int, default=20,
                        help='Alpha pool capacity')
    parser.add_argument('--std-lam', type=float, default=1.0,
                        help='Standard deviation lambda')
    parser.add_argument('--symbols', type=str, default='top10',
                        choices=['top10', 'top20', 'top100', 'all'],
                        help='Symbol group to trade')
    parser.add_argument('--timeframe', type=str, default='1h',
                        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                        help='Candle timeframe')
    parser.add_argument('--target-periods', type=int, default=1,
                        help='预测多少期后的收益（1h 下填 1 表示 1 小时 forward return）')
    parser.add_argument('--data-dir', type=str, default='AlphaQCM_data/alphagen_ready',
                        help="数据目录（默认 alphagen_ready 宽表；也可填 AlphaQCM_data/crypto_data）")
    parser.add_argument('--train-start', type=str, default='2020-01-01',
                        help='Train start (inclusive)')
    parser.add_argument('--train-end', type=str, default='2024-01-01 00:00:00+00:00',
                        help='Train end (inclusive，建议 UTC)')
    parser.add_argument('--valid-start', type=str, default='2024-01-01 00:00:00+00:00',
                        help='Validation start (inclusive，建议 UTC)')
    parser.add_argument('--valid-end', type=str, default='2024-07-01 00:00:00+00:00',
                        help='Validation end (inclusive，建议 UTC)')
    parser.add_argument('--test-start', type=str, default='2024-07-01 00:00:00+00:00',
                        help='Test start (inclusive，建议 UTC)')
    parser.add_argument('--test-end', type=str, default='2025-02-15',
                        help='Test end (inclusive)')
    args = parser.parse_args()
    run(args)
