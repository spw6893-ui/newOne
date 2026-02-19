#!/usr/bin/env python3
"""
AlphaGen（子模块 alphagen/）加密货币因子挖掘训练入口。

重要说明（避免踩坑）：
1) 仓库里同时存在 `alphagen/`（子模块）与 `AlphaQCM/alphagen`（历史目录），
   必须保证 Python 导入时优先使用子模块 alphagen 的实现。
2) 如果你希望把宽表里“所有因子列”都纳入 AlphaGen 的特征空间，
   需要在导入 alphagen 前动态构造 FeatureType（枚举）= 所有可用特征列。
"""

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch


repo_root = Path(__file__).resolve().parent
# 导入优先级：
# 1) alphagen 子模块（repo_root/alphagen）用于 AlphaGen 核心实现
# 2) AlphaQCM 作为“包”使用（repo_root 默认已在 sys.path），只通过 `AlphaQCM.alphagen_qlib` 引用适配层
sys.path.insert(0, str(repo_root / "alphagen"))


@dataclass(frozen=True)
class FeatureSpace:
    feature_cols: List[str]


def _read_csv_header(fp: Path) -> List[str]:
    with fp.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [h.strip() for h in header if h is not None]


def _detect_feature_space(data_dir: Path) -> FeatureSpace:
    """
    从 alphagen_ready/*.csv 的表头推断“特征列全集”。

    规则：
    - 排除元数据：datetime/symbol/split
    - 排除 label：y_*
    - 其余列默认都作为特征列（包含 open/high/low/close/volume 以及你工程生成的高阶因子）
    """
    files = sorted(data_dir.glob("*_train.csv"))
    if not files:
        raise FileNotFoundError(f"未找到训练数据：{data_dir}（期望 *_train.csv）")

    exclude = {"datetime", "symbol", "split"}
    exclude_env = os.environ.get("ALPHAGEN_EXCLUDE_COLS", "").strip()
    exclude_cols = {c.strip() for c in exclude_env.split(",") if c.strip()} if exclude_env else set()

    # 优先使用 prepare_alphagen_training_data.py 写出的 schema.json（若存在）
    schema_fp = data_dir / "schema.json"
    if schema_fp.exists():
        try:
            obj = json.loads(schema_fp.read_text(encoding="utf-8"))
            cols = obj.get("columns", [])
            header = [str(c).strip() for c in cols if str(c).strip()]
        except Exception:
            header = []
    else:
        header = []

    # 如果没有 schema.json，则用扫描表头的方式构造特征列集合
    # mode=union（默认）：把所有币种出现过的列都纳入（“全因子”）
    # mode=intersection：只保留所有币种都具备的列（显著减少 NaN，适合先跑通训练）
    mode = os.environ.get("ALPHAGEN_FEATURE_SCHEMA_MODE", "union").strip().lower()
    if not header:
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
        if mode == "intersection":
            inter = set.intersection(*per_file_sets) if per_file_sets else set()
            header = [c for c in ordered if c in inter]
        else:
            header = ordered

    feature_cols = [c for c in header if c and (c not in exclude) and (not c.startswith("y_")) and (c not in exclude_cols)]
    if not feature_cols:
        raise RuntimeError(f"未能从表头推断出任何特征列：{data_dir}")

    return FeatureSpace(feature_cols=feature_cols)


def _cs_mean_pearson_ic(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    计算“按时点的截面皮尔逊相关(IC)”，并对所有时点取均值。

    x/y: shape=(time, symbols)，允许 NaN/inf，自动忽略无效值。
    """
    if x.shape != y.shape:
        raise ValueError(f"x/y 形状不一致: {tuple(x.shape)} vs {tuple(y.shape)}")
    # mask=True 表示无效值（不参与统计）
    mask = (~torch.isfinite(x)) | (~torch.isfinite(y))
    n = (~mask).sum(dim=1)  # (time,)

    # 至少需要 2 个样本才能算相关
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

    # std=0 的时点视为无效（价格常数/全相等等会出现）
    valid = valid & (std_x > 0) & (std_y > 0)
    if not bool(valid.any()):
        return float("nan")

    corr = (xc * yc).sum(dim=1) / (n_safe * std_x * std_y)
    ic = corr[valid].mean().item()
    return float(ic)


def _select_top_features_by_ic(
    data: torch.Tensor,
    feature_cols: Sequence[str],
    k: int,
    corr_threshold: float,
    ensure_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    用训练集做单变量 IC 打分并选 Top-K，同时用 mutual-IC 做去冗余（近似“高度相关特征剔除”）。

    - IC 口径：每个时点做截面相关（跨币种），再对时点取均值。
    - mutual-IC：同样口径对 (feature_i, feature_j) 做相关，用于剔除 |corr|>=阈值 的冗余特征。
    """
    if k <= 0:
        return list(feature_cols)
    if data.ndim != 3:
        raise ValueError(f"data 期望 shape=(time, features, symbols)，但得到: {tuple(data.shape)}")
    if len(feature_cols) != int(data.shape[1]):
        raise ValueError(f"feature_cols 数量与 data features 维不一致: {len(feature_cols)} vs {int(data.shape[1])}")

    ensure = [c for c in (ensure_cols or []) if c]
    ensure_set = set(ensure)

    if "close" not in feature_cols:
        raise RuntimeError("特征列里缺少 close，无法计算 forward return 作为 target")
    close_idx = list(feature_cols).index("close")
    close = data[:, close_idx, :]  # (time, symbols)
    # forward 1h return: close[t+1]/close[t]-1
    y = close[1:, :] / close[:-1, :] - 1.0

    scores: List[tuple[str, float]] = []
    time_aligned = slice(0, -1)
    for j, col in enumerate(feature_cols):
        x = data[time_aligned, j, :]
        ic = _cs_mean_pearson_ic(x, y)
        if not np.isfinite(ic):
            ic = 0.0
        scores.append((str(col), float(ic)))

    # 按 |IC| 排序（绝对值越大越好）
    scores.sort(key=lambda t: abs(t[1]), reverse=True)

    # 贪心去冗余：从高分到低分依次尝试加入
    corr_threshold = float(corr_threshold)
    corr_threshold = max(0.0, min(1.0, corr_threshold))

    kept: List[str] = []

    # 先把 ensure_cols 放进去（顺序保持），避免 target 依赖列丢失
    for c in ensure:
        if c in feature_cols and c not in kept:
            kept.append(c)

    for col, _ic in scores:
        if col in ensure_set:
            continue
        if col in kept:
            continue
        if len(kept) >= k:
            break

        # 与已保留特征计算 mutual-IC，过高则跳过
        j = list(feature_cols).index(col)
        x = data[time_aligned, j, :]
        redundant = False
        for exist in kept:
            jj = list(feature_cols).index(exist)
            xx = data[time_aligned, jj, :]
            mic = _cs_mean_pearson_ic(x, xx)
            if np.isfinite(mic) and abs(float(mic)) >= corr_threshold:
                redundant = True
                break
        if not redundant:
            kept.append(col)

    # 如果 ensure 占满了 k，也允许超过 k（因为 close 必须有）；否则补足到至少包含 ensure
    if "close" in feature_cols and "close" not in kept:
        kept = ["close"] + kept

    # 最终保持原始列名（snake_case），并保证都在 feature_cols 内
    out = [c for c in kept if c in feature_cols]
    # 兜底：至少返回 close
    if not out:
        out = ["close"]
    return out


def _install_dynamic_feature_type(feature_cols: Sequence[str]) -> None:
    """
    动态构造 alphagen_qlib.stock_data.FeatureType，使得 AlphaGen 能把"宽表全部因子列"当作可选 Feature。

    注意：必须在导入 `alphagen.data.tokens` / `alphagen.rl.env.wrapper` 之前调用，
    因为它们会在 import 时读取 `len(FeatureType)` 来构建 action space。
    """
    from enum import IntEnum

    import alphagen_qlib.stock_data as sd

    members = {}
    for i, col in enumerate(feature_cols):
        name = col.upper()
        # 防御：列名如果意外包含非法字符，做一次保守归一化
        #（正常情况下你的列名都是 snake_case，不会触发）
        name = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
        if not name or name[0].isdigit():
            name = f"F_{name}"
        # 去重
        base = name
        k = 2
        while name in members:
            name = f"{base}_{k}"
            k += 1
        members[name] = i

    sd.FeatureType = IntEnum("FeatureType", members)  # type: ignore[attr-defined]
    # 同时暴露"列顺序"，供 CryptoData 做 index->列名映射
    sd.FEATURE_COLUMNS = list(feature_cols)  # type: ignore[attr-defined]


def _dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    # ==================== 配置参数 ====================

    # 数据配置
    # 训练数据默认落在 AlphaQCM/AlphaQCM_data/alphagen_ready（由 run_training.sh / prepare_alphagen_training_data.py 生成）
    DATA_DIR = os.environ.get("ALPHAGEN_DATA_DIR", "AlphaQCM/AlphaQCM_data/alphagen_ready")
    SYMBOLS = os.environ.get("ALPHAGEN_SYMBOLS", "top100")  # 或指定列表: ['BTCUSDT', 'ETHUSDT', ...]

    # 时间分割
    START_TIME = os.environ.get("ALPHAGEN_START_TIME", "2020-01-01")
    TRAIN_END = os.environ.get("ALPHAGEN_TRAIN_END", "2024-01-01")
    VAL_END = os.environ.get("ALPHAGEN_VAL_END", "2024-07-01")
    END_TIME = os.environ.get("ALPHAGEN_END_TIME", "2025-02-15")

    # 特征：默认把 alphagen_ready 里的"全部因子列"都扔进 AlphaGen（FeatureType 动态构造）
    # 可选：用训练集的单变量 IC 做预筛选，从而缩小 action space（更容易探索/更快收敛）
    feature_space = _detect_feature_space(Path(DATA_DIR))

    features_max = int(os.environ.get("ALPHAGEN_FEATURES_MAX", "0").strip() or 0)
    prune_corr = float(os.environ.get("ALPHAGEN_FEATURES_PRUNE_CORR", "0.95").strip() or 0.95)
    if features_max > 0:
        # 先用“全特征”构造 FeatureType，加载一次数据做打分，然后再用筛选后的特征重建 FeatureType。
        _install_dynamic_feature_type(feature_space.feature_cols)
        from AlphaQCM.alphagen_qlib.crypto_data import CryptoData

        print(f"计算特征 IC 以做预筛选: topK={features_max}, prune_corr={prune_corr}")
        score_data = CryptoData(
            symbols=SYMBOLS,
            start_time=START_TIME,
            end_time=TRAIN_END,
            timeframe="1h",
            data_dir=DATA_DIR,
            max_backtrack_periods=100,
            max_future_periods=30,
            features=None,
            device=torch.device("cpu"),
        )
        selected_cols = _select_top_features_by_ic(
            data=score_data.data.detach().cpu(),
            feature_cols=feature_space.feature_cols,
            k=features_max,
            corr_threshold=prune_corr,
            ensure_cols=["close"],
        )
        feature_space = FeatureSpace(feature_cols=selected_cols)
        print(f"预筛选后特征数: {len(feature_space.feature_cols)}")
        print(f"预筛选特征列表: {feature_space.feature_cols}")
        # 记录到输出目录，方便复现实验
        try:
            out_dir = Path("./alphagen_output")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "selected_features.json").write_text(
                json.dumps({"features": feature_space.feature_cols}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"✓ 预筛选结果已保存: {out_dir / 'selected_features.json'}")
        except Exception as e:
            print(f"⚠ 保存预筛选结果失败: {e}")

    _install_dynamic_feature_type(feature_space.feature_cols)
    # alphagen wrapper 的 state dtype 默认是 uint8，因此 action_space 不能超过 255
    # action_space 大小约等于 len(features) + 常量/算子开销（约 42）
    if len(feature_space.feature_cols) + 42 > 255:
        raise RuntimeError(
            f"特征列过多（{len(feature_space.feature_cols)}），会导致 AlphaGen action_space>255（uint8 溢出）。"
            f"请减少特征列或改造 alphagen 的 wrapper dtype。"
        )

    # 现在再 import alphagen（确保 action space 读到的是动态 FeatureType）
    # 注意：当前 alphagen 版本没有 Close()/Open() 这类快捷构造器，使用 Feature(FeatureType.X) 即可。
    from alphagen.data.expression import Feature, Ref
    # 兼容：alphagen 上游 rolling Std/Var 在窗口=1 时会触发 dof<=0 警告并产生 NaN（unbiased=True 的默认行为）。
    # 这里做一次运行时 monkey patch，避免需要修改 submodule 指针（否则会导致他人无法拉取特定 commit）。
    import alphagen.data.expression as _expr_mod
    def _std_apply_unbiased_false(self, operand):  # type: ignore[no-redef]
        return operand.std(dim=-1, unbiased=False)
    def _var_apply_unbiased_false(self, operand):  # type: ignore[no-redef]
        return operand.var(dim=-1, unbiased=False)
    _expr_mod.Std._apply = _std_apply_unbiased_false  # type: ignore[assignment]
    _expr_mod.Var._apply = _var_apply_unbiased_false  # type: ignore[assignment]
    from alphagen.models.linear_alpha_pool import MeanStdAlphaPool, MseAlphaPool
    from alphagen.rl.env.wrapper import AlphaEnv
    import alphagen.rl.env.wrapper as env_wrapper
    from alphagen.rl.policy import LSTMSharedNet
    from alphagen.utils import reseed_everything
    from sb3_contrib.ppo_mask import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.callbacks import CallbackList

    from AlphaQCM.alphagen_qlib.crypto_data import CryptoData
    from AlphaQCM.alphagen_qlib.calculator import QLibStockDataCalculator, TensorQLibStockDataCalculator
    import alphagen_qlib.stock_data as sd

    import alphagen as alphagen_pkg
    alphagen_file = getattr(alphagen_pkg, "__file__", None)
    alphagen_path = list(getattr(alphagen_pkg, "__path__", []))
    # namespace package 场景下 __file__ 可能为 None，用 __path__ 更可靠
    print(f"alphagen 包路径: {alphagen_file if alphagen_file else alphagen_path}")

    class TensorboardCallback(BaseCallback):
        """记录训练指标到TensorBoard"""

        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            return True

    class IcLowerBoundScheduleCallback(BaseCallback):
        """
        动态 IC lower bound：
        - 训练初期放宽（更容易把“尚可”的表达式塞进 pool）
        - 训练后期收紧（逼迫更强的 alpha 进入 pool）
        """

        def __init__(
            self,
            pool,
            total_timesteps: int,
            start_lb: float,
            end_lb: float,
            update_every: int,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self.pool = pool
            self.total_timesteps = max(1, int(total_timesteps))
            self.start_lb = float(start_lb)
            self.end_lb = float(end_lb)
            self.update_every = max(1, int(update_every))
            self._last_lb: Optional[float] = None

        def _compute_lb(self) -> float:
            frac = min(1.0, float(self.num_timesteps) / float(self.total_timesteps))
            return self.start_lb + frac * (self.end_lb - self.start_lb)

        def _on_step(self) -> bool:
            if (self.num_timesteps % self.update_every) != 0:
                return True
            lb = float(self._compute_lb())
            # LinearAlphaPool 内部用的是 `_ic_lower_bound`（float），这里直接更新即可。
            setattr(self.pool, "_ic_lower_bound", lb)
            self.logger.record("pool/ic_lower_bound", lb)
            self._last_lb = lb
            return True

    class NaNFriendlyMeanStdAlphaPool(MeanStdAlphaPool):
        """
        MeanStdAlphaPool 的 NaN 友好版本：
        - 单个因子缺失视为 0
        - 只有“全因子都缺失”的位置才保留 NaN

        目的：避免加权求和时 NaN 传播，导致组合 alpha 大面积 NaN，从而把 IC/ICIR 压成 0。
        """

        def _calc_obj_impl(self, alpha_values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            from alphagen.utils.correlation import batch_pearsonr  # 延迟导入避免循环

            target_value = self.calculator.target
            all_nan = torch.isnan(alpha_values).all(dim=0)
            weighted = (weights[:, None, None] * torch.nan_to_num(alpha_values, nan=0.0)).sum(dim=0)
            weighted[all_nan] = torch.nan
            ics = batch_pearsonr(weighted, target_value)
            mean, std = ics.mean(), ics.std()
            if getattr(self, "_lcb_beta", None) is not None:
                return mean - float(getattr(self, "_lcb_beta")) * std
            return mean / std

    # 训练配置
    SEED = int(os.environ.get("ALPHAGEN_SEED", "42"))
    BATCH_SIZE = int(os.environ.get("ALPHAGEN_BATCH_SIZE", "128"))
    TOTAL_TIMESTEPS = int(os.environ.get("ALPHAGEN_TOTAL_TIMESTEPS", "100000"))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_obj = torch.device(DEVICE)

    # 输出配置
    OUTPUT_DIR = Path('./alphagen_output')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Alpha Pool配置 - 针对高维特征优化
    POOL_CAPACITY = int(os.environ.get("ALPHAGEN_POOL_CAPACITY", "10"))
    IC_LOWER_BOUND = float(os.environ.get("ALPHAGEN_IC_LOWER_BOUND", "0.01"))
    L1_ALPHA = float(os.environ.get("ALPHAGEN_POOL_L1_ALPHA", "0.005"))
    POOL_TYPE = os.environ.get("ALPHAGEN_POOL_TYPE", "mse").strip().lower()  # mse / meanstd
    pool_lcb_beta_raw = os.environ.get("ALPHAGEN_POOL_LCB_BETA", "none").strip().lower()
    POOL_LCB_BETA: Optional[float]
    if pool_lcb_beta_raw in {"none", "null", ""}:
        POOL_LCB_BETA = None
    else:
        POOL_LCB_BETA = float(pool_lcb_beta_raw)

    # 动态 threshold：start/end 任意一个被设置就启用（默认与 IC_LOWER_BOUND 相同 => 等价于关闭）
    ic_lb_start = float(os.environ.get("ALPHAGEN_IC_LOWER_BOUND_START", str(IC_LOWER_BOUND)))
    ic_lb_end = float(os.environ.get("ALPHAGEN_IC_LOWER_BOUND_END", str(IC_LOWER_BOUND)))
    ic_lb_update_every = int(os.environ.get("ALPHAGEN_IC_LOWER_BOUND_UPDATE_EVERY", "2048"))

    # 模型/训练超参（允许通过环境变量配置，便于 alphagen_config.sh 生效）
    LSTM_LAYERS = int(os.environ.get("ALPHAGEN_LSTM_LAYERS", "2"))
    LSTM_DIM = int(os.environ.get("ALPHAGEN_LSTM_DIM", "128"))
    LSTM_DROPOUT = float(os.environ.get("ALPHAGEN_LSTM_DROPOUT", "0.1"))
    LEARNING_RATE = float(os.environ.get("ALPHAGEN_LEARNING_RATE", "3e-4"))
    N_STEPS = int(os.environ.get("ALPHAGEN_N_STEPS", "2048"))
    GAE_LAMBDA = float(os.environ.get("ALPHAGEN_GAE_LAMBDA", "0.95"))
    CLIP_RANGE = float(os.environ.get("ALPHAGEN_CLIP_RANGE", "0.2"))
    # 默认不额外加熵正则（保持与 PPO 默认一致，避免在大 action space 下冷启动卡死）
    ENT_COEF = float(os.environ.get("ALPHAGEN_ENT_COEF", "0.0"))
    # 默认关闭 KL early-stop（更贴近最初的训练脚本行为；需要时可显式设置数值开启）
    target_kl_raw = os.environ.get("ALPHAGEN_TARGET_KL", "none").strip().lower()
    TARGET_KL: Optional[float]
    if target_kl_raw in {"none", "null", ""}:
        TARGET_KL = None
    else:
        v = float(target_kl_raw)
        TARGET_KL = None if v <= 0 else v

    # 每步惩罚（鼓励更短表达式/更早 SEP），默认 0
    reward_per_step = float(os.environ.get("ALPHAGEN_REWARD_PER_STEP", "0").strip() or 0.0)
    env_wrapper.REWARD_PER_STEP = reward_per_step

    print("=" * 60)
    print("AlphaGen Crypto Factor Mining")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Symbols: {SYMBOLS}")
    print(f"Train: {START_TIME} -> {TRAIN_END}")
    print(f"Val: {TRAIN_END} -> {VAL_END}")
    print(f"Test: {VAL_END} -> {END_TIME}")
    print(f"Features (dynamic): {len(feature_space.feature_cols)}")
    print()

    # ==================== 设置随机种子 ====================
    reseed_everything(SEED)

    # ==================== 加载训练数据 ====================
    print("Loading training data...")
    train_data = CryptoData(
        symbols=SYMBOLS,
        start_time=START_TIME,
        end_time=TRAIN_END,
        timeframe='1h',
        data_dir=DATA_DIR,
        max_backtrack_periods=100,
        max_future_periods=30,
        features=None,  # 使用动态 FeatureType 的全集
        device=device_obj
    )

    print(f"Train data: {train_data.n_days} days, {train_data.n_stocks} symbols, {train_data.n_features} features")

    # ==================== 定义目标 ====================
    # 预测1小时后的收益率
    if "close" not in feature_space.feature_cols:
        raise RuntimeError(
            f"特征列中未找到 close，无法构造目标。请检查 {DATA_DIR} 下 *_train.csv 表头。"
        )
    close_idx = feature_space.feature_cols.index("close")
    close_expr = Feature(sd.FeatureType(close_idx))
    target = Ref(close_expr, -1) / close_expr - 1

    print(f"Target: 1-hour forward return")

    # ==================== 创建Calculator ====================
    print("\nInitializing calculator...")
    if POOL_TYPE == "meanstd":
        calculator = TensorQLibStockDataCalculator(train_data, target)
    else:
        calculator = QLibStockDataCalculator(train_data, target)

    # ==================== 创建Alpha Pool ====================
    # 如果启用动态阈值，初始化用 start（避免刚开始就被“后期阈值”卡死）
    init_ic_lb = ic_lb_start if (ic_lb_start != ic_lb_end) else IC_LOWER_BOUND
    print(
        f"Creating alpha pool (type={POOL_TYPE}, capacity={POOL_CAPACITY}, IC threshold={init_ic_lb})..."
        + (f" [schedule {ic_lb_start}->{ic_lb_end}]" if ic_lb_start != ic_lb_end else "")
    )
    if POOL_TYPE == "meanstd":
        pool = NaNFriendlyMeanStdAlphaPool(
            capacity=POOL_CAPACITY,
            calculator=calculator,  # type: ignore[arg-type]
            ic_lower_bound=init_ic_lb,
            l1_alpha=L1_ALPHA,
            lcb_beta=POOL_LCB_BETA,
            device=device_obj,
        )
    else:
        pool = MseAlphaPool(
            capacity=POOL_CAPACITY,
            calculator=calculator,
            ic_lower_bound=init_ic_lb,
            l1_alpha=L1_ALPHA,
            device=device_obj,
        )

    # ==================== 创建RL环境 ====================
    print("Setting up RL environment...")
    env = AlphaEnv(pool=pool, device=device_obj)

    # ==================== 创建PPO模型 ====================
    print("Creating PPO model...")
    policy_kwargs = dict(
        features_extractor_class=LSTMSharedNet,
        features_extractor_kwargs=dict(
            n_layers=LSTM_LAYERS,
            d_model=LSTM_DIM,
            dropout=LSTM_DROPOUT,
            device=device_obj,
        )
    )

    resume_flag = os.environ.get("ALPHAGEN_RESUME", "0").strip() in {"1", "true", "yes", "y"}
    resume_path = os.environ.get("ALPHAGEN_RESUME_PATH", str(OUTPUT_DIR / "model_final.zip")).strip()
    if resume_flag and Path(resume_path).exists():
        print(f"Resuming PPO model from: {resume_path}")
        model = MaskablePPO.load(
            resume_path,
            env=env,
            device=DEVICE,
        )
        # 允许在恢复训练时微调部分超参（不改网络结构）
        model.ent_coef = ENT_COEF
        model.target_kl = TARGET_KL
        # learning_rate 在 SB3 里可能是 schedule，这里不强行覆盖，避免产生误解
    else:
        if resume_flag and not Path(resume_path).exists():
            print(f"⚠ ALPHAGEN_RESUME=1 但未找到模型文件：{resume_path}（将从头训练）")
        model = MaskablePPO(
            'MlpPolicy',
            env,
            policy_kwargs=policy_kwargs,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            gamma=0.99,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            target_kl=TARGET_KL,
            device=DEVICE,
            verbose=1,
            tensorboard_log=str(OUTPUT_DIR / 'tensorboard')
        )

    # ==================== 开始训练 ====================
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"Monitor training: tensorboard --logdir={OUTPUT_DIR / 'tensorboard'}")
    print()

    callbacks = [TensorboardCallback()]
    if ic_lb_start != ic_lb_end:
        callbacks.append(
            IcLowerBoundScheduleCallback(
                pool=pool,
                total_timesteps=TOTAL_TIMESTEPS,
                start_lb=ic_lb_start,
                end_lb=ic_lb_end,
                update_every=ic_lb_update_every,
            )
        )
    callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback
    )

    # ==================== 保存结果 ====================
    print("\n" + "=" * 60)
    print("Training complete! Saving results...")
    print("=" * 60)

    model_path = OUTPUT_DIR / 'model_final'
    pool_path = OUTPUT_DIR / 'alpha_pool.json'

    model.save(str(model_path))
    _dump_json(pool_path, pool.to_json_dict())

    print(f"✓ Model saved: {model_path}")
    print(f"✓ Alpha pool saved: {pool_path}")
    print(f"✓ Best alphas: {pool.size}")

    # ==================== 验证集评估 ====================
    print("\n" + "=" * 60)
    print("Evaluating on validation set...")
    print("=" * 60)

    val_data = CryptoData(
        symbols=SYMBOLS,
        start_time=TRAIN_END,
        end_time=VAL_END,
        timeframe='1h',
        data_dir=DATA_DIR,
        max_backtrack_periods=100,
        max_future_periods=30,
        features=None,
        device=device_obj,
    )

    if POOL_TYPE == "meanstd":
        val_calculator = TensorQLibStockDataCalculator(val_data, target)
    else:
        val_calculator = QLibStockDataCalculator(val_data, target)

    if pool.size > 0:
        exprs = [e for e in pool.exprs[:pool.size] if e is not None]
        weights = list(pool.weights)
        ic, ric = val_calculator.calc_pool_all_ret(exprs, weights)
        print(f"Validation IC: {ic:.4f}")
        print(f"Validation Rank IC: {ric:.4f}")

        # 保存验证结果
        val_results = {
            'ic': float(ic),
            'rank_ic': float(ric),
            'n_factors': len(exprs),
            'factors': [str(expr) for expr in exprs],
            'weights': [float(w) for w in weights],
            'n_features_total': int(train_data.n_features),
            'feature_columns': feature_space.feature_cols,
        }

        with open(OUTPUT_DIR / 'validation_results.json', 'w', encoding="utf-8") as f:
            json.dump(val_results, f, indent=2)

        print(f"✓ Validation results saved: {OUTPUT_DIR / 'validation_results.json'}")
    else:
        print("⚠ No factors in pool")

    # ==================== 测试集评估 ====================
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    test_data = CryptoData(
        symbols=SYMBOLS,
        start_time=VAL_END,
        end_time=END_TIME,
        timeframe='1h',
        data_dir=DATA_DIR,
        max_backtrack_periods=100,
        max_future_periods=30,
        features=None,
        device=device_obj,
    )

    if POOL_TYPE == "meanstd":
        test_calculator = TensorQLibStockDataCalculator(test_data, target)
    else:
        test_calculator = QLibStockDataCalculator(test_data, target)

    if pool.size > 0:
        exprs = [e for e in pool.exprs[:pool.size] if e is not None]
        weights = list(pool.weights)
        ic, ric = test_calculator.calc_pool_all_ret(exprs, weights)
        print(f"Test IC: {ic:.4f}")
        print(f"Test Rank IC: {ric:.4f}")

        test_results = {
            'ic': float(ic),
            'rank_ic': float(ric),
            'n_factors': len(exprs),
            'factors': [str(expr) for expr in exprs],
            'weights': [float(w) for w in weights],
            'n_features_total': int(train_data.n_features),
            'feature_columns': feature_space.feature_cols,
            'test_start': str(VAL_END),
            'test_end': str(END_TIME),
        }

        with open(OUTPUT_DIR / 'test_results.json', 'w', encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)

        print(f"✓ Test results saved: {OUTPUT_DIR / 'test_results.json'}")
    else:
        print("⚠ No factors in pool (skip test eval)")

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review factors: cat {pool_path}")
    print(f"2. View training logs: tensorboard --logdir={OUTPUT_DIR / 'tensorboard'}")
    print(f"3. Backtest on test set: {VAL_END} -> {END_TIME}")


if __name__ == '__main__':
    main()
