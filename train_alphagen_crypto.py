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
from typing import List, Optional, Sequence, Tuple

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

    # 可选：限制特征数量（避免 action_space>255 或内存压力过大）
    max_n = os.environ.get("ALPHAGEN_FEATURES_MAX", "").strip()
    if max_n:
        n = int(max_n)
        if n > 0 and len(feature_cols) > n:
            keep_core = [c for c in ("open", "high", "low", "close", "volume", "volume_clean", "vwap") if c in feature_cols]
            rest = [c for c in feature_cols if c not in set(keep_core)]
            feature_cols = keep_core + rest[: max(0, n - len(keep_core))]

    return FeatureSpace(feature_cols=feature_cols)


def _install_dynamic_feature_type(feature_cols: Sequence[str]) -> None:
    """
    动态构造 alphagen_qlib.stock_data.FeatureType，使得 AlphaGen 能把“宽表全部因子列”当作可选 Feature。

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
    # 同时暴露“列顺序”，供 CryptoData 做 index->列名映射
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
    SYMBOLS = os.environ.get("ALPHAGEN_SYMBOLS", "top20")  # 或指定列表: ['BTCUSDT', 'ETHUSDT', ...]

    # 时间分割
    START_TIME = os.environ.get("ALPHAGEN_START_TIME", "2020-01-01")
    TRAIN_END = os.environ.get("ALPHAGEN_TRAIN_END", "2024-01-01")
    VAL_END = os.environ.get("ALPHAGEN_VAL_END", "2024-07-01")
    END_TIME = os.environ.get("ALPHAGEN_END_TIME", "2025-02-15")

    # 特征：默认把 alphagen_ready 里的“全部因子列”都扔进 AlphaGen（FeatureType 动态构造）
    feature_space = _detect_feature_space(Path(DATA_DIR))
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
    from alphagen.models.linear_alpha_pool import MseAlphaPool
    from alphagen.rl.env.wrapper import AlphaEnv
    from alphagen.rl.policy import LSTMSharedNet
    from alphagen.utils import reseed_everything
    from sb3_contrib.ppo_mask import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback

    from AlphaQCM.alphagen_qlib.crypto_data import CryptoData
    from AlphaQCM.alphagen_qlib.calculator import QLibStockDataCalculator
    import AlphaQCM.alphagen_qlib.stock_data as sd

    import alphagen as alphagen_pkg
    print(f"alphagen 包路径: {getattr(alphagen_pkg, '__file__', 'UNKNOWN')}")

    class TensorboardCallback(BaseCallback):
        """记录训练指标到TensorBoard"""

        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            return True

    # 训练配置
    SEED = int(os.environ.get("ALPHAGEN_SEED", "42"))
    BATCH_SIZE = int(os.environ.get("ALPHAGEN_BATCH_SIZE", "128"))
    TOTAL_TIMESTEPS = int(os.environ.get("ALPHAGEN_TOTAL_TIMESTEPS", "100000"))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_obj = torch.device(DEVICE)

    # 输出配置
    OUTPUT_DIR = Path('./alphagen_output')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Alpha Pool配置
    POOL_CAPACITY = 10
    IC_LOWER_BOUND = 0.01

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
    calculator = QLibStockDataCalculator(train_data, target)

    # ==================== 创建Alpha Pool ====================
    print(f"Creating alpha pool (capacity={POOL_CAPACITY}, IC threshold={IC_LOWER_BOUND})...")
    pool = MseAlphaPool(
        capacity=POOL_CAPACITY,
        calculator=calculator,
        ic_lower_bound=IC_LOWER_BOUND,
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
            n_layers=2,
            d_model=128,
            dropout=0.1,
            device=device_obj,
        )
    )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        batch_size=BATCH_SIZE,
        learning_rate=3e-4,
        n_steps=2048,
        gamma=0.99,
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

    callback = TensorboardCallback()

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

        import json
        with open(OUTPUT_DIR / 'validation_results.json', 'w') as f:
            json.dump(val_results, f, indent=2)

        print(f"✓ Validation results saved: {OUTPUT_DIR / 'validation_results.json'}")
    else:
        print("⚠ No factors in pool")

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review factors: cat {pool_path}")
    print(f"2. View training logs: tensorboard --logdir={OUTPUT_DIR / 'tensorboard'}")
    print(f"3. Backtest on test set: {VAL_END} -> {END_TIME}")


if __name__ == '__main__':
    main()
