#!/usr/bin/env python3
"""
AlphaGen加密货币因子训练脚本
使用AlphaQCM的crypto_data和calculator适配器
"""
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / 'alphagen'))
sys.path.insert(0, str(Path(__file__).parent / 'AlphaQCM'))

import torch
import numpy as np
from datetime import datetime

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.core import AlphaEnvCore
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen_qlib.crypto_data import CryptoData, FeatureType
from alphagen_qlib.calculator import QLibStockDataCalculator


class TensorboardCallback(BaseCallback):
    """记录训练指标到TensorBoard"""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True


def main():
    # ==================== 配置参数 ====================

    # 数据配置
    # 训练数据默认落在 AlphaQCM/AlphaQCM_data/alphagen_ready（由 run_training.sh / prepare_alphagen_training_data.py 生成）
    DATA_DIR = 'AlphaQCM/AlphaQCM_data/alphagen_ready'
    SYMBOLS = 'top20'  # 或指定列表: ['BTCUSDT', 'ETHUSDT', ...]

    # 时间分割
    START_TIME = '2020-01-01'
    TRAIN_END = '2024-01-01'
    VAL_END = '2024-07-01'
    END_TIME = '2025-02-15'

    # 特征选择(从73个过滤后的特征中选择)
    FEATURES = [
        FeatureType.CLOSE,
        FeatureType.VOLUME,
        # 添加更多特征...
    ]

    # 训练配置
    SEED = 42
    BATCH_SIZE = 128
    TOTAL_TIMESTEPS = 100000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        features=FEATURES,
        device=torch.device(DEVICE)
    )

    print(f"Train data: {train_data.n_days} days, {train_data.n_stocks} symbols, {train_data.n_features} features")

    # ==================== 定义目标 ====================
    # 预测1小时后的收益率
    target = Ref(Close(), -1) / Close() - 1

    print(f"Target: 1-hour forward return")

    # ==================== 创建Calculator ====================
    print("\nInitializing calculator...")
    calculator = QLibStockDataCalculator(train_data, target)

    # ==================== 创建Alpha Pool ====================
    print(f"Creating alpha pool (capacity={POOL_CAPACITY}, IC threshold={IC_LOWER_BOUND})...")
    pool = AlphaPool(
        capacity=POOL_CAPACITY,
        calculator=calculator,
        ic_lower_bound=IC_LOWER_BOUND
    )

    # ==================== 创建RL环境 ====================
    print("Setting up RL environment...")
    env_core = AlphaEnvCore(calculator=calculator, pool=pool)
    env = AlphaEnv(env_core)

    # ==================== 创建PPO模型 ====================
    print("Creating PPO model...")
    policy_kwargs = dict(
        features_extractor_class=LSTMSharedNet,
        features_extractor_kwargs=dict(
            n_layers=2,
            d_model=128
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
    pool.save(str(pool_path))

    print(f"✓ Model saved: {model_path}")
    print(f"✓ Alpha pool saved: {pool_path}")
    print(f"✓ Best alphas: {len(pool.exprs)}")

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
        features=FEATURES,
        device=torch.device(DEVICE)
    )

    val_calculator = QLibStockDataCalculator(val_data, target)

    if len(pool.exprs) > 0:
        ic, ric = val_calculator.calc_pool_all_ret(pool.exprs, pool.weights)
        print(f"Validation IC: {ic:.4f}")
        print(f"Validation Rank IC: {ric:.4f}")

        # 保存验证结果
        val_results = {
            'ic': float(ic),
            'rank_ic': float(ric),
            'n_factors': len(pool.exprs),
            'factors': [str(expr) for expr in pool.exprs],
            'weights': [float(w) for w in pool.weights]
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
