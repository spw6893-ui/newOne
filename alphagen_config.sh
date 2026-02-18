#!/bin/bash
# AlphaGen 训练配置 - 针对70+高频因子优化

# 模型架构
export ALPHAGEN_LSTM_LAYERS=3          # LSTM层数 (默认2→3)
export ALPHAGEN_LSTM_DIM=256           # LSTM维度 (默认128→256)
export ALPHAGEN_LSTM_DROPOUT=0.2       # Dropout率 (默认0.1→0.2)

# Alpha Pool
export ALPHAGEN_POOL_CAPACITY=20       # 因子池容量 (默认10→20)
export ALPHAGEN_IC_LOWER_BOUND=0.005   # IC阈值 (默认0.01→0.005)
export ALPHAGEN_POOL_L1_ALPHA=0.01     # L1正则化 (默认0.005→0.01)

# PPO训练
export ALPHAGEN_BATCH_SIZE=128
export ALPHAGEN_TOTAL_TIMESTEPS=200000 # 增加训练步数
export ALPHAGEN_LEARNING_RATE=2e-4     # 降低学习率 (默认3e-4→2e-4)

# 特征选择
export ALPHAGEN_FEATURES_MAX=50        # 限制最多使用50个特征 (从70+中筛选)
export ALPHAGEN_FEATURE_SCHEMA_MODE=union  # 使用全因子模式

# 数据配置
export ALPHAGEN_DATA_DIR="AlphaQCM/AlphaQCM_data/alphagen_ready"
export ALPHAGEN_SYMBOLS="top20"
export ALPHAGEN_START_TIME="2020-01-01"
export ALPHAGEN_TRAIN_END="2024-01-01"
export ALPHAGEN_VAL_END="2024-07-01"
export ALPHAGEN_END_TIME="2025-02-15"

echo "AlphaGen配置已加载 (针对高维特征优化)"
echo "  - LSTM: ${ALPHAGEN_LSTM_LAYERS}层 x ${ALPHAGEN_LSTM_DIM}维"
echo "  - Pool: 容量${ALPHAGEN_POOL_CAPACITY}, IC阈值${ALPHAGEN_IC_LOWER_BOUND}"
echo "  - 特征: 最多${ALPHAGEN_FEATURES_MAX}个 (从70+中自动筛选)"
