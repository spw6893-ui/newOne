#!/bin/bash
# AlphaGen 训练快速启动脚本 (优化版)
set -e

echo "=========================================="
echo "AlphaGen 优化训练流程"
echo "=========================================="
echo ""

# 加载优化配置
if [ -f "alphagen_config.sh" ]; then
    source alphagen_config.sh
    echo "✓ 已加载优化配置"
else
    echo "⚠ 未找到 alphagen_config.sh，使用默认配置"
fi

echo ""

# 步骤1: 特征重要性分析 (可选)
read -p "是否先分析特征重要性? (y/N): " analyze
if [[ "$analyze" =~ ^[Yy]$ ]]; then
    echo ""
    echo "=========================================="
    echo "步骤1: 分析特征重要性"
    echo "=========================================="
    python3 analyze_features.py
    echo ""
    read -p "查看分析结果后按回车继续..."
fi

# 步骤2: 运行训练
echo ""
echo "=========================================="
echo "步骤2: 开始训练"
echo "=========================================="
echo "当前配置:"
echo "  - LSTM: ${ALPHAGEN_LSTM_LAYERS:-3}层 x ${ALPHAGEN_LSTM_DIM:-256}维"
echo "  - Pool: 容量${ALPHAGEN_POOL_CAPACITY:-20}, IC阈值${ALPHAGEN_IC_LOWER_BOUND:-0.005}"
echo "  - 特征: 最多${ALPHAGEN_FEATURES_MAX:-50}个"
echo "  - 训练步数: ${ALPHAGEN_TOTAL_TIMESTEPS:-200000}"
echo ""

./run_training.sh

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  1. 特征分析: cat ./alphagen_output/feature_importance.csv"
echo "  2. 因子池: cat ./alphagen_output/alpha_pool.json"
echo "  3. 验证结果: cat ./alphagen_output/validation_results.json"
echo ""
echo "监控训练:"
echo "  tensorboard --logdir=./alphagen_output/tensorboard"
