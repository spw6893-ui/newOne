#!/bin/bash
set -e

echo "=========================================="
echo "AlphaGen Crypto Factor Mining - One-Click"
echo "=========================================="
echo ""

# 配置参数
TRAIN_END="2024-01-01 00:00:00+00:00"
VAL_END="2024-07-01 00:00:00+00:00"

# 检查数据文件
if [ ! -f "AlphaQCM_data/final_dataset_filtered.parquet" ]; then
    echo "❌ 训练数据不存在，请先重组数据:"
    echo "   cd AlphaQCM_data"
    echo "   cat final_dataset_filtered.parquet.gz.part_* > final_dataset_filtered.parquet.gz"
    echo "   gunzip final_dataset_filtered.parquet.gz"
    exit 1
fi

echo "✓ 训练数据已就绪"
echo ""

# Step 1: 准备AlphaGen训练数据
echo "=========================================="
echo "Step 1: 准备AlphaGen训练数据"
echo "=========================================="

if [ ! -d "AlphaQCM_data/alphagen_ready" ] || [ -z "$(ls -A AlphaQCM_data/alphagen_ready 2>/dev/null)" ]; then
    echo "准备训练数据..."
    python AlphaQCM/data_collection/prepare_alphagen_training_data.py \
        --input-dir AlphaQCM_data/final_dataset \
        --output-dir AlphaQCM_data/alphagen_ready \
        --horizon-hours 1 \
        --filter-quality \
        --impute ffill \
        --ffill-limit 24 \
        --train-end "$TRAIN_END" \
        --val-end "$VAL_END"
    echo "✓ 数据准备完成"
else
    echo "✓ AlphaGen训练数据已存在，跳过准备步骤"
fi

echo ""

# Step 2: 检查依赖
echo "=========================================="
echo "Step 2: 检查依赖"
echo "=========================================="

if ! python -c "import torch; import sb3_contrib" 2>/dev/null; then
    echo "安装依赖..."
    pip install -q -r alphagen/requirements.txt
    pip install -q -r AlphaQCM/requirements_crypto.txt
    echo "✓ 依赖安装完成"
else
    echo "✓ 依赖已安装"
fi

echo ""

# Step 3: 开始训练
echo "=========================================="
echo "Step 3: 开始AlphaGen训练"
echo "=========================================="
echo "训练配置:"
echo "  - 训练集: 2020-01-01 -> $TRAIN_END"
echo "  - 验证集: $TRAIN_END -> $VAL_END"
echo "  - 测试集: $VAL_END -> 2025-02-15"
echo ""
echo "监控训练进度:"
echo "  tensorboard --logdir=./alphagen_output/tensorboard"
echo ""

python train_alphagen_crypto.py

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo ""
echo "输出文件:"
echo "  - 模型: ./alphagen_output/model_final.zip"
echo "  - 因子池: ./alphagen_output/alpha_pool.json"
echo "  - 验证结果: ./alphagen_output/validation_results.json"
echo ""
echo "查看结果:"
echo "  cat ./alphagen_output/alpha_pool.json"
echo "  cat ./alphagen_output/validation_results.json"
