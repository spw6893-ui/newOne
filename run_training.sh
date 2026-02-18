#!/bin/bash
set -e

echo "=========================================="
echo "AlphaGen Crypto Factor Mining - One-Click"
echo "=========================================="
echo ""

# 可选：指定 Python 解释器（强烈建议用 venv/conda 的 python）
PYTHON="${PYTHON:-python3}"
PIP="$PYTHON -m pip"

# 确保 alphagen 子模块已拉取（否则会出现 `No module named alphagen.rl.env`）
if [ ! -f "alphagen/requirements.txt" ]; then
    echo "检测到 alphagen 子模块未就绪，正在初始化..."
    git submodule update --init --recursive
    echo "✓ alphagen 子模块就绪"
    echo ""
fi

# 配置参数
TRAIN_END="2024-01-01 00:00:00+00:00"
VAL_END="2024-07-01 00:00:00+00:00"
# 允许外部覆盖（例如你只复原到了 final_dataset/ 目录）
INPUT_DIR="${INPUT_DIR:-AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
# 训练 CSV 的 schema 策略：
# - per-file：保持每个币自己的列（默认，兼容旧输出）
# - union：全币种并集（“全因子”）
# - intersection：全币种交集（NaN 最少，适合先跑通）
SCHEMA_MODE="${SCHEMA_MODE:-per-file}"

# 检查数据目录（Vision 基底 + metrics 覆盖完整 85 币种）
if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A "$INPUT_DIR"/*_final.csv 2>/dev/null)" ]; then
    echo "❌ 最终宽表目录不存在或为空：$INPUT_DIR"
    echo "   请先生成 Vision 基底数据集（final_dataset_vision_metrics85）"
    exit 1
fi

echo "✓ 训练数据已就绪"
echo ""

# Step 1: 准备AlphaGen训练数据
echo "=========================================="
echo "Step 1: 准备AlphaGen训练数据"
echo "=========================================="

INPUT_DIR_FOR_PREP="$INPUT_DIR"
if [[ "$INPUT_DIR_FOR_PREP" == AlphaQCM/* ]]; then
    INPUT_DIR_FOR_PREP="${INPUT_DIR_FOR_PREP#AlphaQCM/}"
fi

if [ "$FORCE_REBUILD" = "1" ] || [ ! -d "AlphaQCM/AlphaQCM_data/alphagen_ready" ] || [ -z "$(ls -A AlphaQCM/AlphaQCM_data/alphagen_ready 2>/dev/null)" ]; then
    echo "准备训练数据..."
    $PYTHON AlphaQCM/data_collection/prepare_alphagen_training_data.py \
        --input-dir "$INPUT_DIR_FOR_PREP" \
        --output-dir AlphaQCM_data/alphagen_ready \
        --horizon-hours 1 \
        --filter-quality \
        --impute ffill \
        --ffill-limit 24 \
        --schema "$SCHEMA_MODE" \
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

if ! $PYTHON -c "import torch, gymnasium, stable_baselines3, sb3_contrib" 2>/dev/null; then
    echo "安装依赖（最小集，避免 alphagen/requirements.txt 的过老版本）..."
    $PIP install -q -U pip setuptools wheel
    $PIP install -q -r requirements_alphagen_train.txt
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

# libgomp 对 OMP_NUM_THREADS 要求是正整数；有些平台会注入非整数值导致告警
if [ -n "${OMP_NUM_THREADS:-}" ] && ! [[ "${OMP_NUM_THREADS}" =~ ^[0-9]+$ ]]; then
    echo "⚠ OMP_NUM_THREADS=${OMP_NUM_THREADS} 不是整数，已 unset（避免 libgomp 警告）"
    unset OMP_NUM_THREADS
fi

$PYTHON train_alphagen_crypto.py

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
