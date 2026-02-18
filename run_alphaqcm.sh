#!/bin/bash
set -e

echo "=========================================="
echo "AlphaQCM Crypto Factor Mining - One-Click"
echo "=========================================="
echo ""

PYTHON="${PYTHON:-python3}"
PIP="$PYTHON -m pip"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  echo "用法:"
  echo "  ./run_alphaqcm.sh [model] [symbols] [timeframe] [pool] [target_periods]"
  echo ""
  echo "示例:"
  echo "  ./run_alphaqcm.sh"
  echo "  ./run_alphaqcm.sh qrdqn top20 1h 20 20"
  echo "  ./run_alphaqcm.sh iqn top100 1h 50 20"
  echo ""
  echo "可选：用环境变量覆盖时间切分（与 AlphaQCM/train_qcm_crypto.py 参数对应）:"
  echo "  QCM_TRAIN_START, QCM_TRAIN_END, QCM_VALID_START, QCM_VALID_END, QCM_TEST_START, QCM_TEST_END"
  echo ""
  echo "可选：数据源与特征预筛选（与 AlphaGen 对齐）:"
  echo "  QCM_DATA_DIR=AlphaQCM/AlphaQCM_data/alphagen_ready  # 默认"
  echo "  ALPHAGEN_FEATURES_MAX=20  # 可选，缩小 action space"
  exit 0
fi

MODEL="${1:-qrdqn}"
SYMBOLS="${2:-top10}"
TIMEFRAME="${3:-1h}"
POOL="${4:-20}"
TARGET_PERIODS="${5:-1}"

TRAIN_START="${QCM_TRAIN_START:-2020-01-01}"
TRAIN_END="${QCM_TRAIN_END:-2024-01-01 00:00:00+00:00}"
VALID_START="${QCM_VALID_START:-2024-01-01 00:00:00+00:00}"
VALID_END="${QCM_VALID_END:-2024-07-01 00:00:00+00:00}"
TEST_START="${QCM_TEST_START:-2024-07-01 00:00:00+00:00}"
TEST_END="${QCM_TEST_END:-2025-02-15}"

DATA_DIR="${QCM_DATA_DIR:-AlphaQCM/AlphaQCM_data/alphagen_ready}"
if [ ! -d "$DATA_DIR" ]; then
  echo "❌ 数据目录不存在：$DATA_DIR"
  exit 1
fi
if ls -A "$DATA_DIR"/*_train.csv >/dev/null 2>&1; then
  :
elif ls -A "$DATA_DIR"/*_"$TIMEFRAME".csv >/dev/null 2>&1; then
  :
else
  echo "❌ 数据目录无可用 CSV：$DATA_DIR（期望 *_train.csv 或 *_${TIMEFRAME}.csv）"
  echo "   如果你想用 OHLCV 训练，请设置：QCM_DATA_DIR=AlphaQCM/AlphaQCM_data/crypto_data"
  exit 1
fi

echo "配置:"
echo "  - model=$MODEL"
echo "  - symbols=$SYMBOLS"
echo "  - timeframe=$TIMEFRAME"
echo "  - pool=$POOL"
echo "  - target_periods=$TARGET_PERIODS"
echo "  - train: $TRAIN_START -> $TRAIN_END"
echo "  - valid: $VALID_START -> $VALID_END"
echo "  - test : $TEST_START -> $TEST_END"
echo "  - data_dir=$DATA_DIR"
echo "  - ALPHAGEN_FEATURES_MAX=${ALPHAGEN_FEATURES_MAX:-0}"
echo ""

if ! $PYTHON -c "import torch, yaml, pandas, numpy, gymnasium" 2>/dev/null; then
  echo "安装依赖（AlphaQCM/requirements_crypto.txt）..."
  $PIP install -q -U pip setuptools wheel
  $PIP install -q -r AlphaQCM/requirements_crypto.txt
  echo "✓ 依赖安装完成"
else
  echo "✓ 依赖已安装"
fi

echo ""
echo "开始训练..."
$PYTHON AlphaQCM/train_qcm_crypto.py \
  --model "$MODEL" \
  --symbols "$SYMBOLS" \
  --timeframe "$TIMEFRAME" \
  --pool "$POOL" \
  --target-periods "$TARGET_PERIODS" \
  --data-dir "$DATA_DIR" \
  --train-start "$TRAIN_START" \
  --train-end "$TRAIN_END" \
  --valid-start "$VALID_START" \
  --valid-end "$VALID_END" \
  --test-start "$TEST_START" \
  --test-end "$TEST_END"

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo "日志目录: AlphaQCM/AlphaQCM_data/crypto_logs/${SYMBOLS}_${TIMEFRAME}/"
