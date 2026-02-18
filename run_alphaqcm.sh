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
  exit 0
fi

MODEL="${1:-qrdqn}"
SYMBOLS="${2:-top10}"
TIMEFRAME="${3:-1h}"
POOL="${4:-20}"
TARGET_PERIODS="${5:-20}"

TRAIN_START="${QCM_TRAIN_START:-2020-01-01}"
TRAIN_END="${QCM_TRAIN_END:-2023-12-31}"
VALID_START="${QCM_VALID_START:-2024-01-01}"
VALID_END="${QCM_VALID_END:-2024-06-30}"
TEST_START="${QCM_TEST_START:-2024-07-01}"
TEST_END="${QCM_TEST_END:-2024-12-31}"

DATA_DIR="AlphaQCM/AlphaQCM_data/crypto_data"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR"/*_${TIMEFRAME}.csv 2>/dev/null)" ]; then
  echo "❌ 未找到 AlphaQCM crypto 数据：$DATA_DIR（期望 *_${TIMEFRAME}.csv）"
  echo "   解决：进入 AlphaQCM 后运行："
  echo "     python3 data_collection/fetch_crypto_data.py"
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

