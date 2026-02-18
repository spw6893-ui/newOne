#!/usr/bin/env bash
set -euo pipefail

# 一键回测：Top100 截面分位分组（10 组，多 5 组 / 空 5 组）
#
# 用法：
#   ./factor_ls_pipeline/run_backtest_top100.sh alphagen_output/alpha_pool.json
#
# 可选环境变量覆盖：
#   DATA_DIR   默认 AlphaQCM/AlphaQCM_data/alphagen_ready
#   SYMBOLS    默认 top100
#   START/END  默认 2024-07-01 / 2025-02-15
#   TIMEFRAME  默认 1h
#   HORIZON    默认 1
#   QUANTILES  默认 10
#   TOP_K      默认 5
#   BOTTOM_K   默认 5
#   OUTPUT_DIR 默认 factor_ls_output

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <factor_file>"
  exit 2
fi

FACTOR_FILE="$1"
DATA_DIR="${DATA_DIR:-AlphaQCM/AlphaQCM_data/alphagen_ready}"
SYMBOLS="${SYMBOLS:-top100}"
START="${START:-2024-07-01}"
END="${END:-2025-02-15}"
TIMEFRAME="${TIMEFRAME:-1h}"
HORIZON="${HORIZON:-1}"
QUANTILES="${QUANTILES:-10}"
TOP_K="${TOP_K:-5}"
BOTTOM_K="${BOTTOM_K:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-factor_ls_output}"

python3 factor_ls_pipeline/run_ls_backtest.py \
  --factor-file "$FACTOR_FILE" \
  --data-dir "$DATA_DIR" \
  --symbols "$SYMBOLS" \
  --start "$START" \
  --end "$END" \
  --timeframe "$TIMEFRAME" \
  --horizon "$HORIZON" \
  --quantiles "$QUANTILES" \
  --top-k "$TOP_K" \
  --bottom-k "$BOTTOM_K" \
  --output-dir "$OUTPUT_DIR"

