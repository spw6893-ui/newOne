#!/bin/bash
set -e

echo "用法：每 30 秒打印一次最新标量（离线，不用网页）"
echo ""

PYTHON="${PYTHON:-python3}"
INTERVAL="${INTERVAL:-30}"
LOG_ROOT="${LOG_ROOT:-AlphaQCM/AlphaQCM_data/crypto_logs}"

while true; do
  date
  $PYTHON AlphaQCM/tools/tb_tail.py --log-root "$LOG_ROOT" || true
  echo "----"
  sleep "$INTERVAL"
done

