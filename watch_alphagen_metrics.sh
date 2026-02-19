#!/bin/bash
set -e

# 在服务器上不用网页也能“实时看两个指标”：
# - 默认看 eval/val_ic 与 eval/test_ic
# - 默认从 alphagen_output/tensorboard 里递归找最新事件文件

PATH_TO_WATCH="${1:-alphagen_output/tensorboard}"

python3 tools/watch_tb_scalars.py --path "$PATH_TO_WATCH" --tags eval/val_ic eval/test_ic --every 10

