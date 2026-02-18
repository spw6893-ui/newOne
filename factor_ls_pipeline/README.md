# 因子多空回测管线（独立）

这个小工程用于：**读取 AlphaGen / AlphaQCM 输出的因子表达式与权重**，在你准备好的 `alphagen_ready` 宽表数据上做**截面多空组合**，并输出回撤/收益等指标。

特点：

- 不依赖网页 TensorBoard，直接在命令行输出 + 落盘 CSV
- 因子表达式支持两种常见格式：
  - AlphaGen（submodule `alphagen`）风格：`Mul(-5.0,$seg_head_logret)`
  - AlphaQCM 内置 alphagen 风格：`Div(Constant(-2.0),$qrs_beta_close_per_hour)`

## 输入

### 1) 因子文件

支持两类：

- `alpha_pool.json`（AlphaGen 训练脚本保存的）
  - 形如：`{"exprs":[...], "weights":[...]}`
- `validation_results.json`（AlphaGen 训练脚本保存的）
  - 形如：`{"factors":[...], "weights":[...]}`

也支持从 AlphaQCM 的 `valid_best_table.csv` / `test_best_table.csv` 读取（需要包含 `exprs`、`weight` 列）。

### 2) 数据目录

默认使用宽表：`AlphaQCM/AlphaQCM_data/alphagen_ready`（每币一份 `*_train.csv`）。

## 快速开始

以测试集（2024-07-01 -> 2025-02-15）做回测：

```bash
python3 factor_ls_pipeline/run_ls_backtest.py \
  --factor-file alphagen_output/alpha_pool.json \
  --data-dir AlphaQCM/AlphaQCM_data/alphagen_ready \
  --symbols top100 \
  --start 2024-07-01 \
  --end 2025-02-15 \
  --timeframe 1h \
  --horizon 1 \
  --long-frac 0.2 \
  --short-frac 0.2 \
  --cost-bps 0
```

### 分位数组合（推荐：10 组，做多 5 组 / 做空 5 组）

你提到的“多 5 组空 5 组”，对应 **10 分组（Q1..Q10）**，然后做：

- **多头**：Q6..Q10（最高 5 组）
- **空头**：Q1..Q5（最低 5 组）
- **多空**：平均(Q6..Q10) - 平均(Q1..Q5)

命令：

```bash
python3 factor_ls_pipeline/run_ls_backtest.py \
  --factor-file alphagen_output/alpha_pool.json \
  --data-dir AlphaQCM/AlphaQCM_data/alphagen_ready \
  --symbols top100 \
  --start 2024-07-01 \
  --end 2025-02-15 \
  --timeframe 1h \
  --horizon 1 \
  --quantiles 10 \
  --top-k 5 \
  --bottom-k 5
```

也可以直接一键运行（同样是 top100 + 10 分组 + 多 5 空 5）：

```bash
./factor_ls_pipeline/run_backtest_top100.sh alphagen_output/alpha_pool.json
```

如果你的因子来自 `validation_results.json`：

```bash
python3 factor_ls_pipeline/run_ls_backtest.py \
  --factor-file alphagen_output/validation_results.json \
  --data-dir AlphaQCM/AlphaQCM_data/alphagen_ready
```

## 输出

默认输出到 `factor_ls_output/`：

- `factor_ls_output/curve.csv`：净值曲线、组合收益、回撤、换手等
- `factor_ls_output/summary.json`：年化收益/波动/夏普/最大回撤等汇总指标
- `factor_ls_output/quantiles.csv`：分位数组合收益（q1..qN）+ 多空序列（如 `ls_top5_bot5`），以及对应的 `*_equity` / `*_drawdown`
- `factor_ls_output/quantiles_summary.json`：分位多空序列的汇总指标（年化、夏普、最大回撤等）

## 选币逻辑（--symbols top100）

优先读取 `AlphaQCM/data_collection/top100_symbols.txt`（按市值排序的 top100 列表），并自动映射为宽表里常见的文件名前缀（例如 `BTC/USDT:USDT` -> `BTC_USDT:USDT`）。如果匹配不足，会用 `data-dir` 下实际存在的币种补齐，保证可跑。
