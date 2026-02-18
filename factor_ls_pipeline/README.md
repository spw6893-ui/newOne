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
  --start 2024-07-01 \
  --end 2025-02-15 \
  --timeframe 1h \
  --horizon 1 \
  --long-frac 0.2 \
  --short-frac 0.2 \
  --cost-bps 0
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

