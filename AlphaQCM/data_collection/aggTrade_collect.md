# aggTrades 采集与聚合说明（Binance Vision）

本文档说明本仓库如何从 **Binance Vision** 归档站点采集 `aggTrades`（逐笔聚合成交）并聚合为可用于 1H 策略的订单流特征（CVD 等）。

> 口径约定：本文所有时间戳均为 **UTC**。

---

## 1. 目标与产物（你最终会得到什么）

本工程对 `aggTrades` 采用“三层结构”，目的是：**避免回测/反复特征工程时重新扫逐笔大文件**。

1) **月度 ZIP（源站）**
- 仅作为下载/解析输入，默认不长期保存（节省空间）

2) **1m 中间层（Parquet，推荐复用入口）**
- 路径：`AlphaQCM/AlphaQCM_data/binance_aggTrades_1m_parquet/{SYMBOL}/{YYYY-MM}.parquet`
- 粒度：1 分钟
- 用途：后续你想重新聚合 1H / 2H / 4H，或计算新的分钟级订单流因子时，直接复用

3) **1h 因子（CSV，最终喂给宽表）**
- 路径：`AlphaQCM/AlphaQCM_data/binance_aggTrades/{SYMBOL}_aggTrades.csv` + `*.meta.json`
- 粒度：1 小时
- 用途：在 `build_final_dataset.py` 中按 symbol join，统一加 `at_` 前缀写入最终宽表

---

## 2. 数据源（Binance Vision URL 规则）

UM（USDT 本位永续）月度 `aggTrades` 的标准格式：

```
https://data.binance.vision/data/futures/um/monthly/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-YYYY-MM.zip
```

例如：

```
https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-12.zip
```

---

## 3. 入口脚本与并发运行方式

### 3.1 单币/单进程（底层函数）

核心下载与处理入口：
- `AlphaQCM/data_collection/download_binance_efficient.py`
  - `download_symbol_range(..., data_type="aggTrades", ...)`

它会按月下载 ZIP，并执行：
1) ZIP 内流式读取 CSV（不落地大 CSV）
2) 逐笔 → 1m 中间层（Parquet）
3) 1m → 1h 因子（CSV）

### 3.2 多 worker 并发队列（推荐）

脚本：
- `AlphaQCM/data_collection/run_aggtrades_queue.py`

特点：
- 多个 worker 共享一个队列 state（JSON + 文件锁）
- 断点续跑（机器挂了/进程被 kill 后，超时未更新的任务会回滚为 pending）
- `--skip-existing` 会优先检查 `*_aggTrades.csv.meta.json`，已完成就跳过

典型用法（3 worker 并发）：

```
tmux new-session -d -s cq_aggA "stdbuf -oL -eL python3 -u AlphaQCM/data_collection/run_aggtrades_queue.py --worker-id A --symbols-file AlphaQCM/data_collection/top100_perp_symbols.txt --market um --start 2020-01-01 --end 2025-02-15 --state AlphaQCM_data/_tmp/aggTrades_queue_state.json --aggtrades-1m-dir AlphaQCM_data/binance_aggTrades_1m_parquet --output-dir AlphaQCM_data/binance_aggTrades --temp-dir AlphaQCM_data/_tmp/binance_vision --skip-existing |& tee -a AlphaQCM/AlphaQCM_data/_tmp/logs/aggTrades_queue_A.log"
tmux new-session -d -s cq_aggB "stdbuf -oL -eL python3 -u AlphaQCM/data_collection/run_aggtrades_queue.py --worker-id B --symbols-file AlphaQCM/data_collection/top100_perp_symbols.txt --market um --start 2020-01-01 --end 2025-02-15 --state AlphaQCM_data/_tmp/aggTrades_queue_state.json --aggtrades-1m-dir AlphaQCM_data/binance_aggTrades_1m_parquet --output-dir AlphaQCM_data/binance_aggTrades --temp-dir AlphaQCM_data/_tmp/binance_vision --skip-existing |& tee -a AlphaQCM/AlphaQCM_data/_tmp/logs/aggTrades_queue_B.log"
tmux new-session -d -s cq_aggC "stdbuf -oL -eL python3 -u AlphaQCM/data_collection/run_aggtrades_queue.py --worker-id C --symbols-file AlphaQCM/data_collection/top100_perp_symbols.txt --market um --start 2020-01-01 --end 2025-02-15 --state AlphaQCM_data/_tmp/aggTrades_queue_state.json --aggtrades-1m-dir AlphaQCM_data/binance_aggTrades_1m_parquet --output-dir AlphaQCM_data/binance_aggTrades --temp-dir AlphaQCM_data/_tmp/binance_vision --skip-existing |& tee -a AlphaQCM/AlphaQCM_data/_tmp/logs/aggTrades_queue_C.log"
```

监控（可选）：
- `AlphaQCM/data_collection/monitor_aggtrades_progress.py`

---

## 4. ZIP → 1m（Parquet 中间层）聚合逻辑

实现函数：
- `process_aggtrades_to_1m(...)`（`AlphaQCM/data_collection/download_binance_efficient.py`）

### 4.1 输入 CSV 列含义（Vision aggTrades）

`aggTrades` 归档 CSV 在不同月份可能“无表头/有表头”，但列顺序一致：

| 列序号 | 字段 | 说明 |
|---:|---|---|
| 1 | `price` | 成交价 |
| 2 | `quantity` | 成交量（基础币数量） |
| 5 | `transact_time` | 成交时间（毫秒时间戳或 ISO） |
| 6 | `is_buyer_maker` | 是否买方为 Maker |

时间解析：
- `transact_time` 统一转 `UTC datetime`，再取 `floor('min')` 得到分钟桶。

### 4.2 主动方向口径（核心）

Binance 的约定：
- `is_buyer_maker = True`：买方是 Maker ⇒ 卖方是 Taker ⇒ **该笔属于“主动卖出”**
- 因此：
  - `taker_buy = ~is_buyer_maker`
  - `taker_sell = is_buyer_maker`

### 4.3 分钟级聚合字段

对每分钟做 sum 聚合，得到（按 minute 索引）：

- 成交与量：
  - `trade_count`
  - `base_volume`
  - `quote_volume = price * quantity`
- 主动买卖（qty/quote 两套）：
  - `taker_buy_qty`, `taker_sell_qty`
  - `taker_buy_quote`, `taker_sell_quote`
- HHI 需要的平方和（sum(amount^2)）：
  - `taker_buy_quote_sq`, `taker_sell_quote_sq`
- 交易笔数（买/卖拆分）：
  - `taker_buy_trade_count`, `taker_sell_trade_count`
- 大小单拆分（按单笔 quote_amount 判断）：
  - `whale_*`：单笔 `quote_amount >= 50,000 USD`
  - `retail_*`：单笔 `quote_amount <= 1,000 USD`
  - 输出：`whale_buy_quote`, `whale_sell_quote`, `retail_buy_quote`, `retail_sell_quote`
- 分钟价格路径（用于后续构造小时 OHLC）：
  - `first_ts`, `price_first`
  - `last_ts`, `price_last`
  - `price_high`, `price_low`

### 4.4 中间层落盘

每月写一个 Parquet：

```
AlphaQCM/AlphaQCM_data/binance_aggTrades_1m_parquet/{SYMBOL}/{YYYY-MM}.parquet
```

复用策略：
- 如果该 Parquet 已存在，会直接读取复用，避免重复下载/重复处理逐笔。

---

## 5. 1m → 1h（订单流因子）聚合逻辑

实现函数：
- `process_aggtrades_1m_to_hourly(minute_df)`（`AlphaQCM/data_collection/download_binance_efficient.py`）

### 5.1 小时聚合（sum + 价格路径）

按 `hour = minute.floor('h')` 聚合：

- sum 类（直接相加）：
  - `trade_count`, `base_volume`, `quote_volume`
  - `taker_buy_qty`, `taker_sell_qty`
  - `taker_buy_quote`, `taker_sell_quote`
  - `taker_buy_quote_sq`, `taker_sell_quote_sq`
  - `taker_buy_trade_count`, `taker_sell_trade_count`
  - `whale_buy_quote`, `whale_sell_quote`, `retail_buy_quote`, `retail_sell_quote`

- 小时 OHLC（由分钟 first/last/high/low 构造）：
  - `price_open`：该小时第一分钟的 `price_first`
  - `price_close`：该小时最后一分钟的 `price_last`
  - `price_high`：小时内分钟 `price_high` 的 max
  - `price_low`：小时内分钟 `price_low` 的 min

### 5.2 小时衍生因子（核心输出）

- CVD：
  - `cvd_qty = taker_buy_qty - taker_sell_qty`
  - `cvd_quote = taker_buy_quote - taker_sell_quote`
- VWAP：
  - `vwap = quote_volume / base_volume`
- 大小单 CVD：
  - `whale_cvd_quote = whale_buy_quote - whale_sell_quote`
  - `retail_cvd_quote = retail_buy_quote - retail_sell_quote`
- 成交强度：
  - `avg_trade_quote = quote_volume / trade_count`
  - `avg_trade_base = base_volume / trade_count`
  - `taker_buy_avg_trade_quote = taker_buy_quote / taker_buy_trade_count`
  - `taker_sell_avg_trade_quote = taker_sell_quote / taker_sell_trade_count`
- 主动性集中度（HHI proxy）：
  - `taker_buy_hhi_quote = sum(amount^2) / sum(amount)^2`
  - `taker_sell_hhi_quote = sum(amount^2) / sum(amount)^2`
- 失衡度（简化 VPIN proxy）：
  - `imbalance_ratio_quote = abs(cvd_quote) / (taker_buy_quote + taker_sell_quote)`
- 时间加权攻击性（越靠近小时末权重越大）：
  - `tw_cvd_quote_norm`：按分钟权重 `1..60` 加权的分钟 CVD，并归一化到同一权重尺度
- 冲击/吸收类：
  - `price_change = price_close - price_open`
  - `cvd_per_abs_price_change = cvd_quote / abs(price_change)`（price_change=0 时为 NaN）

### 5.3 小时因子落盘

输出文件：

```
AlphaQCM/AlphaQCM_data/binance_aggTrades/{SYMBOL}_aggTrades.csv
AlphaQCM/AlphaQCM_data/binance_aggTrades/{SYMBOL}_aggTrades.csv.meta.json
```

注意：
- CSV 第一列名为 `minute`，但语义是“小时 bar 的起始时间戳”（例如 `2020-01-01 00:00:00+00:00` 表示 `[00:00,00:59]` 这一小时）。

---

## 6. 如何被最终宽表使用（join 口径）

在最终宽表构建 `build_final_dataset.py` 中：
- 会读取 `AlphaQCM/AlphaQCM_data/binance_aggTrades/{BINANCE_SYMBOL}_aggTrades.csv`
- 并对其所有列加前缀 `at_`
- 再按小时索引 join 到最终宽表

因此最终宽表中相关列为：
- `at_cvd_quote`, `at_whale_cvd_quote`, `at_retail_cvd_quote`, ...

---

## 7. 可逆性与信息损失（重要）

当前 **1m Parquet 中间层不能无损复原 tick 级逐笔数据**，原因：
- 中间层只保留了“分钟聚合值 + 分钟 first/last/high/low”，没有保存每笔成交明细与顺序。

如果你未来确实需要 tick 可逆：
- 必须额外保留原始 ZIP/CSV（或将逐笔写入自建存储，例如 Parquet tick 分区 / ClickHouse）。
- 本工程默认 `aggtrades_keep_zip=False`（处理完即删 ZIP）是为了控制磁盘占用。

---

## 8. 常见缺失/异常来源与处理策略

1) **新币种从 2020 开始大量 404**
- 已实现“自动跳过连续 404 月份”的优化（只在明确 404 时跳过，避免误判）。

2) **中途断点/系统重启**
- `run_aggtrades_queue.py` 的 state + `--skip-existing` + `.meta.json` 可确保不重复跑已完成的 symbol。

3) **`.zip.part` 残留**
- `_download_stream_to_file` 会尝试把“其实已经是完整 ZIP”的 `.part` 原子搬运成 `.zip` 继续处理；
  否则会删掉损坏 part 重新下载。

---

## 9. 字段快速对照（1h 输出 CSV）

`AlphaQCM/AlphaQCM_data/binance_aggTrades/{SYMBOL}_aggTrades.csv` 当前列集合示例（以 BTC 为例）：
- `minute`
- `trade_count`, `base_volume`, `quote_volume`
- `taker_buy_qty`, `taker_sell_qty`, `taker_buy_quote`, `taker_sell_quote`
- `whale_buy_quote`, `whale_sell_quote`, `retail_buy_quote`, `retail_sell_quote`
- `price_open`, `price_close`, `price_high`, `price_low`
- `cvd_qty`, `cvd_quote`, `vwap`, `whale_cvd_quote`, `retail_cvd_quote`
- `avg_trade_quote`, `avg_trade_base`
- `taker_buy_avg_trade_quote`, `taker_sell_avg_trade_quote`
- `taker_buy_hhi_quote`, `taker_sell_hhi_quote`
- `imbalance_ratio_quote`, `tw_cvd_quote_norm`
- `price_change`, `cvd_per_abs_price_change`

