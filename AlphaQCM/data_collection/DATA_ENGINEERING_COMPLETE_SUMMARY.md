# 加密货币永续合约数据工程完整总结

## 项目概述
为小时级（1H）永续合约交易策略构建的数据工程管道，包含数据下载、聚合、清洗和特征工程。

补充：最终宽表（`final_dataset`）字段口径与解释见：`AlphaQCM/data_collection/data_view.md`。

## 数据源
- **交易所**：Binance（币安）
- **合约类型**：USDT本位永续合约（U本位）
- **币种数量**：
  - CCXT 基础数据（`crypto_*`）：92 个 Top 币种（用于策略骨架与基础训练）
  - Binance Vision 归档补齐（`binance_*`）：默认过滤后 90 个（见 `data_collection/top100_perp_symbols.txt`）
- **时间范围**：2020-01-01 至 2025-02-15（Binance Vision 补齐覆盖到每日 23:00 UTC）
  - 说明：K线类（`klines/mark/index/premium`）与 `fundingRate` 覆盖到 `2025-02-15 23:00 UTC`（按小时）。
  - `metrics` 为 daily 归档，且“可用起始日期”因合约不同而不同（普遍为 2021-12 起；更早日期大量 404 属于源站缺失，并非下载失败）。

## 当前已落盘数据清单（AlphaQCM/AlphaQCM_data）

本仓库目前存在两套互补数据资产：
- `crypto_*`：通过 CCXT/API 获取的基础 OHLCV/资金费率等（覆盖面广、字段相对基础）。
- `binance_*`：通过 Binance Vision 历史归档补齐的合约特征（更贴近你 checklist 的 OI/TopTrader/CVD 等）。
 - `final_dataset_all.parquet`：把全部 `final_dataset/*_final.csv` 合并后的单表版本（便于训练/分析）。
 - `final_dataset_metrics85/`：按 “metrics 覆盖完整”裁剪后的 85 个币种最终宽表（单币 CSV）。
 - `final_dataset_metrics85_all.parquet`：上述 85 币种的全局单表版本（Parquet）。

### A. CCXT/API 基础数据（策略骨架）

1) **1 分钟 OHLCV**
- 目录：`AlphaQCM/AlphaQCM_data/crypto_1min/`
- 文件：`{SYMBOL}_1m.csv`（例如 `BTC_USDT:USDT_1m.csv`）
- 字段：`datetime, open, high, low, close, volume`
- 规模：约 2.3GB，92 个币种

2) **小时 OHLCV（原始/基础）**
- 目录：`AlphaQCM/AlphaQCM_data/crypto_data/`
- 文件：`{SYMBOL}_1h.csv`（例如 `BTC_USDT:USDT_1h.csv`）
- 用途：训练/回测的基础价格数据（与 `crypto_hourly_*` 属于不同阶段产物）

3) **资金费率（API 拉取，8h）**
- 目录：`AlphaQCM/AlphaQCM_data/crypto_derivatives/`
- 文件：`{SYMBOL}_funding_rate.csv`
- 覆盖：49 个币种（API/品种限制导致并非所有币都有）

4) **小时聚合（含 VWAP）**
- 目录：`AlphaQCM/AlphaQCM_data/crypto_hourly_basic/`
- 文件：`{SYMBOL}_hourly.csv`
- 字段：`datetime, open, high, low, close, volume, vwap`

5) **小时全量（叠加资金费率特征）**
- 目录：`AlphaQCM/AlphaQCM_data/crypto_hourly_full/`
- 文件：`{SYMBOL}_hourly_full.csv`
- 关键字段（示例）：`funding_rate, funding_annualized, funding_delta, arb_pressure`

6) **小时清洗后数据（训练/因子使用推荐入口）**
- 目录：`AlphaQCM/AlphaQCM_data/crypto_hourly_cleaned/`
- 文件：`{SYMBOL}_cleaned.csv`
- 关键字段（示例）：`is_stable, is_spike, is_volume_spike, volume_clean, is_mature`

7) **小时高阶动量因子（分段/量能/QRS）**
- 目录：`AlphaQCM/AlphaQCM_data/crypto_hourly_momentum/`
- 文件：`{SYMBOL}_momentum.csv`
- 关键字段（示例）：
  - 分段动量：`seg_head_logret, seg_tail_logret, seg_tail_share, seg_us_open_60m_logret`
  - 量能动量：`vol_top20_logret_sum, vol_top20_frac, amihud_signed, amihud_abs`
  - QRS：`qrs_beta_*_per_hour, qrs_r2_*, qrs_*`

8) **小时波动率因子（成交量/极比/上行下行占比）**
- 目录：`AlphaQCM/AlphaQCM_data/crypto_hourly_volatility/`
- 文件：`{SYMBOL}_volatility.csv`
- 关键字段（示例）：
  - 成交量脉冲：`vol_1m_std, vol_1m_cv`
  - 插针/流动性：`range_ratio_1m_std, log_range_1m_std`
  - 非对称波动：`up_var_share, down_var_share`
  - 波动率强度/稳定性：`rv_std_sqrt60, rv_l2, rv_stability_24h, rv_per_vol`
  - 分布形态（收益率/量比）：`shape_skew, shape_kurt, shape_skratio, shape_skewVol, shape_kurtVol, shape_skratioVol`
  - 量价相关（收益率/价格 vs 成交量）：`corr_prv, corr_prvr, corr_pv, corr_pvd, corr_pvl, corr_pvr`
  - 虚拟流动性（纯 1m 近似）：`liq_amihud, liq_last_5min_R, liq_funding_impact, liq_top_of_hour_ratio, is_funding_hour, liq_range_vol_ratio, liq_spread_std, liq_tail_risk`
  - 筹码分布（收益率分组成交量）：`doc_kurt, doc_skew, doc_std, doc_vol_pdf60, doc_vol_pdf70, doc_vol_pdf80, doc_vol_pdf90, doc_vol_pdf90bi, doc_vol_pdf95, doc_vol5_ratio, doc_vol10_ratio, doc_vol50_ratio`
    - 口径说明：`doc_vol_pdfXX` 为“分钟收益率 r 的成交量加权 XX% 分位”；`doc_vol_pdf90bi` 为 `abs(r)` 的成交量加权 90% 分位（即 90% 成交量落在 `[-x,+x]` 的阈值 x）。
- 监控脚本：`python3 AlphaQCM/data_collection/monitor_volatility_progress.py --interval 15`

9) **小时衍生品聚合（旧管线产物，部分脚本使用）**
- 目录：`AlphaQCM/AlphaQCM_data/crypto_hourly_derivatives/`
- 文件：`{SYMBOL}_hourly.csv`
- 说明：该目录属于早期/旁路产物；当前核心以 `crypto_hourly_full/cleaned` 为主。

### B. Binance Vision 归档补齐（um 永续，2020-01-01~2025-02-15）

> 命名说明：以下 `binance_*` 目录内的文件名使用 Binance Vision 的合约符号（例如 `BTCUSDT`）。
> 默认有效 universe 以 `data_collection/top100_perp_symbols.txt`（过滤后 90 个）为准；部分目录中可能仍保留过滤前下载的“额外文件”，不影响默认 universe 使用。

1) **metrics（OI + 多空比）**
- 目录：`AlphaQCM/AlphaQCM_data/binance_metrics/`
- 文件：`{SYMBOL}_metrics.csv`
- 字段（示例）：`sum_open_interest, sum_open_interest_value, sum_toptrader_long_short_ratio, sum_taker_long_short_vol_ratio`
- 备注：该目录包含额外 3 个（`KEYUSDT/MDTUSDT/STPTUSDT`），默认 universe 已剔除。
  - 覆盖检查（以 `2025-02-15` 为结束日期，按“覆盖到结束日期”口径）：90 个 symbol 中 **85 个**可覆盖到结束日期。
  - 以下 5 个合约在 Binance Vision 的 `metrics` 归档中存在**硬缺口（之后日期持续 404）**，因此无法补齐到 `2025-02-15`（截至当前落盘的最后时间戳如下）：
    - `CVCUSDT`：最后 `2024-07-15 12:00:00+00:00`
    - `DGBUSDT`：最后 `2024-07-15 12:00:00+00:00`
    - `SCUSDT`：最后 `2024-07-15 12:00:00+00:00`
    - `WAVESUSDT`：最后 `2024-07-15 12:00:00+00:00`
    - `MATICUSDT`：最后 `2025-01-22 02:00:00+00:00`（之后日期 404）

2) **Funding Rate（归档，按小时对齐）**
- 目录：`AlphaQCM/AlphaQCM_data/binance_fundingRate/`
- 文件：`{SYMBOL}_fundingRate.csv`（含 `*.meta.json` 完成标记）
- 备注：该目录可能包含额外 4 个（`BTSUSDT/KEYUSDT/MDTUSDT/STPTUSDT`）。

3) **Last Price K线（1h）**
- 目录：`AlphaQCM/AlphaQCM_data/binance_klines_1h/`
- 文件：`{SYMBOL}_klines.csv`（含 `*.meta.json`）

4) **Mark Price K线（1h）**
- 目录：`AlphaQCM/AlphaQCM_data/binance_markPriceKlines_1h/`
- 文件：`{SYMBOL}_markPriceKlines.csv`（含 `*.meta.json`）

5) **Index Price K线（1h）**
- 目录：`AlphaQCM/AlphaQCM_data/binance_indexPriceKlines_1h/`
- 文件：`{SYMBOL}_indexPriceKlines.csv`（含 `*.meta.json`）

6) **Premium Index K线（1h）**
- 目录：`AlphaQCM/AlphaQCM_data/binance_premiumIndexKlines_1h/`
- 文件：`{SYMBOL}_premiumIndexKlines.csv`（含 `*.meta.json`）
- 说明：这是 premium index 的 OHLC（用于情绪/溢价 proxy），并非 REST `premiumIndex` 快照字段全集。

7) **aggTrades（订单流特征，1h 因子）**
- 目录：`AlphaQCM/AlphaQCM_data/binance_aggTrades/`
- 文件：`{SYMBOL}_aggTrades.csv` + `{SYMBOL}_aggTrades.csv.meta.json`
- 字段（示例）：`taker_buy_quote, taker_sell_quote, cvd_quote, avg_trade_quote, whale_* , retail_* , tw_cvd_quote_norm ...`
- 覆盖：默认 universe 90/90 已完成

8) **aggTrades 1m 中间层（Parquet，回测/再聚合推荐入口）**
- 目录：`AlphaQCM/AlphaQCM_data/binance_aggTrades_1m_parquet/{SYMBOL}/`
- 文件：`{YYYY-MM}.parquet`
- 覆盖：90 个 symbol 目录，约 4277 个 parquet 文件

9) **liquidations（强平）**
- 目录：`AlphaQCM/AlphaQCM_data/binance_liquidations/`
- 状态：目前为空（Binance Vision `um` liquidationSnapshot 不稳定/缺失，需另选数据源或改用实时 WS 落盘）

10) **临时目录与运行日志**
- 目录：`AlphaQCM/AlphaQCM_data/_tmp/`
  - `binance_vision/`：下载缓存（`.zip` / `.zip.part`）
  - `logs/`：下载/聚合日志（含 queue worker 日志）
  - `aggTrades_queue_state.json`：共享队列 state（断点续跑用）

### C. 中间/历史产物（可选）
- `AlphaQCM/AlphaQCM_data/binance_archive/`、`AlphaQCM/AlphaQCM_data/binance_archive_processed/`
  - 说明：早期/中间输出（体积很小），一般无需直接使用；核心使用 `binance_*` 与 `crypto_*` 主目录即可。

### D. 全局大表（所有币种合并）

- Parquet：`AlphaQCM/AlphaQCM_data/final_dataset_all.parquet`
  - 说明：将 `final_dataset/{SYMBOL}_final.csv` 合并为单表，额外增加 `symbol` 列（文件名里的 symbol）。
  - 生成脚本：`python3 AlphaQCM/data_collection/build_global_final_table.py --output AlphaQCM_data/final_dataset_all.parquet`

## 数据管道架构

### 阶段1：原始数据下载
**脚本**：`data_collection/fetch_1min_data.py`

**下载内容**：
- 1分钟OHLCV数据
- 数据大小：2.3GB
- 存储位置：`AlphaQCM_data/crypto_1min/`

**数据字段**：
```
datetime, open, high, low, close, volume
```

**下载结果**：
- 成功：92/92个币种
- BTC数据量：28,700小时（约3.3年）
- 其他币种：约7,067小时（约10个月）

### 阶段2：资金费率数据
**脚本**：`data_collection/fetch_derivatives_data.py`

**下载内容**：
- 资金费率（Funding Rate）
- 更新频率：8小时
- 币种数量：49个
- 存储位置：`AlphaQCM_data/crypto_derivatives/`

**数据字段**：
```
datetime, funding_rate
```

### 阶段3：小时级聚合
**脚本**：`data_collection/aggregate_hourly_basic.py`

**聚合逻辑**：
- 从1分钟数据聚合到小时
- VWAP计算：`(typical_price * volume).sum() / volume.sum()`
- 时区统一：UTC

**输出数据**：
- 大小：52MB
- 存储位置：`AlphaQCM_data/crypto_hourly_basic/`

**数据字段**：
```
datetime, open, high, low, close, volume, vwap
```

### 阶段4：添加衍生品特征
**脚本**：`data_collection/add_funding_features.py`

**新增特征**：
1. `funding_rate`：原始资金费率
2. `funding_annualized`：年化资金费率 = funding_rate × 365 × 3
3. `funding_delta`：资金费率变化（一阶差分）
4. `arb_pressure`：套利压力标志（年化费率 > 30%）

**输出数据**：
- 大小：63MB
- 存储位置：`AlphaQCM_data/crypto_hourly_full/`

### 阶段5：数据清洗
**脚本**：`data_collection/clean_one_symbol.py`

**清洗逻辑**：

1. **缺失值处理**
   - 价格（open/high/low/close/vwap）：前向填充（ffill）
   - 成交量（volume）：填充为0
   - 资金费率：前向填充（ffill）

2. **维护期检测**
   - 使用“数据断流检测”替代公告/维护表（防御性编程）
   - 规则：`current_time - last_timestamp > 3605s` 视为断流/维护
   - 动作：标记恢复后的冷静期（默认 2 小时）为不可交易
   - 产出字段：`under_maintenance, cooldown_no_trade, trade_allowed`（并用 `is_stable = cooldown_no_trade==0` 兼容旧流程）

3. **异常值检测**
   - 价格尖峰：`|log_return| > 0.5`（50%波动）
   - 成交量尖峰：Z-score > 5
   - 成交量Winsorization：99th percentile截断

4. **新币过滤**
   - 最小历史要求：24小时
   - 标记：`is_mature`

5. **时点完整性检查**
   - 检测缺失的小时数
   - 确保时间序列连续性

**输出数据**：
- 大小：80MB
- 存储位置：`AlphaQCM_data/crypto_hourly_cleaned/`
- 成功率：92/92个币种（100%）

**新增字段**：
```
is_stable, is_spike, is_volume_spike, volume_clean, is_mature
```

## 最终数据集详情

### 数据位置
`AlphaQCM_data/crypto_hourly_cleaned/`

### 数据规模
- 文件数量：92个CSV文件
- 总大小：80MB
- 平均每个币种：约870KB

### 完整字段列表（16个字段）

**基础OHLCV**：
1. `datetime` - 时间戳（UTC）
2. `open` - 开盘价
3. `high` - 最高价
4. `low` - 最低价
5. `close` - 收盘价
6. `volume` - 成交量
7. `vwap` - 成交量加权平均价

**资金费率特征**：
8. `funding_rate` - 原始资金费率
9. `funding_annualized` - 年化资金费率
10. `funding_delta` - 资金费率变化
11. `arb_pressure` - 套利压力标志（0/1）

**数据质量标志**：
12. `is_stable` - 是否稳定（维护期标记）
13. `is_spike` - 是否价格尖峰
14. `is_volume_spike` - 是否成交量尖峰
15. `volume_clean` - Winsorized成交量
16. `is_mature` - 是否满足最小历史要求

### 数据质量统计

**BTC示例**：
- 总行数：28,699
- 稳定行数：28,699（100%）
- 价格尖峰：0
- 成交量尖峰：170
- 成熟行数：28,675
- 缺失小时：0

**全币种统计**：
- 平均价格尖峰：0-3个/币种
- 平均成交量尖峰：40-170个/币种
- 时间连续性：良好（缺失小时极少）

## 数据使用示例

```python
import pandas as pd

# 加载清洗后的数据
df = pd.read_csv('AlphaQCM_data/crypto_hourly_cleaned/BTC_USDT:USDT_cleaned.csv',
                 index_col=0, parse_dates=True)

# 过滤稳定且成熟的数据
df_clean = df[df['is_stable'] & df['is_mature'] & ~df['is_spike']]

# 使用清洗后的成交量
df_clean['returns'] = df_clean['close'].pct_change()

# 检查高资金费率时段
high_funding = df_clean[df_clean['arb_pressure'] == 1]
```

## 已实现的合约特有处理

### ✅ 已完成
1. **资金费率处理**
   - 时间窗口平滑（8小时间隔，forward fill到小时）
   - 费率年化计算
   - 费率差分（捕捉费率快速上升）
   - 套利压力标志（30%阈值）

2. **数据清洗**
   - 缺失值分类处理（价格ffill，成交量填0）
   - 维护期检测和标记
   - 异常值检测（价格尖峰、成交量尖峰）
   - 新币过滤（最小24小时历史）
   - 时点完整性检查

3. **数据工程最佳实践**
   - 逐个币种处理（避免内存爆炸）
   - UTC时区统一
   - 时点完整性维护（T+1ms规则）

## Binance Vision 归档数据补齐（最新进展）

### 时间与范围
- **更新时间**：2026-02-16
- **数据源**：Binance Vision（`https://data.binance.vision/data/futures/`）
- **市场**：`um`（USDT 本位永续）
- **覆盖区间**：2020-01-01 至 2025-02-15（小时级聚合后覆盖到每日 23:00 UTC）

### 已补齐的数据类型（非 aggTrades）
已通过归档 ZIP 下载并按小时聚合，输出到 `AlphaQCM/AlphaQCM_data/`：
- `metrics`（日度归档；包含 OI/价格等字段，按小时对齐）
- `fundingRate`（月度归档）
- `klines 1h`（月度归档）
- `markPriceKlines 1h`（月度归档）
- `indexPriceKlines 1h`（月度归档）
- `premiumIndexKlines 1h`（月度归档）

### aggTrades 全量（推荐：队列并发模式）
`aggTrades` 体量最大（每币每月一个 ZIP，且月内逐笔数据行数巨大），本工程采用“**月度 ZIP → 1m Parquet 中间层 → 1h 因子**”三层结构，避免回测时反复扫逐笔数据。

推荐用 `run_aggtrades_queue.py` 跑 2~3 个 worker 并发（谁空闲谁自动领取下一个 symbol），支持断点续跑且不会重复下载已完成产物：

1) 启动 3 个 tmux worker：
```bash
cd /home/ppw/CryptoQuant
tmux new-session -d -s cq_aggA "stdbuf -oL -eL python3 -u AlphaQCM/data_collection/run_aggtrades_queue.py --worker-id A --symbols-file AlphaQCM/data_collection/top100_perp_symbols.txt --market um --start 2020-01-01 --end 2025-02-15 --state AlphaQCM_data/_tmp/aggTrades_queue_state.json --aggtrades-1m-dir AlphaQCM_data/binance_aggTrades_1m_parquet --output-dir AlphaQCM_data/binance_aggTrades --temp-dir AlphaQCM_data/_tmp/binance_vision --skip-existing |& tee -a AlphaQCM/AlphaQCM_data/_tmp/logs/aggTrades_queue_A.log"
tmux new-session -d -s cq_aggB "stdbuf -oL -eL python3 -u AlphaQCM/data_collection/run_aggtrades_queue.py --worker-id B --symbols-file AlphaQCM/data_collection/top100_perp_symbols.txt --market um --start 2020-01-01 --end 2025-02-15 --state AlphaQCM_data/_tmp/aggTrades_queue_state.json --aggtrades-1m-dir AlphaQCM_data/binance_aggTrades_1m_parquet --output-dir AlphaQCM_data/binance_aggTrades --temp-dir AlphaQCM_data/_tmp/binance_vision --skip-existing |& tee -a AlphaQCM/AlphaQCM_data/_tmp/logs/aggTrades_queue_B.log"
tmux new-session -d -s cq_aggC "stdbuf -oL -eL python3 -u AlphaQCM/data_collection/run_aggtrades_queue.py --worker-id C --symbols-file AlphaQCM/data_collection/top100_perp_symbols.txt --market um --start 2020-01-01 --end 2025-02-15 --state AlphaQCM_data/_tmp/aggTrades_queue_state.json --aggtrades-1m-dir AlphaQCM_data/binance_aggTrades_1m_parquet --output-dir AlphaQCM_data/binance_aggTrades --temp-dir AlphaQCM_data/_tmp/binance_vision --skip-existing |& tee -a AlphaQCM/AlphaQCM_data/_tmp/logs/aggTrades_queue_C.log"
```

2) 监控进度（同时显示队列 pending/done/failed + 最近产物写入）：
```bash
python3 AlphaQCM/data_collection/monitor_aggtrades_progress.py --interval 15 --state AlphaQCM_data/_tmp/aggTrades_queue_state.json
```

输出落盘位置：
- 1m 中间层：`AlphaQCM/AlphaQCM_data/binance_aggTrades_1m_parquet/{SYMBOL}/{YYYY-MM}.parquet`
- 1h 因子：`AlphaQCM/AlphaQCM_data/binance_aggTrades/{SYMBOL}_aggTrades.csv`（含 `*.meta.json` 完成标记）

### 符号列表过滤说明（重要）
在跑全量补齐时发现部分合约在 Binance Vision 的 `futures/um` 归档中长期返回 404（不是 UI 目录索引问题，而是对象不存在），会导致：
- **非 aggTrades 五件套**无法补齐（资金费率/价格三元组缺失）
- `metrics`（OI 与多空比来源）也缺失，无法构造 `Open Interest` 与 `Top Trader Long/Short Ratio` 等核心因子

因此为了保证数据集字段完整性，已从 `AlphaQCM/data_collection/top100_perp_symbols.txt` 中剔除以下合约（共 6 个）：
- `BTSUSDT`（`um/daily/metrics` 404）
- `DATAUSDT`（`um/monthly` 多类数据 404；`um/daily/metrics` 404）
- `DOCKUSDT`（同上）
- `FUNUSDT`（同上）
- `REPUSDT`（同上）
- `WTCUSDT`（同上）

另外，在 `aggTrades` 全量跑数过程中发现以下合约在 Binance Vision `futures/um` 的月度 `aggTrades` 归档中**无可用月份数据**（脚本输出“无可用月份数据，跳过”），为避免数据集出现空洞，也已一并剔除（共 3 个）：
- `KEYUSDT`
- `MDTUSDT`
- `STPTUSDT`

过滤后当前默认 symbols 列表为 **90 个**（原 99 个）。

### ⚠️ 待实现（需要币安历史数据归档）

以下特征需要从币安官方历史数据归档下载：

1. **价格三元组（Price Triad）**
   - 标记价格（Mark Price）
   - 指数价格（Index Price）
   - Basis因子：`(Last - Index) / Index`
   - 数据源：`data/futures/um/daily/metrics/`

2. **持仓量（Open Interest）**
   - 历史OI数据
   - OI变化率：`ΔOI / Volume`
   - USD价值归一化
   - 数据源：`data/futures/um/daily/metrics/`

3. **强平流（Liquidations）**
   - 多空强平金额聚合
   - Z-Score异常检测
   - 数据源：`data/futures/um/daily/liquidationSnapshot/`

4. **主动成交差（CVD）**
   - Taker Buy - Taker Sell Volume
   - 需要aggTrades数据的is_buyer_maker标签
   - 数据源：`data/futures/um/daily/aggTrades/`

5. **多空持仓人数比**
   - 散户情绪指标
   - 需要额外数据源

## 脚本清单

### 数据下载
- `fetch_1min_data.py` - 下载1分钟OHLCV
- `fetch_derivatives_data.py` - 下载资金费率

### 数据处理
- `aggregate_hourly_basic.py` - 聚合到小时级
- `add_funding_features.py` - 添加资金费率特征
- `clean_one_symbol.py` - 单币种数据清洗

### 主流程
- `run_full_pipeline.py` - 完整数据管道

### 文档
- `README_PIPELINE.md` - 数据管道架构文档
- `PIPELINE_STATUS.md` - 管道状态文档
- `DATA_ENGINEERING_SUMMARY.md` - 数据工程总结

## 下一步计划

### 立即执行：下载币安历史数据归档
从 https://data.binance.vision/?prefix=data/futures/um/daily/ 下载：

1. **metrics/** - 持仓量、价格三元组
   - 字段：sum_open_interest, sum_open_interest_value, index_price, mark_price, last_price
   - 频率：每分钟

2. **liquidationSnapshot/** - 强平流
   - 字段：时间、价格、数量、方向
   - 频率：每笔强平

3. **aggTrades/** - 主动成交数据
   - 字段：price, quantity, is_buyer_maker
   - 频率：每笔成交

### 后续处理
1. 解析CSV文件
2. 聚合到小时级别
3. 计算衍生特征（Basis, OI Delta, CVD, 强平压力）
4. 合并到现有数据集

## 技术栈
- Python 3.10
- pandas - 数据处理
- numpy - 数值计算
- ccxt - 交易所API

## 存储格式
- 格式：CSV
- 编码：UTF-8
- 时区：UTC
- 索引：datetime

## 数据完整性保证
1. ✅ 时区统一（UTC）
2. ✅ 时间序列连续性检查
3. ✅ 缺失值处理策略明确
4. ✅ 异常值检测和标记
5. ✅ 新币冷启动过滤
6. ✅ 点时完整性（T+1ms规则）

## 性能指标
- 1分钟数据下载：约1小时（92个币种）
- 小时级聚合：约2分钟（92个币种）
- 数据清洗：约1分钟（92个币种）
- 内存占用：峰值约500MB（逐个处理）

## 总结
当前数据工程已完成基础数据的下载、聚合和清洗，数据质量良好，可以直接用于因子挖掘。下一步需要从币安历史数据归档下载高级合约特征（OI、强平、CVD等），以构建完整的小时级交易数据集。
