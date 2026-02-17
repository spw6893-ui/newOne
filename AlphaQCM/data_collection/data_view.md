# 数据视图（Data View）

本文档用于解释本项目当前落盘的**最终宽表**字段口径（以及它们来自哪些中间产物）。所有时间戳默认均为 **UTC**。

## 1. 最终宽表（推荐直接用）

- 路径：`AlphaQCM/AlphaQCM_data/final_dataset/{SYMBOL}_final.csv`
- 粒度：1 小时（1H）
- 行含义：`datetime` 对应该小时的 bar（例如 `10:00:00+00:00` 表示 `[10:00, 10:59]` 这一小时）
- 主要来源：
  - 基础行情 + 资金费：`AlphaQCM/AlphaQCM_data/crypto_hourly_cleaned/{SYMBOL}_cleaned.csv`
  - 动量因子：`AlphaQCM/AlphaQCM_data/crypto_hourly_momentum/{SYMBOL}_momentum.csv`
  - 波动率/微观结构/形态/流动性/筹码分布：`AlphaQCM/AlphaQCM_data/crypto_hourly_volatility/{SYMBOL}_volatility.csv`
  - 额外衍生列（跨品种/滚动统计）：由 `AlphaQCM/data_collection/build_final_dataset.py` 在生成最终宽表时追加

如果你需要“所有币种合并后的一张大表”（便于一次性喂给训练/分析）：
- 路径：`AlphaQCM/AlphaQCM_data/final_dataset_all.parquet`
- 粒度：1 小时（1H）
- schema：与单币种 `{SYMBOL}_final.csv` 完全一致，**仅额外增加一列 `symbol`**（文件名里的 symbol，例如 `BTC_USDT:USDT`）
- 生成脚本：`python3 AlphaQCM/data_collection/build_global_final_table.py`

> 备注：为避免“新币/下架币”被覆盖率过滤导致 0 行，最终宽表默认 **不强制做横截面对齐**；跨品种统计（如 `funding_pressure`、`relative_volume`）按每个时间点可用币种集合计算（`NaN` 自动跳过）。

## 1.1 全局大表（把所有币合并到一张表）

- 路径：`AlphaQCM/AlphaQCM_data/final_dataset_all.parquet`
- 粒度：1 小时（1H）
- 行含义：每行 = 某个 `symbol` 在某个 `datetime` 的一根小时 bar
- 字段：与单币最终宽表完全一致，只额外增加：
  - `symbol`：币种/合约标识（与 `final_dataset/*_final.csv` 文件名一致，例如 `BTC_USDT:USDT`）

## 1.2 全局大表（metrics 覆盖完整的 85 币种版本，推荐训练用）

- 单币 CSV 目录：`AlphaQCM/AlphaQCM_data/final_dataset_metrics85/`
- 全局 Parquet：`AlphaQCM/AlphaQCM_data/final_dataset_metrics85_all.parquet`
- 口径：以 `top100_perp_symbols.txt`（过滤后 90）为 universe，进一步剔除 `metrics` 尾部持续缺失的币种（剩余 85）。

---

## 2. 字段解释（`{SYMBOL}_final.csv`）

### A) 基础行情（1H OHLCV + VWAP）

| 字段 | 含义 | 备注 |
|---|---|---|
| `datetime` | 小时时间戳（UTC） | CSV 第一列 |
| `open` | 该小时开盘价 | 价格单位与交易所一致 |
| `high` | 该小时最高价 |  |
| `low` | 该小时最低价 |  |
| `close` | 该小时收盘价 |  |
| `volume` | 该小时成交量（基础币数量） | 来自 1m 汇总 |
| `vwap` | 该小时成交量加权平均价（VWAP） | `typical_price=(H+L+C)/3`，`vwap=sum(typical_price*vol)/sum(vol)` |

### B) 资金费率与套利压力（衍生品成本/价差 proxy）

| 字段 | 含义 | 公式/口径 |
|---|---|---|
| `funding_rate` | 资金费率（8 小时间隔） | 逐小时对齐后 ffill |
| `funding_annualized` | 资金费率年化 | `funding_rate * 365 * 3`（每天 3 次结算） |
| `funding_delta` | 资金费率变化 | `funding_rate.diff()` |
| `arb_pressure` | 套利压力标志（0/1） | `abs(funding_annualized) > 0.30` 视为 1（年化 > 30%） |

### B2) 价格三元组（Mark / Index / Premium）与 Basis

| 字段 | 含义 | 口径 |
|---|---|---|
| `triad_mark_close` | Mark Price（标记价格）收盘值 | 来自 Binance Vision `markPriceKlines 1h` |
| `triad_index_close` | Index Price（指数价格）收盘值 | 来自 Binance Vision `indexPriceKlines 1h` |
| `triad_premium_close` | Premium Index（溢价指数）收盘值 | 来自 Binance Vision `premiumIndexKlines 1h`（可视作“预测资金费/溢价”proxy） |
| `triad_basis` | Basis（基差） | `(Last - Index)/Index`，`Last` 使用最终宽表的 `close` |

### B3) 持仓量（Open Interest）与多空比（来自 Binance Vision metrics）

| 字段 | 含义 | 口径 |
|---|---|---|
| `oi_open_interest` | OI（合约张数口径） | `sum_open_interest` |
| `oi_open_interest_usd` | OI（USD 价值口径） | `sum_open_interest_value` |
| `oi_delta` | OI 变化（合约张数） | `oi_open_interest.diff()` |
| `oi_delta_over_volume` | OI 变化率/成交量 | `oi_delta / volume`（volume 为基础币数量） |
| `oi_delta_usd` | OI 变化（USD 价值） | `oi_open_interest_usd.diff()` |
| `oi_delta_usd_over_quote_volume` | OI USD 变化率/成交额 | `oi_delta_usd / (close*volume)`（近似 quote 成交额） |
| `ls_toptrader_long_short_ratio` | Top Trader 多空比 | `sum_toptrader_long_short_ratio`（偏“主力”） |
| `ls_taker_long_short_vol_ratio` | 主动成交多空比（量口径） | `sum_taker_long_short_vol_ratio`（偏“订单流”） |

### C) 断流/维护检测（替代“公告/维护时间表”）

| 字段 | 含义 | 口径 |
|---|---|---|
| `gap_seconds` | 与上一根 bar 的时间差（秒） | 用于检测断流 |
| `under_maintenance` | 是否为“断流后恢复的第一根 bar”（0/1） | `gap_seconds > 3605` 的下一根标记为 1 |
| `cooldown_no_trade` | 恢复后冷静期（0/1） | 默认 2 小时（含恢复当下这根） |
| `trade_allowed` | 是否允许交易（0/1） | `cooldown_no_trade==0` |
| `is_stable` | 稳定状态（True/False） | 当前实现等价于 `cooldown_no_trade==0` |

### D) 清洗/异常标记

| 字段 | 含义 | 口径 |
|---|---|---|
| `is_spike` | 价格极端跳变（True/False） | `abs(log(close/close.shift(1))) > 0.5` |
| `is_volume_spike` | 成交量异常（True/False） | 168 小时滚动 zscore，`abs(z)>5` |
| `volume_clean` | 清洗后的成交量 | `volume` 上截断到 99% 分位 |
| `is_mature` | 是否“上市已满”标记（True/False） | 默认前 24 根小时 bar 视为未成熟 |

### E) 动量因子（1m→1H）

| 字段 | 含义 | 口径 |
|---|---|---|
| `seg_head_logret` | 头部 45 分钟 log 收益 | `log(C_head) - log(O_0)`，`minute<45` |
| `seg_tail_logret` | 尾部 15 分钟 log 收益 | `log(C_last) - log(C_head)`，`minute>=45` |
| `seg_tail_minus_head` | 尾部动量减头部动量 | `seg_tail_logret - seg_head_logret` |
| `seg_tail_share` | 尾部动量占比 | `seg_tail_logret / (seg_head_logret+seg_tail_logret)` |
| `seg_us_open_60m_logret` | 美股开盘后 60min 动量 | NY 09:30→10:29 的 log 收益，按窗口结束所在 UTC 小时落盘 |
| `vol_top20_frac` | “高放量分钟”占比 | 该小时内 `volume>=80%分位` 的分钟占比 |
| `vol_top20_logret_sum` | 高放量分钟的 log 收益和 | 仅在高放量分钟上求和 |
| `amihud_signed` | Amihud（带方向） | `sum(logret / (volume*close))` |
| `amihud_abs` | Amihud（绝对值） | `sum(abs(logret) / (volume*close))` |
| `qrs_beta_close_per_hour` | QRS 斜率（close，按小时） | 对 `log(close)` 回归，`slope_per_min*60` |
| `qrs_r2_close` | QRS 拟合优度（close） | R²，范围 [0,1] |
| `qrs_close` | QRS（close） | `qrs_beta_close_per_hour * qrs_r2_close` |
| `qrs_beta_high_per_hour` | QRS 斜率（high） | 同上 |
| `qrs_r2_high` | QRS 拟合优度（high） |  |
| `qrs_high` | QRS（high） |  |
| `qrs_beta_low_per_hour` | QRS 斜率（low） | 同上 |
| `qrs_r2_low` | QRS 拟合优度（low） |  |
| `qrs_low` | QRS（low） |  |

### F) 波动率/微观结构（1m→1H）

| 字段 | 含义 | 口径 |
|---|---|---|
| `n_minutes` | 该小时原始分钟数 | 用于诊断缺失 |
| `n_minutes_kept` | 过滤后分钟数 | 默认过滤 `high/low>1.2` 的异常分钟 |
| `min_range_ratio_max` | 该小时内 `high/low` 最大值 | 诊断插针/脏数据 |
| `vol_1m_mean` | 分钟成交量均值 |  |
| `vol_1m_std` | 分钟成交量标准差 | 资金进入节奏（脉冲） |
| `vol_1m_cv` | 分钟成交量变异系数 | `std/(mean+eps)` |
| `range_ratio_1m_std` | 分钟 `high/low` 标准差 | 插针/流动性薄厚 proxy |
| `log_range_1m_std` | 分钟 `log(high/low)` 标准差 | 更稳定的口径 |
| `up_vol_l2` | 上行波动 L2 | `sqrt(sum(r^2 for r>0))` |
| `down_vol_l2` | 下行波动 L2 | `sqrt(sum(r^2 for r<0))` |
| `up_var_share` | 上行方差占比 | `up_ss/(up_ss+down_ss)` |
| `down_var_share` | 下行方差占比 |  |
| `rv_std_sqrt60` | 实现波动率（std*sqrt(60)） | `std(r)*sqrt(60)`，r 为分钟 log return |
| `rv_l2` | 实现波动率（L2） | `sqrt(sum(r^2))` |
| `rv_per_vol` | 单位成交量波动率 | `rv_std_sqrt60/(vol_1m_mean+eps)` |
| `rv_stability_24h` | 波动率稳定性（相对 24H 均值） | `rv_std_sqrt60 / MA24(rv_std_sqrt60)` |
| `realized_vol` | 兼容列：实现波动率 | 当前等价于 `rv_std_sqrt60` |

### G) 形态（偏度/峰度）与量价相关

| 字段 | 含义 | 口径 |
|---|---|---|
| `shape_skew` | 分钟收益率偏度 | 对 `r[1:]`（去掉首分钟人为 0） |
| `shape_kurt` | 分钟收益率超额峰度 | pandas 口径：正态=0 |
| `shape_skratio` | 偏度/峰度比值 | `skew/(abs(kurt)+eps)` |
| `shape_skewVol` | 分钟量比偏度 | `vol_i/sum(vol)` 的偏度 |
| `shape_kurtVol` | 分钟量比超额峰度 |  |
| `shape_skratioVol` | 分钟量比偏度/峰度比 | 同上 |
| `corr_prv` | 分钟收益率 vs 成交量相关 | Pearson，样本<10 或方差=0→NaN |
| `corr_prvr` | 分钟收益率 vs 成交量变化率相关 | `vol_ret=vol[t]/vol[t-1]-1`（clip 到 [-10,10]） |
| `corr_pv` | 分钟收盘价 vs 成交量相关 | Pearson |
| `corr_pvd` | 分钟收盘价 vs 滞后成交量相关 | `corr(close[t], vol[t-1])` |
| `corr_pvl` | 分钟收盘价 vs 领先成交量相关 | `corr(close[t-1], vol[t])` |
| `corr_pvr` | 分钟收盘价 vs 成交量变化率相关 |  |

### H) 流动性/非流动性（1m 近似，Crypto 适配）

| 字段 | 含义 | 口径 |
|---|---|---|
| `liq_amihud` | 小时非流动性 | `mean(abs(r[1:]) / vol[1:])` |
| `liq_last_5min_R` | 收线成交量占比 | 最后 5 分钟量 / 小时总量 |
| `liq_funding_impact` | 整点前后量占比 | (前 5 + 后 5) 分钟量 / 总量 |
| `liq_top_of_hour_ratio` | 整点效应（算法参与度 proxy） | 前 10 分钟量 / 总量 |
| `is_funding_hour` | 资金费时刻标记（0/1） | UTC 小时为 00/08/16 |
| `liq_range_vol_ratio` | 近似价差/深度 proxy | `mean((high-low)/volume)` |
| `liq_spread_std` | 相对价差波动 | `std((high-low)/close)` |
| `liq_tail_risk` | 尾部风险（L2 比例） | 最后 10min 波动 L2 / 全小时波动 L2 |
| `liq_tail_var_share` | 尾部方差占比 | 最后 10min `sum(r^2)` / 全小时 `sum(r^2)` |

### I) 筹码分布（成交量在“收益率维度”的分布，doc_*）

核心思想：把分钟收益率当作 x 轴、分钟成交量当作权重，描述该小时成交量集中在哪些“涨跌幅水平”。

| 字段 | 含义 | 口径 |
|---|---|---|
| `doc_vol_pdf60` | 成交量加权收益率 60% 分位 | 对分钟 log return `r[1:]` 做加权分位 |
| `doc_vol_pdf70` | 成交量加权收益率 70% 分位 |  |
| `doc_vol_pdf80` | 成交量加权收益率 80% 分位 |  |
| `doc_vol_pdf90` | 成交量加权收益率 90% 分位 |  |
| `doc_vol_pdf95` | 成交量加权收益率 95% 分位 |  |
| `doc_vol_pdf90bi` | 双边 90% 阈值 | `abs(r)` 的成交量加权 90% 分位（90% 成交量落在 `[-x,+x]` 的 x） |
| `doc_std` | 分箱成交量标准差 | 将 `r` 按小时内 min/max 动态分 25 箱，统计每箱成交量后求 std |
| `doc_skew` | 分箱成交量偏度 | 对“每箱成交量向量”求偏度 |
| `doc_kurt` | 分箱成交量超额峰度 | 对“每箱成交量向量”求超额峰度（正态=0） |
| `doc_vol5_ratio` | 前 5 大 bin 成交量占比 | `sum(top5_bin_vol)/sum(all_bin_vol)` |
| `doc_vol10_ratio` | 前 10 大 bin 成交量占比 |  |
| `doc_vol50_ratio` | 前 50 大 bin 成交量占比 | 若 bin 数不足则用全部 |

### J) 最终宽表追加衍生列（生成时计算）

| 字段 | 含义 | 口径 |
|---|---|---|
| `funding_pressure` | 资金费横截面压力 | `funding_rate - median(funding_rate across symbols at same datetime)` |
| `relative_volume` | 相对成交量 | `volume / mean(volume across symbols at same datetime)` |
| `turnover_ratio` | 异常换手比 | `volume / rolling_mean(volume, 24h)` |
| `price_efficiency` | 价格效率 | `realized_vol / (ATR_24 / close)` |
| `returns_1h` | 1 小时收益率 | `close.pct_change(1)` |
| `returns_24h` | 24 小时收益率 | `close.pct_change(24)` |
| `returns_168h` | 168 小时收益率 | `close.pct_change(168)` |
| `vol_regime` | 波动率状态 | `realized_vol / rolling_mean(realized_vol, 168h)` |
| `cs_universe_size` | 横截面可用币种数 | 该时间点 `volume` 非空的币种数（动态 universe） |
| `cs_coverage_frac` | 横截面覆盖率 | `cs_universe_size / 总币种数` |
| `bar_end_time` | bar 结束时间（UTC） | `datetime + 1h` |
| `feature_time` | 特征可用时间（UTC） | `bar_end_time + 1ms`（用于防止把 bar 内信息当作 bar 开始时已知） |
| `is_valid_for_training` | 训练可用掩码（0/1） | 组合质量标记：允许交易 + 已成熟 + 非维护/冷静期 + 非明显异常（并要求 `n_minutes_kept>=50` 若存在） |

---

## 3. aggTrades 订单流/CVD（来自 Binance Vision，`at_*` 前缀）

这部分字段来自 `AlphaQCM/AlphaQCM_data/binance_aggTrades/{BINANCE_SYMBOL}_aggTrades.csv`（已是 1H 聚合结果），在最终宽表中统一加 `at_` 前缀，以免与基础 OHLCV 混淆。

| 字段 | 含义 | 口径 |
|---|---|---|
| `at_trade_count` | 该小时成交笔数 | aggTrades 聚合计数 |
| `at_base_volume` | 该小时基础币成交量 | qty 汇总 |
| `at_quote_volume` | 该小时计价币成交量（近似 USD） | `sum(qty*price)` |
| `at_taker_buy_qty` / `at_taker_sell_qty` | 主动买/卖数量 | 基于 `isBuyerMaker` 分组 |
| `at_taker_buy_quote` / `at_taker_sell_quote` | 主动买/卖成交额（quote） | `sum(qty*price)` |
| `at_taker_buy_trade_count` / `at_taker_sell_trade_count` | 主动买/卖笔数 |  |
| `at_cvd_qty` | CVD（数量口径） | `taker_buy_qty - taker_sell_qty` |
| `at_cvd_quote` | CVD（成交额口径） | `taker_buy_quote - taker_sell_quote` |
| `at_whale_buy_quote` / `at_whale_sell_quote` | 大单主动买/卖成交额 | 默认阈值见 aggTrades 聚合脚本 |
| `at_retail_buy_quote` / `at_retail_sell_quote` | 小单主动买/卖成交额 |  |
| `at_whale_cvd_quote` | 大单 CVD（quote） | `whale_buy_quote - whale_sell_quote` |
| `at_retail_cvd_quote` | 小单 CVD（quote） | `retail_buy_quote - retail_sell_quote` |
| `at_avg_trade_quote` / `at_avg_trade_base` | 平均单笔成交额/数量 | `quote_volume/trade_count` 等 |
| `at_taker_buy_avg_trade_quote` / `at_taker_sell_avg_trade_quote` | 主动买/卖平均单笔成交额 |  |
| `at_taker_buy_hhi_quote` / `at_taker_sell_hhi_quote` | 主动买/卖集中度（HHI） | 反映是否由少数大单主导 |
| `at_imbalance_ratio_quote` | 主动性不平衡强度 | `abs(cvd_quote)/ (taker_buy_quote+taker_sell_quote)` |
| `at_tw_cvd_quote_norm` | 时间加权主动性（归一化） | 越接近小时末权重越大 |
| `at_price_open/close/high/low` | aggTrades 价格序列的 OHLC | 从逐笔成交重建 |
| `at_vwap` | aggTrades 口径 VWAP | `quote_volume/base_volume` |
| `at_price_change` | 价格变动（close-open） | aggTrades 口径 |
| `at_cvd_per_abs_price_change` | 单位价格变动的 CVD | `cvd_quote/abs(price_change)` |

> 备注：`at_*` 与基础 1m→1H 的 `open/high/low/close/vwap` 可能存在细微差异（不同数据源/聚合口径），一般属于可接受范围。

---

## 4. AlphaGen 训练准备（建议）

本仓库提供了一个“训练集准备脚本”，用于：
- 构造未来收益标签（默认 `y_logret_fwd_1h`）
- 可选过滤低质量区间（维护/冷静期/新币/异常）
- 可选缺失值策略（none/ffill/zero）

脚本：`AlphaQCM/data_collection/prepare_alphagen_training_data.py`

示例：
```bash
python3 AlphaQCM/data_collection/prepare_alphagen_training_data.py --filter-quality --impute ffill --ffill-limit 24
```

输出：`AlphaQCM/AlphaQCM_data/alphagen_ready/{SYMBOL}_train.csv`

---

## 5. 常见数值现象与建议

1) **出现 `-0.0`**  
浮点数显示导致（极小负数四舍五入），与 `0.0` 数学等价。

2) **部分列为 NaN**  
通常是滚动窗口不足（如 24H/168H）或小时内分钟数不足导致统计不稳定；最终宽表保留 NaN，建议在建模前统一做缺失值处理策略（删行/填充/分桶）。
