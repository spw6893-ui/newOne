# 特征清理（B）：winsorize + 标准化

- 输入：`AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet`
- 输出：`AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered_scaled.parquet`
- winsorize 分位数：[0.005, 0.995]（逐币种）
- 标准化：逐币种 z-score（winsorize 后，ddof=0）
- 模式：`streaming`

## 统计
- 行数：2,900,184
- 币种数：85
- 特征数：85

## 缺失率变化（Top 20）

| feature | NaN_before | NaN_after |
| --- | ---: | ---: |
| `ls_toptrader_long_short_ratio` | 0.4688 | 0.4688 |
| `ls_taker_long_short_vol_ratio` | 0.3579 | 0.3579 |
| `oi_delta` | 0.2861 | 0.2861 |
| `oi_delta_over_volume` | 0.2861 | 0.2861 |
| `oi_delta_usd` | 0.2861 | 0.2861 |
| `oi_delta_usd_over_quote_volume` | 0.2861 | 0.2861 |
| `oi_open_interest_usd` | 0.2860 | 0.2860 |
| `oi_open_interest` | 0.2860 | 0.2860 |
| `at_cvd_per_abs_price_change` | 0.0248 | 0.0248 |
| `seg_tail_share` | 0.0189 | 0.0189 |
| `rv_stability_24h` | 0.0112 | 0.0112 |
| `at_taker_buy_hhi_quote` | 0.0087 | 0.0087 |
| `at_taker_sell_hhi_quote` | 0.0087 | 0.0087 |
| `at_avg_trade_base` | 0.0087 | 0.0087 |
| `at_avg_trade_quote` | 0.0087 | 0.0087 |
| `at_cvd_qty` | 0.0087 | 0.0087 |
| `at_imbalance_ratio_quote` | 0.0087 | 0.0087 |
| `at_taker_sell_quote_sq` | 0.0087 | 0.0087 |
| `at_taker_buy_quote_sq` | 0.0087 | 0.0087 |
| `at_retail_buy_quote` | 0.0087 | 0.0087 |

