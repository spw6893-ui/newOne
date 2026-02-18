# 特征清理（B）：winsorize + 标准化

- 输入：`AlphaQCM/AlphaQCM_data/final_dataset_filtered_pruned.parquet`
- 输出：`AlphaQCM/AlphaQCM_data/final_dataset_filtered_pruned_scaled.parquet`
- winsorize 分位数：[0.005, 0.995]（逐币种）
- 标准化：逐币种 z-score（winsorize 后，ddof=0）

## 统计
- 行数：590,132
- 币种数：84
- 特征数：73

## 缺失率变化（Top 20）

| feature | NaN_before | NaN_after |
| --- | ---: | ---: |
| `funding_annualized` | 0.3885 | 0.3885 |
| `funding_delta` | 0.3885 | 0.3885 |
| `arb_pressure` | 0.3885 | 0.3885 |
| `returns_168h` | 0.0204 | 0.0204 |
| `vol_regime` | 0.0202 | 0.0202 |
| `at_cvd_per_abs_price_change` | 0.0108 | 0.0108 |
| `seg_tail_share` | 0.0074 | 0.0074 |
| `triad_premium_close` | 0.0052 | 0.0052 |
| `at_taker_buy_hhi_quote` | 0.0033 | 0.0033 |
| `at_taker_sell_hhi_quote` | 0.0033 | 0.0033 |
| `at_whale_sell_quote` | 0.0033 | 0.0386 |
| `at_tw_cvd_quote_norm` | 0.0033 | 0.0033 |
| `at_taker_sell_quote_sq` | 0.0033 | 0.0033 |
| `at_retail_sell_quote` | 0.0033 | 0.0033 |
| `at_taker_buy_quote_sq` | 0.0033 | 0.0033 |
| `at_avg_trade_quote` | 0.0033 | 0.0033 |
| `at_avg_trade_base` | 0.0033 | 0.0033 |
| `at_imbalance_ratio_quote` | 0.0033 | 0.0033 |
| `at_whale_cvd_quote` | 0.0033 | 0.0270 |
| `at_price_change` | 0.0033 | 0.0033 |

## 低信息量提示（std≈0 的币种占比过高）

- `vol_top20_frac`

