# ETH 字段缺失（NaN）检查

- symbol：`ETH_USDT:USDT`

## 1) Vision 最终宽表（raw）
- 行数：44,951；列数：146
- 存在 NaN 的列数：79（无“全为 NaN”的列）

Top 30 NaN 比例：

| 列 | NaN占比 |
| --- | ---: |
| `seg_us_open_60m_logret` | 0.9583 |
| `ls_toptrader_long_short_ratio` | 0.5447 |
| `ls_taker_long_short_vol_ratio` | 0.4429 |
| `oi_delta_usd` | 0.3740 |
| `oi_delta_over_volume` | 0.3740 |
| `oi_delta_usd_over_quote_volume` | 0.3740 |
| `oi_delta` | 0.3740 |
| `oi_open_interest` | 0.3740 |
| `oi_open_interest_usd` | 0.3740 |
| `rv_stability_24h` | 0.0118 |
| `at_cvd_per_abs_price_change` | 0.0053 |
| `at_price_close` | 0.0043 |
| `at_retail_cvd_quote` | 0.0043 |
| `at_whale_cvd_quote` | 0.0043 |
| `at_vwap` | 0.0043 |
| `at_cvd_quote` | 0.0043 |
| `at_cvd_qty` | 0.0043 |
| `at_price_low` | 0.0043 |
| `at_avg_trade_base` | 0.0043 |
| `at_taker_sell_trade_count` | 0.0043 |
| `at_whale_sell_quote` | 0.0043 |
| `at_whale_buy_quote` | 0.0043 |
| `at_retail_sell_quote` | 0.0043 |
| `at_price_open` | 0.0043 |
| `at_imbalance_ratio_quote` | 0.0043 |
| `at_tw_cvd_quote_norm` | 0.0043 |
| `at_taker_sell_qty` | 0.0043 |
| `at_taker_buy_quote_sq` | 0.0043 |
| `at_taker_buy_trade_count` | 0.0043 |
| `at_taker_sell_quote` | 0.0043 |

关键列首个非 NaN 时间（用于判断“从哪天开始可用”）：

| 列 | 首个非NaN datetime(UTC) | 非NaN占比 |
| --- | --- | ---: |
| `funding_rate` | 2020-01-01 00:00:00+00:00 | 1.0000 |
| `triad_mark_close` | 2020-01-01 00:00:00+00:00 | 0.9989 |
| `triad_index_close` | 2020-01-01 00:00:00+00:00 | 0.9984 |
| `triad_premium_close` | 2020-01-01 00:00:00+00:00 | 0.9962 |
| `triad_basis` | 2020-01-01 00:00:00+00:00 | 0.9984 |
| `oi_open_interest` | 2021-12-01 00:00:00+00:00 | 0.6260 |
| `oi_open_interest_usd` | 2021-12-01 00:00:00+00:00 | 0.6260 |
| `ls_toptrader_long_short_ratio` | 2021-12-01 00:00:00+00:00 | 0.4553 |
| `ls_taker_long_short_vol_ratio` | 2021-12-01 00:00:00+00:00 | 0.5571 |

## 2) metrics85 filtered + scaled（训练就绪版）
- 行数：44,635；列数：89
- 存在 NaN 的列数：49（无“全为 NaN”的列）

Top 30 NaN 比例：

| 列 | NaN占比 |
| --- | ---: |
| `ls_toptrader_long_short_ratio` | 0.5445 |
| `ls_taker_long_short_vol_ratio` | 0.4427 |
| `oi_delta` | 0.3737 |
| `oi_delta_over_volume` | 0.3737 |
| `oi_delta_usd` | 0.3737 |
| `oi_delta_usd_over_quote_volume` | 0.3737 |
| `oi_open_interest_usd` | 0.3736 |
| `oi_open_interest` | 0.3736 |
| `rv_stability_24h` | 0.0113 |
| `at_cvd_per_abs_price_change` | 0.0053 |
| `at_price_change` | 0.0043 |
| `at_quote_volume` | 0.0043 |
| `at_cvd_qty` | 0.0043 |
| `at_avg_trade_base` | 0.0043 |
| `at_retail_buy_quote` | 0.0043 |
| `at_avg_trade_quote` | 0.0043 |
| `at_imbalance_ratio_quote` | 0.0043 |
| `at_whale_cvd_quote` | 0.0043 |
| `at_taker_sell_quote_sq` | 0.0043 |
| `at_taker_sell_trade_count` | 0.0043 |
| `at_taker_sell_hhi_quote` | 0.0043 |
| `at_tw_cvd_quote_norm` | 0.0043 |
| `at_taker_buy_quote_sq` | 0.0043 |
| `at_retail_cvd_quote` | 0.0043 |
| `at_taker_buy_hhi_quote` | 0.0043 |
| `at_cvd_quote` | 0.0043 |
| `triad_premium_close` | 0.0038 |
| `returns_168h` | 0.0032 |
| `vol_regime` | 0.0032 |
| `triad_basis` | 0.0016 |

