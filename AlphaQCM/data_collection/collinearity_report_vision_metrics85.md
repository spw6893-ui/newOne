# 共线性裁剪报告

- 输入：`AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_all.parquet`
- 输出：`AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet`
- 阈值：`|corr| > 0.95`
- 最小覆盖率：`0.50`
- corr 采样行数：`200000`（0=不采样）

## 统计
- 保留特征：85
- 删除（覆盖率过低）：1
- 删除（常数/全空）：1
- 删除（高相关）：42

## 清单
- 保留列：`AlphaQCM/data_collection/collinearity_kept_cols_vision_metrics85.txt`
- 删除列：`AlphaQCM/data_collection/collinearity_dropped_cols_vision_metrics85.txt`

## 保留特征中的高相关对（抽样/近阈值展示）

| col_a | col_b | |corr| |
| --- | --- | ---: |
| `at_retail_buy_quote` | `at_taker_sell_trade_count` | 0.9494 |
| `at_cvd_quote` | `at_whale_cvd_quote` | 0.9451 |

