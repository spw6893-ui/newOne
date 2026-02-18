# 共线性过滤审计（现有输出）

- source：`AlphaQCM/AlphaQCM_data/final_dataset_metrics85_all.parquet`
- filtered：`AlphaQCM/AlphaQCM_data/final_dataset_filtered.parquet`
- 阈值：`|corr| > 0.95`（本审计只做抽样核验）
- 抽样行数：`20,000`（按 row group 顺序取前 N 行）

## 列数量
- source 数值特征列数（剔除元数据/标志列）：129
- filtered 保留特征列数：73
- 估算删除特征列数：56

## 抽样相关性核验
- max(|corr|) ≈ 0.9940
- worst pair：`at_taker_sell_quote_sq` vs `at_whale_sell_quote` -> 0.9940

## 清单文件
- 保留列：`AlphaQCM/data_collection/collinearity_kept_cols_existing.txt`
- 删除列：`AlphaQCM/data_collection/collinearity_dropped_cols_existing.txt`

