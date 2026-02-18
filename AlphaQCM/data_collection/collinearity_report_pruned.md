# 共线性二次裁剪报告（基于 final_dataset_filtered.parquet）

- 输入：`AlphaQCM/AlphaQCM_data/final_dataset_filtered.parquet`
- 输出：`AlphaQCM/AlphaQCM_data/final_dataset_filtered_pruned.parquet`
- 阈值：`|corr| > 0.95`（抽样 corr）
- min coverage：`0.30`
- 抽样行数：`120000`

## 结果
- 保留特征：73
- 删除（覆盖率过低）：0
- 删除（常数/全空）：0
- 删除（高相关）：0
- pruned 后 max(|corr|) ≈ 0.9383

## 清单
- 保留列：`AlphaQCM/data_collection/collinearity_kept_cols_pruned.txt`
- 删除列：`AlphaQCM/data_collection/collinearity_dropped_cols_pruned.txt`

