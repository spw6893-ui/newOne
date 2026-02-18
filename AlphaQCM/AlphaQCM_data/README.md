# 训练数据下载与复原说明（GitHub 100MB 限制）

本仓库默认不把超大数据集直接提交到 Git（GitHub 单文件 100MB 上限），而是把数据切分成 `*.part_*` 分片提交，
在使用时再 `cat` 复原。

## 1) 旧版（压缩包分片）复原：`final_dataset_filtered.parquet.gz`

```bash
cd AlphaQCM/AlphaQCM_data
cat final_dataset_filtered.parquet.gz.part_* > final_dataset_filtered.parquet.gz
gunzip final_dataset_filtered.parquet.gz
ls -lh final_dataset_filtered.parquet
```

## 2) 训练就绪大表（Parquet 分片）复原：`final_dataset_vision_metrics85_filtered.parquet`

该文件较大（约 1.8GB），因此按 90MB 切分为：
`final_dataset_vision_metrics85_filtered.parquet.part_aa` …（若干个）

```bash
cd AlphaQCM/AlphaQCM_data
cat final_dataset_vision_metrics85_filtered.parquet.part_* > final_dataset_vision_metrics85_filtered.parquet
```

可选校验（推荐）：
```bash
cd /path/to/repo/root
sha256sum -c AlphaQCM/data_collection/final_dataset_vision_metrics85_filtered.parquet.sha256
```

更多细节见：
`AlphaQCM/data_collection/publish_large_dataset_to_github.md`
