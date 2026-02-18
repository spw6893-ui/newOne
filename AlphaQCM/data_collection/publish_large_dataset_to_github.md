# 通过 GitHub 共享大体积数据集（Parquet > 100MB）

GitHub 普通 `git push` **单文件上限是 100MB**，因此像
`AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet`（约 1.8GB）无法直接提交。

本仓库约定：`AlphaQCM/AlphaQCM_data/` 下的原始大文件默认不进 Git，只保留可复原的分片（`*.part_*`）。

## 方案 A（推荐）：分片进 Git（与本仓库 `.gitignore` 规则一致）

### 1) 切分（本地执行一次即可）
```bash
split -b 90M \
  AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet \
  AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet.part_
```

会得到：
`AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet.part_aa` …（若干个）

### 2) 校验文件（可选，但强烈建议）
仓库里已生成：
- `AlphaQCM/data_collection/final_dataset_vision_metrics85_filtered.parquet.sha256`
- `AlphaQCM/data_collection/final_dataset_vision_metrics85_filtered.parquet.parts.sha256`

校验分片：
```bash
sha256sum -c AlphaQCM/data_collection/final_dataset_vision_metrics85_filtered.parquet.parts.sha256
```

### 3) 提交与推送
```bash
git add .gitignore \
  AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet.part_* \
  AlphaQCM/data_collection/final_dataset_vision_metrics85_filtered.parquet.sha256 \
  AlphaQCM/data_collection/final_dataset_vision_metrics85_filtered.parquet.parts.sha256

git commit -m "Add final_dataset parquet parts (for GitHub)"
git push
```

### 4) 在另一台机器复原
```bash
cat AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet.part_* \
  > AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet

sha256sum -c AlphaQCM/data_collection/final_dataset_vision_metrics85_filtered.parquet.sha256
```

## 方案 B：GitHub Release 附件（不污染 Git 历史）
如果你只是“发一个可下载文件”，可以在 GitHub 的 Releases 页面创建一个 release，
把 `*.parquet` 作为 asset 上传（GitHub release 单个附件通常允许到 2GB 左右；以实际提示为准）。

优点：不增加仓库克隆体积；缺点：管理/权限与下载链接另行维护。

### 命令行上传（不依赖 gh CLI）
仓库提供了一个轻量脚本（用 GitHub API）：
`AlphaQCM/data_collection/github_release_upload.py`

你只需要在本机安全终端里设置 token（不要粘贴到聊天或写进文件）：
```bash
export GH_TOKEN="***"
```

然后上传附件（示例）：
```bash
python3 AlphaQCM/data_collection/github_release_upload.py \
  --tag data-2025-02-15 \
  --name "Dataset data-2025-02-15" \
  --files AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet \
          AlphaQCM/data_collection/final_dataset_vision_metrics85_filtered.parquet.sha256
```
