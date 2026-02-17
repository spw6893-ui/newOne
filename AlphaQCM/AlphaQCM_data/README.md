# Training Data Download Instructions

The training data is split into 4 parts due to GitHub's 100MB file size limit.

## Download and Reconstruct

```bash
# In the repository root
cd AlphaQCM/AlphaQCM_data

# Reconstruct the compressed file
cat final_dataset_filtered.parquet.gz.part_* > final_dataset_filtered.parquet.gz

# Decompress
gunzip final_dataset_filtered.parquet.gz

# Verify
ls -lh final_dataset_filtered.parquet
# Should be 311MB
```

## Dataset Info

- **Size**: 311 MB (uncompressed)
- **Rows**: 590,132
- **Features**: 73 (after collinearity filtering)
- **Symbols**: 84 crypto pairs
- **Date Range**: 2020-01-01 to 2025-02-15
- **Frequency**: 1 hour bars
