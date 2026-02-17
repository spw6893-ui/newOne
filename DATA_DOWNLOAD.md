# Download Training Data

The training data is too large for GitHub (311MB). Please download from one of these sources:

## Option 1: Generate Locally

```bash
cd AlphaQCM/data_collection
python build_final_dataset.py
python collinearity_filter.py
```

This will create `AlphaQCM_data/final_dataset_filtered.parquet`

## Option 2: Download Pre-processed Data

Upload `final_dataset_filtered.parquet` to your preferred cloud storage:
- Google Drive
- AWS S3
- Dropbox
- OneDrive

Then add the download link here.

## Dataset Specifications

- **File**: `final_dataset_filtered.parquet`
- **Size**: 311 MB
- **Rows**: 590,132
- **Features**: 73
- **Symbols**: 84 crypto pairs
- **Date Range**: 2020-01-01 to 2025-02-15
- **Frequency**: 1 hour bars
