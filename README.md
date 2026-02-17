# AlphaQCM Crypto Trading

AlphaGen-based quantitative crypto trading system with comprehensive data engineering pipeline.

## Data

Training data is too large for GitHub (311MB). Download from:
- **Google Drive**: [Add your link here]
- **AWS S3**: [Add your link here]

Or regenerate using:
```bash
cd AlphaQCM/data_collection
python build_final_dataset.py
python collinearity_filter.py
```

## Project Structure

```
AlphaQCM/
├── alphagen/              # AlphaGen core library
├── alphagen_qlib/         # Qlib integration
├── data_collection/       # Data pipeline scripts
│   ├── build_final_dataset.py
│   └── collinearity_filter.py
└── AlphaQCM_data/        # Training data (not in git)
    └── final_dataset_filtered.parquet  (311MB)
```

## Dataset Info

- **Samples**: 590,132 rows
- **Features**: 73 (after collinearity filtering)
- **Symbols**: 84 crypto pairs
- **Date Range**: 2020-01-01 to 2025-02-15
- **Frequency**: 1 hour bars

## Quick Start

1. Download training data (see above)
2. Install dependencies: `pip install -r requirements_crypto.txt`
3. Run training: `python train_qcm_crypto.py`

## Data Engineering

- Point-in-time integrity enforced
- Collinearity filtered (r > 0.95)
- Quality mask applied (`is_valid_for_training`)
- 73 features covering momentum, volatility, liquidity, order flow
