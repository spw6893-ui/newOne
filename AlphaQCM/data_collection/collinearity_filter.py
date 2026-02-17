#!/usr/bin/env python3
"""
共线性过滤和基础数据工程
"""
import pandas as pd
import numpy as np
from pathlib import Path

def filter_collinearity(df, feature_cols, threshold=0.95, min_coverage=0.3):
    """
    过滤高度共线的特征

    Args:
        df: 数据框
        feature_cols: 特征列名列表
        threshold: 相关系数阈值
        min_coverage: 最小数据覆盖率（低于此值的特征直接删除）

    Returns:
        保留的特征列表
    """
    # 1. 删除缺失率过高的特征
    coverage = df[feature_cols].notna().mean()
    valid_features = coverage[coverage >= min_coverage].index.tolist()
    print(f"\n=== Coverage Filter ===")
    print(f"Dropped {len(feature_cols) - len(valid_features)} features with <{min_coverage*100}% coverage")

    # 2. 计算相关矩阵（只用有效训练数据）
    valid_data = df[df['is_valid_for_training'] == 1][valid_features]
    corr_matrix = valid_data.corr().abs()

    # 3. 找出高度相关的特征对
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for column in upper_tri.columns:
        correlated = upper_tri[column][upper_tri[column] > threshold].index.tolist()
        if correlated:
            # 保留缺失率更低的特征
            for corr_col in correlated:
                if coverage[column] >= coverage[corr_col]:
                    to_drop.add(corr_col)
                else:
                    to_drop.add(column)

    kept_features = [f for f in valid_features if f not in to_drop]
    print(f"\n=== Collinearity Filter ===")
    print(f"Dropped {len(to_drop)} highly correlated features (r>{threshold})")
    print(f"Kept {len(kept_features)} features")

    return kept_features

def main():
    input_file = Path('AlphaQCM/AlphaQCM_data/final_dataset_metrics85_all.parquet')
    output_file = Path('AlphaQCM/AlphaQCM_data/final_dataset_filtered.parquet')

    print(f"Loading {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Shape: {df.shape}")

    # 定义特征列（排除元数据和标志列）
    exclude_cols = [
        'symbol', 'datetime', 'bar_end_time', 'feature_time',
        'is_valid_for_training', 'trade_allowed', 'under_maintenance',
        'cooldown_no_trade', 'is_stable', 'is_spike', 'is_volume_spike',
        'is_mature', 'cs_universe_size', 'cs_coverage_frac',
        'gap_seconds', 'n_minutes', 'n_minutes_kept', 'is_funding_hour'
    ]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    print(f"\nTotal features before filtering: {len(feature_cols)}")

    # 共线性过滤
    kept_features = filter_collinearity(df, feature_cols, threshold=0.95, min_coverage=0.3)

    # 保留元数据列 + 过滤后的特征
    meta_cols = ['symbol', 'datetime', 'is_valid_for_training',
                 'cs_universe_size', 'cs_coverage_frac']
    output_cols = meta_cols + kept_features

    df_filtered = df[output_cols].copy()

    # 基础数据工程：只保留有效训练数据
    df_filtered = df_filtered[df_filtered['is_valid_for_training'] == 1].copy()
    df_filtered = df_filtered.drop(columns=['is_valid_for_training'])

    print(f"\n=== Final Dataset ===")
    print(f"Shape: {df_filtered.shape}")
    print(f"Features: {len(kept_features)}")
    print(f"Rows: {len(df_filtered):,}")

    # 保存
    df_filtered.to_parquet(output_file, index=False)
    print(f"\nSaved to {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024**2:.1f} MB")

if __name__ == '__main__':
    main()
