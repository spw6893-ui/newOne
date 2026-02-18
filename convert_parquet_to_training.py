#!/usr/bin/env python3
"""将 final_dataset_filtered.parquet 转换为训练格式"""
import pandas as pd
from pathlib import Path

# 读取 parquet 文件
print("读取 parquet 文件...")
df = pd.read_parquet('AlphaQCM/AlphaQCM_data/final_dataset_filtered.parquet')
print(f"总数据: {len(df)} 行, {len(df.columns)} 列")

# 创建输出目录
output_dir = Path('AlphaQCM/AlphaQCM_data/alphagen_ready')
output_dir.mkdir(parents=True, exist_ok=True)

# 按币种分组并保存
symbols = df['symbol'].unique()
print(f"发现 {len(symbols)} 个币种")

for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol].copy()
    symbol_df = symbol_df.set_index('datetime').sort_index()

    # 保存为 CSV
    output_file = output_dir / f"{symbol}_train.csv"
    symbol_df.to_csv(output_file)
    print(f"✓ {symbol}: {len(symbol_df)} 行")

print(f"\n完成! 数据已保存到 {output_dir}")
