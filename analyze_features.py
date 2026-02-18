#!/usr/bin/env python3
"""
分析70+因子的IC分布，识别高价值特征
"""
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "alphagen"))

from train_alphagen_crypto import _detect_feature_space, _install_dynamic_feature_type

def analyze_feature_importance():
    DATA_DIR = "AlphaQCM/AlphaQCM_data/alphagen_ready"

    # 检测特征空间
    feature_space = _detect_feature_space(Path(DATA_DIR))
    _install_dynamic_feature_type(feature_space.feature_cols)

    from alphagen.data.expression import Feature
    from AlphaQCM.alphagen_qlib.crypto_data import CryptoData
    from AlphaQCM.alphagen_qlib.calculator import QLibStockDataCalculator
    import AlphaQCM.alphagen_qlib.stock_data as sd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练数据
    print(f"加载数据... (共{len(feature_space.feature_cols)}个特征)")
    train_data = CryptoData(
        symbols="top20",
        start_time="2020-01-01",
        end_time="2024-01-01",
        timeframe='1h',
        data_dir=DATA_DIR,
        max_backtrack_periods=100,
        max_future_periods=30,
        features=None,
        device=device
    )

    # 构造目标
    close_idx = feature_space.feature_cols.index("close")
    from alphagen.data.expression import Ref
    close_expr = Feature(sd.FeatureType(close_idx))
    target = Ref(close_expr, -1) / close_expr - 1

    calculator = QLibStockDataCalculator(train_data, target)

    # 计算每个特征的IC
    print("\n计算特征IC...")
    results = []
    for i, col in enumerate(feature_space.feature_cols):
        try:
            feat = Feature(sd.FeatureType(i))
            ic = calculator.calc_single_IC_ret(feat)
            ric = calculator.calc_single_rIC_ret(feat)
            results.append({
                'index': i,
                'feature': col,
                'ic': ic,
                'rank_ic': ric,
                'abs_ic': abs(ic),
                'abs_rank_ic': abs(ric)
            })
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(feature_space.feature_cols)}")
        except Exception as e:
            print(f"  跳过 {col}: {e}")
            results.append({
                'index': i,
                'feature': col,
                'ic': 0.0,
                'rank_ic': 0.0,
                'abs_ic': 0.0,
                'abs_rank_ic': 0.0
            })

    df = pd.DataFrame(results)

    # 保存结果
    output_path = Path('./alphagen_output/feature_importance.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # 打印统计
    print("\n" + "="*60)
    print("特征IC统计")
    print("="*60)
    print(f"总特征数: {len(df)}")
    print(f"IC均值: {df['ic'].mean():.4f}")
    print(f"IC标准差: {df['ic'].std():.4f}")
    print(f"|IC| > 0.01: {(df['abs_ic'] > 0.01).sum()} 个")
    print(f"|IC| > 0.02: {(df['abs_ic'] > 0.02).sum()} 个")
    print(f"|IC| > 0.03: {(df['abs_ic'] > 0.03).sum()} 个")

    print("\nTop 20 特征 (按|IC|排序):")
    print("-"*60)
    top20 = df.nlargest(20, 'abs_ic')
    for _, row in top20.iterrows():
        print(f"{row['feature']:30s} IC={row['ic']:7.4f}  RankIC={row['rank_ic']:7.4f}")

    print("\nBottom 20 特征 (按|IC|排序):")
    print("-"*60)
    bottom20 = df.nsmallest(20, 'abs_ic')
    for _, row in bottom20.iterrows():
        print(f"{row['feature']:30s} IC={row['ic']:7.4f}  RankIC={row['rank_ic']:7.4f}")

    print(f"\n结果已保存: {output_path}")

    # 推荐配置
    print("\n" + "="*60)
    print("推荐配置")
    print("="*60)
    high_ic_count = (df['abs_ic'] > 0.01).sum()
    print(f"建议设置: export ALPHAGEN_FEATURES_MAX={min(high_ic_count, 50)}")
    print("这将自动选择IC最高的特征进行训练")

if __name__ == '__main__':
    analyze_feature_importance()
