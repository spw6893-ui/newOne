"""
Build final dataset with cross-sectional alignment and point-in-time integrity
"""
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def build_final_dataset(
    hourly_micro_dir='AlphaQCM_data/crypto_hourly_cleaned',
    derivatives_dir='AlphaQCM_data/crypto_derivatives',
    trades_dir='AlphaQCM_data/crypto_trades',
    momentum_dir='AlphaQCM_data/crypto_hourly_momentum',
    volatility_dir='AlphaQCM_data/crypto_hourly_volatility',
    aggtrades_hourly_dir='AlphaQCM_data/binance_aggTrades',
    binance_metrics_dir='AlphaQCM_data/binance_metrics',
    binance_mark_dir='AlphaQCM_data/binance_markPriceKlines_1h',
    binance_index_dir='AlphaQCM_data/binance_indexPriceKlines_1h',
    binance_premium_dir='AlphaQCM_data/binance_premiumIndexKlines_1h',
    output_dir='AlphaQCM_data/final_dataset',
    min_coverage=0.5,
    align_cross_section=False
):
    """
    Build final dataset with:
    1. Cross-sectional alignment (dynamic universe)
    2. Point-in-time integrity (T+1ms rule)
    3. High-value derived factors
    """
    alphaqcm_root = Path(__file__).resolve().parents[1]
    hourly_micro_dir = str((alphaqcm_root / hourly_micro_dir).resolve()) if not os.path.isabs(hourly_micro_dir) else hourly_micro_dir
    derivatives_dir = str((alphaqcm_root / derivatives_dir).resolve()) if not os.path.isabs(derivatives_dir) else derivatives_dir
    trades_dir = str((alphaqcm_root / trades_dir).resolve()) if not os.path.isabs(trades_dir) else trades_dir
    momentum_dir = str((alphaqcm_root / momentum_dir).resolve()) if not os.path.isabs(momentum_dir) else momentum_dir
    volatility_dir = str((alphaqcm_root / volatility_dir).resolve()) if not os.path.isabs(volatility_dir) else volatility_dir
    aggtrades_hourly_dir = str((alphaqcm_root / aggtrades_hourly_dir).resolve()) if not os.path.isabs(aggtrades_hourly_dir) else aggtrades_hourly_dir
    binance_metrics_dir = str((alphaqcm_root / binance_metrics_dir).resolve()) if not os.path.isabs(binance_metrics_dir) else binance_metrics_dir
    binance_mark_dir = str((alphaqcm_root / binance_mark_dir).resolve()) if not os.path.isabs(binance_mark_dir) else binance_mark_dir
    binance_index_dir = str((alphaqcm_root / binance_index_dir).resolve()) if not os.path.isabs(binance_index_dir) else binance_index_dir
    binance_premium_dir = str((alphaqcm_root / binance_premium_dir).resolve()) if not os.path.isabs(binance_premium_dir) else binance_premium_dir
    output_dir = str((alphaqcm_root / output_dir).resolve()) if not os.path.isabs(output_dir) else output_dir

    os.makedirs(output_dir, exist_ok=True)

    def _ccxt_to_binance_um_symbol(ccxt_symbol: str) -> str:
        # 例：BTC_USDT:USDT -> BTCUSDT
        s = str(ccxt_symbol).split(":", 1)[0]
        return s.replace("_", "")

    # Load all symbols
    csv_files = glob.glob(os.path.join(hourly_micro_dir, '*_cleaned.csv'))
    base_suffix = '_cleaned.csv'
    if not csv_files:
        csv_files = glob.glob(os.path.join(hourly_micro_dir, '*_hourly_micro.csv'))
        base_suffix = '_hourly_micro.csv'

    all_data = {}
    for csv_file in csv_files:
        symbol = os.path.basename(csv_file).replace(base_suffix, '')
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
        df.index.name = 'datetime'

        # Merge momentum factors if available
        mom_file = os.path.join(momentum_dir, f"{symbol}_momentum.csv")
        if os.path.exists(mom_file):
            df_mom = pd.read_csv(mom_file, index_col=0, parse_dates=True)
            df_mom.index = pd.to_datetime(df_mom.index, utc=True, errors='coerce')
            df = df.join(df_mom, how='left')

        # Merge volatility factors if available
        vol_file = os.path.join(volatility_dir, f"{symbol}_volatility.csv")
        if os.path.exists(vol_file):
            df_vol = pd.read_csv(vol_file, index_col=0, parse_dates=True)
            df_vol.index = pd.to_datetime(df_vol.index, utc=True, errors='coerce')
            df = df.join(df_vol, how='left')

        # Merge aggTrades-derived orderflow/CVD factors if available (Binance Vision)
        # 输出文件名使用 Binance Vision 合约符号（如 BTCUSDT），这里做一次简单映射。
        at_symbol = _ccxt_to_binance_um_symbol(symbol)
        at_file = os.path.join(aggtrades_hourly_dir, f"{at_symbol}_aggTrades.csv")
        if os.path.exists(at_file):
            df_at = pd.read_csv(at_file, index_col=0, parse_dates=True)
            df_at.index = pd.to_datetime(df_at.index, utc=True, errors='coerce')
            df_at = df_at.add_prefix('at_')
            df = df.join(df_at, how='left')

        # Merge Binance Vision metrics (OI + Long/Short ratios)
        bm_file = os.path.join(binance_metrics_dir, f"{at_symbol}_metrics.csv")
        if os.path.exists(bm_file):
            df_bm = pd.read_csv(bm_file, parse_dates=['create_time'])
            if 'create_time' in df_bm.columns:
                df_bm['create_time'] = pd.to_datetime(df_bm['create_time'], utc=True, errors='coerce')
                df_bm = df_bm.dropna(subset=['create_time']).sort_values('create_time')
                df_bm = df_bm.drop_duplicates(subset=['create_time'], keep='last').set_index('create_time')

                rename_map = {
                    'sum_open_interest': 'oi_open_interest',
                    'sum_open_interest_value': 'oi_open_interest_usd',
                    'sum_toptrader_long_short_ratio': 'ls_toptrader_long_short_ratio',
                    'sum_taker_long_short_vol_ratio': 'ls_taker_long_short_vol_ratio',
                }
                df_bm = df_bm.rename(columns=rename_map)
                keep_cols = [c for c in rename_map.values() if c in df_bm.columns]
                if keep_cols:
                    df = df.join(df_bm[keep_cols], how='left')

        # Merge Binance Vision price triad components (mark/index/premium) and compute basis
        mark_file = os.path.join(binance_mark_dir, f"{at_symbol}_markPriceKlines.csv")
        if os.path.exists(mark_file):
            df_mark = pd.read_csv(mark_file, parse_dates=['open_time'])
            df_mark['open_time'] = pd.to_datetime(df_mark['open_time'], utc=True, errors='coerce')
            df_mark = df_mark.dropna(subset=['open_time']).drop_duplicates(subset=['open_time'], keep='last').set_index('open_time')
            if 'mark_close' in df_mark.columns:
                df = df.join(df_mark[['mark_close']].rename(columns={'mark_close': 'triad_mark_close'}), how='left')

        index_file = os.path.join(binance_index_dir, f"{at_symbol}_indexPriceKlines.csv")
        if os.path.exists(index_file):
            df_index = pd.read_csv(index_file, parse_dates=['open_time'])
            df_index['open_time'] = pd.to_datetime(df_index['open_time'], utc=True, errors='coerce')
            df_index = df_index.dropna(subset=['open_time']).drop_duplicates(subset=['open_time'], keep='last').set_index('open_time')
            if 'index_close' in df_index.columns:
                df = df.join(df_index[['index_close']].rename(columns={'index_close': 'triad_index_close'}), how='left')

        prem_file = os.path.join(binance_premium_dir, f"{at_symbol}_premiumIndexKlines.csv")
        if os.path.exists(prem_file):
            df_prem = pd.read_csv(prem_file, parse_dates=['open_time'])
            df_prem['open_time'] = pd.to_datetime(df_prem['open_time'], utc=True, errors='coerce')
            df_prem = df_prem.dropna(subset=['open_time']).drop_duplicates(subset=['open_time'], keep='last').set_index('open_time')
            if 'premium_close' in df_prem.columns:
                df = df.join(df_prem[['premium_close']].rename(columns={'premium_close': 'triad_premium_close'}), how='left')

        # Basis: (Last - Index) / Index；Last 使用最终宽表的 close（来源为 CCXT 1m->1h 聚合）
        if 'triad_index_close' in df.columns and 'close' in df.columns:
            df['triad_basis'] = (df['close'] - df['triad_index_close']) / df['triad_index_close']

        # OI 衍生：变化率、归一化
        if 'oi_open_interest' in df.columns:
            df['oi_delta'] = df['oi_open_interest'].diff()
            if 'volume' in df.columns:
                df['oi_delta_over_volume'] = df['oi_delta'] / (df['volume'] + 1e-12)
        if 'oi_open_interest_usd' in df.columns:
            df['oi_delta_usd'] = df['oi_open_interest_usd'].diff()
            if 'close' in df.columns and 'volume' in df.columns:
                df['oi_delta_usd_over_quote_volume'] = df['oi_delta_usd'] / (df['close'] * df['volume'] + 1e-12)

        # Merge taker buy ratio if available
        taker_file = os.path.join(trades_dir, f"{symbol}_taker_ratio.csv")
        if os.path.exists(taker_file):
            df_taker = pd.read_csv(taker_file, index_col=0, parse_dates=True)
            df = df.join(df_taker[['taker_buy_ratio']], how='left')
            # 缺失值策略交给下游训练准备脚本统一处理，避免在最终宽表中引入不一致的硬编码填充值

        # 兼容：若存在 rv_std_sqrt60（本仓库波动率聚合），则提供 realized_vol 别名
        if 'realized_vol' not in df.columns and 'rv_std_sqrt60' in df.columns:
            df['realized_vol'] = df['rv_std_sqrt60']

        all_data[symbol] = df

    print(f"Loaded {len(all_data)} symbols")

    if align_cross_section:
        # Get union of all timestamps
        all_timestamps = pd.DatetimeIndex([])
        for df in all_data.values():
            all_timestamps = all_timestamps.union(df.index)
        all_timestamps = all_timestamps.sort_values()

        print(f"Total timestamps: {len(all_timestamps)}")

        # Calculate coverage for each timestamp
        coverage = pd.Series(0, index=all_timestamps)
        for df in all_data.values():
            coverage[df.index] += 1

        # Filter timestamps with sufficient coverage
        min_symbols = max(1, int(len(all_data) * min_coverage))
        valid_timestamps = coverage[coverage >= min_symbols].index

        print(f"Valid timestamps (>={min_coverage*100}% coverage): {len(valid_timestamps)}")

        # Align all symbols to valid timestamps
        aligned_data = {}
        for symbol, df in all_data.items():
            # Reindex to valid timestamps with forward fill（避免 bfill 引入未来信息）
            df_aligned = df.reindex(valid_timestamps).ffill().infer_objects(copy=False)
            aligned_data[symbol] = df_aligned
    else:
        # 不做强制对齐：每个币种保留自身时间轴（避免新币/下架币被覆盖率过滤成 0 行）
        aligned_data = {s: d.copy() for s, d in all_data.items()}

    # Calculate high-value cross-sectional factors
    print("Calculating cross-sectional factors...")

    # 1. Funding rate pressure (deviation from cross-sectional median)
    if any('funding_rate' in d.columns for d in aligned_data.values()):
        all_funding = pd.DataFrame({s: d['funding_rate'] for s, d in aligned_data.items() if 'funding_rate' in d.columns})
        funding_median = all_funding.median(axis=1)

    # 2. Relative volume (vs cross-sectional mean)
    all_volume = pd.DataFrame({s: d['volume'] for s, d in aligned_data.items() if 'volume' in d.columns})
    volume_mean = all_volume.mean(axis=1)
    cs_universe_size = all_volume.notna().sum(axis=1).astype("int64")
    cs_coverage_frac = (cs_universe_size / max(1, int(len(aligned_data)))).astype("float64")

    # 统一输出列顺序（缺失列自动补 NaN，便于下游做统一 schema）
    base_cols = [
        'open', 'high', 'low', 'close', 'volume', 'vwap',
        'funding_rate', 'funding_annualized', 'funding_delta', 'arb_pressure',
        'triad_mark_close', 'triad_index_close', 'triad_premium_close', 'triad_basis',
        'oi_open_interest', 'oi_open_interest_usd',
        'oi_delta', 'oi_delta_over_volume', 'oi_delta_usd', 'oi_delta_usd_over_quote_volume',
        'ls_toptrader_long_short_ratio', 'ls_taker_long_short_vol_ratio',
        'bar_end_time', 'feature_time',
        'cs_universe_size', 'cs_coverage_frac',
        'gap_seconds', 'under_maintenance', 'cooldown_no_trade', 'trade_allowed',
        'is_stable', 'is_spike', 'is_volume_spike', 'volume_clean', 'is_mature',
        'is_valid_for_training',
    ]
    momentum_cols = [
        'seg_head_logret', 'seg_tail_logret', 'seg_tail_minus_head', 'seg_tail_share', 'seg_us_open_60m_logret',
        'vol_top20_frac', 'vol_top20_logret_sum', 'amihud_signed', 'amihud_abs',
        'qrs_beta_close_per_hour', 'qrs_r2_close', 'qrs_close',
        'qrs_beta_high_per_hour', 'qrs_r2_high', 'qrs_high',
        'qrs_beta_low_per_hour', 'qrs_r2_low', 'qrs_low',
    ]
    volatility_cols = [
        'n_minutes', 'n_minutes_kept', 'min_range_ratio_max',
        'vol_1m_mean', 'vol_1m_std', 'vol_1m_cv',
        'range_ratio_1m_std', 'log_range_1m_std',
        'up_vol_l2', 'down_vol_l2', 'up_var_share', 'down_var_share',
        'rv_std_sqrt60', 'rv_l2', 'rv_per_vol',
        'shape_skew', 'shape_kurt', 'shape_skratio',
        'shape_skewVol', 'shape_kurtVol', 'shape_skratioVol',
        'liq_amihud', 'liq_last_5min_R', 'liq_funding_impact', 'liq_top_of_hour_ratio',
        'is_funding_hour', 'liq_range_vol_ratio', 'liq_spread_std', 'liq_tail_risk', 'liq_tail_var_share',
        'corr_prv', 'corr_prvr', 'corr_pv', 'corr_pvd', 'corr_pvl', 'corr_pvr',
        'doc_vol_pdf60', 'doc_vol_pdf70', 'doc_vol_pdf80', 'doc_vol_pdf90', 'doc_vol_pdf95', 'doc_vol_pdf90bi',
        'doc_std', 'doc_skew', 'doc_kurt', 'doc_vol5_ratio', 'doc_vol10_ratio', 'doc_vol50_ratio',
        'rv_stability_24h',
    ]
    # aggTrades（Binance Vision）订单流 / CVD / 成交结构（统一加前缀 at_，避免与基础 OHLCV 混淆）
    aggtrades_cols = [
        'at_trade_count', 'at_base_volume', 'at_quote_volume',
        'at_taker_buy_qty', 'at_taker_sell_qty',
        'at_taker_buy_quote', 'at_taker_sell_quote',
        'at_taker_buy_quote_sq', 'at_taker_sell_quote_sq',
        'at_taker_buy_trade_count', 'at_taker_sell_trade_count',
        'at_whale_buy_quote', 'at_whale_sell_quote',
        'at_retail_buy_quote', 'at_retail_sell_quote',
        'at_price_open', 'at_price_close', 'at_price_high', 'at_price_low',
        'at_cvd_qty', 'at_cvd_quote', 'at_vwap',
        'at_whale_cvd_quote', 'at_retail_cvd_quote',
        'at_avg_trade_quote', 'at_avg_trade_base',
        'at_taker_buy_avg_trade_quote', 'at_taker_sell_avg_trade_quote',
        'at_taker_buy_hhi_quote', 'at_taker_sell_hhi_quote',
        'at_imbalance_ratio_quote', 'at_tw_cvd_quote_norm',
        'at_price_change', 'at_cvd_per_abs_price_change',
    ]
    derived_cols = [
        'funding_pressure', 'relative_volume', 'turnover_ratio', 'price_efficiency',
        'returns_1h', 'returns_24h', 'returns_168h', 'vol_regime',
    ]
    desired_cols = base_cols + momentum_cols + volatility_cols + aggtrades_cols + derived_cols

    summary = []
    for symbol, df in aligned_data.items():
        # 明确时间语义（避免把“小时 bar 的开始时间”误当作“可用特征时间”）
        df['bar_end_time'] = (df.index + pd.Timedelta(hours=1))
        df['feature_time'] = df['bar_end_time'] + pd.Timedelta(milliseconds=1)

        # 记录横截面可用 universe（用于下游避免“动态 universe”引入的统计口径漂移）
        df['cs_universe_size'] = cs_universe_size.reindex(df.index)
        df['cs_coverage_frac'] = cs_coverage_frac.reindex(df.index)

        if 'funding_rate' in df.columns and 'funding_rate' in all_funding.columns:
            df['funding_pressure'] = df['funding_rate'] - funding_median

        if 'volume' in df.columns:
            df['relative_volume'] = df['volume'] / volume_mean

        # 3. Abnormal turnover (volume / rolling mean)
        if 'volume' in df.columns:
            df['turnover_ratio'] = df['volume'] / df['volume'].rolling(24).mean()

        # 4. Price efficiency (realized vol / ATR)
        if 'realized_vol' in df.columns:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(24).mean()
            df['price_efficiency'] = df['realized_vol'] / (atr / df['close'])
        elif 'rv_std_sqrt60' in df.columns:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(24).mean()
            df['price_efficiency'] = df['rv_std_sqrt60'] / (atr / df['close'])

        # 5. Momentum factors
        if 'close' in df.columns:
            df['returns_1h'] = df['close'].pct_change(1, fill_method=None)
            df['returns_24h'] = df['close'].pct_change(24, fill_method=None)
            df['returns_168h'] = df['close'].pct_change(168, fill_method=None)

        # 6. Volatility regime
        if 'realized_vol' in df.columns:
            vol_ma = df['realized_vol'].rolling(168).mean()
            df['vol_regime'] = df['realized_vol'] / vol_ma
        elif 'rv_std_sqrt60' in df.columns:
            vol_ma = df['rv_std_sqrt60'].rolling(168).mean()
            df['vol_regime'] = df['rv_std_sqrt60'] / vol_ma

        # 7. 质量掩码（不直接过滤，提供给训练侧选择）
        # 规则（保守）：允许交易 + 已成熟 + 非维护/冷静期 + 非明显异常
        valid = pd.Series(True, index=df.index)
        if 'trade_allowed' in df.columns:
            valid &= (df['trade_allowed'].fillna(0) == 1)
        if 'cooldown_no_trade' in df.columns:
            valid &= (df['cooldown_no_trade'].fillna(0) == 0)
        if 'under_maintenance' in df.columns:
            valid &= (df['under_maintenance'].fillna(0) == 0)
        if 'is_mature' in df.columns:
            valid &= df['is_mature'].fillna(False).astype(bool)
        if 'is_spike' in df.columns:
            valid &= (~df['is_spike'].fillna(False).astype(bool))
        if 'is_volume_spike' in df.columns:
            valid &= (~df['is_volume_spike'].fillna(False).astype(bool))
        # 数据完整性：分钟数不足的小时不适合做高阶统计（若存在该字段）
        if 'n_minutes_kept' in df.columns:
            valid &= (pd.to_numeric(df['n_minutes_kept'], errors='coerce').fillna(0) >= 50)
        df['is_valid_for_training'] = valid.astype('int8')

        # 最小约束：保证基本行情字段存在即可；其余特征允许 NaN（早期窗口不足/个别缺失）
        for req in ('open', 'high', 'low', 'close', 'volume'):
            if req not in df.columns:
                raise ValueError(f"{symbol}: 缺少必需列 {req}")
        df = df.dropna(subset=['close'])

        # 统一 schema + 列顺序
        df = df.reindex(columns=desired_cols)

        # Save individual symbol
        output_file = os.path.join(output_dir, f'{symbol}_final.csv')
        df.to_csv(output_file)
        print(f"  Saved {symbol}: {len(df)} rows")

        summary.append({
            'symbol': symbol,
            'rows': int(len(df)),
            'start_date': df.index.min(),
            'end_date': df.index.max(),
            'features': int(len(df.columns)),
        })

    print(f"\nFinal dataset saved to {output_dir}")

    df_summary = pd.DataFrame(summary)
    summary_file = os.path.join(output_dir, 'dataset_summary.csv')
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSummary saved to {summary_file}")

if __name__ == '__main__':
    build_final_dataset(
        hourly_micro_dir='AlphaQCM_data/crypto_hourly_cleaned',
        derivatives_dir='AlphaQCM_data/crypto_derivatives',
        trades_dir='AlphaQCM_data/crypto_trades',
        momentum_dir='AlphaQCM_data/crypto_hourly_momentum',
        volatility_dir='AlphaQCM_data/crypto_hourly_volatility',
        aggtrades_hourly_dir='AlphaQCM_data/binance_aggTrades',
        binance_metrics_dir='AlphaQCM_data/binance_metrics',
        binance_mark_dir='AlphaQCM_data/binance_markPriceKlines_1h',
        binance_index_dir='AlphaQCM_data/binance_indexPriceKlines_1h',
        binance_premium_dir='AlphaQCM_data/binance_premiumIndexKlines_1h',
        output_dir='AlphaQCM_data/final_dataset',
        min_coverage=0.5,
        align_cross_section=False
    )
