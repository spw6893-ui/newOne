# Cryptocurrency Data Pipeline for Hour-Level Trading

## Architecture Overview

This pipeline implements sophisticated data engineering for hour-level cryptocurrency trading with perpetual futures. It addresses the key challenge: **at hourly granularity, simple OHLC data is insufficient for alpha generation**.

## Pipeline Stages

### Stage 1: 1-Minute Data Collection
**Script**: `fetch_1min_data.py`

Downloads 1-minute OHLCV candles for all perpetual futures symbols. This granular data enables microstructure feature calculation.

**Why 1-minute instead of hourly?**
- Captures intra-hour price dynamics
- Enables realized volatility calculation from 60 returns
- Provides volume distribution patterns
- Allows VWAP deviation measurement

**Output**: `AlphaQCM_data/crypto_1min/`

### Stage 2: Trade-Level Data (Optional)
**Script**: `fetch_trades_data.py`

Fetches individual trades to calculate taker buy ratio (buying pressure indicator).

**Limitation**: Most exchanges only provide ~7 days of trade history via REST API. For historical data, consider:
- Binance aggTrades endpoint
- Websocket collection for real-time
- Third-party data providers (Tardis, Kaiko)

**Output**: `AlphaQCM_data/crypto_trades/`

### Stage 3: Microstructure Aggregation
**Script**: `aggregate_microstructure.py`

Aggregates 1-minute data to hourly with microstructure features:

1. **VWAP Deviation**: `(close - vwap) / vwap`
   - Measures price efficiency
   - Negative = trading below fair value

2. **Realized Volatility**: `std(60 1-min returns) * sqrt(60)`
   - True intra-hour volatility
   - More accurate than high-low range

3. **Volume CV**: `std(60 1-min volumes) / mean(60 1-min volumes)`
   - Volume distribution pattern
   - High CV = concentrated trading activity

4. **Funding Rate**: Time-aligned from derivatives data
   - Forward-filled to hourly (updates every 8h)
   - Point-in-time integrity maintained

**Output**: `AlphaQCM_data/crypto_hourly_micro/`

### Stage 4: Final Dataset Construction
**Script**: `build_final_dataset.py`

Builds production-ready dataset with:

#### Cross-Sectional Alignment
- **Dynamic Universe**: Uses union of all timestamps
- **Coverage Filter**: Keeps timestamps with ≥50% symbol coverage
- **Forward Fill**: Handles newly listed coins without data loss
- **Survivorship Bias**: Naturally handled by including delisted coins

#### High-Value Derived Factors

1. **Funding Pressure**: `funding_rate - cross_sectional_median`
   - Identifies relative funding stress
   - High pressure = expensive to hold long

2. **Relative Volume**: `volume / cross_sectional_mean`
   - Normalized volume across universe
   - Identifies abnormal activity

3. **Turnover Ratio**: `volume / rolling_mean(24h)`
   - Detects volume spikes
   - High ratio = potential breakout/breakdown

4. **Price Efficiency**: `realized_vol / (ATR / close)`
   - Measures price discovery quality
   - Low efficiency = choppy, inefficient market

5. **Momentum Factors**: 1h, 24h, 168h returns
   - Multi-timeframe momentum
   - Captures trend persistence

6. **Volatility Regime**: `realized_vol / rolling_mean(168h)`
   - Identifies vol expansion/contraction
   - >1 = high vol regime

#### Point-in-Time Integrity
- All factors calculated with T+1ms rule
- No look-ahead bias
- Forward fill only (never backward)
- NaN rows dropped after calculation

**Output**: `AlphaQCM_data/final_dataset/`

## Usage

### Full Pipeline
```bash
cd /home/ppw/CryptoQuant/AlphaQCM/data_collection
python run_full_pipeline.py
```

### Partial Pipeline
```bash
# Skip 1-minute download (use existing data)
python run_full_pipeline.py --skip-1min

# Include trade data download
python run_full_pipeline.py --download-trades

# Only aggregation and final build
python run_full_pipeline.py --skip-1min --skip-aggregate
```

### Individual Scripts
```bash
# Download 1-minute data
python fetch_1min_data.py

# Download derivatives data (funding rate)
python fetch_derivatives_data.py

# Download trade data
python fetch_trades_data.py

# Aggregate to hourly
python aggregate_microstructure.py

# Build final dataset
python build_final_dataset.py
```

## Data Size Estimates

- **1-minute data** (2020-2025, 92 symbols): ~10-20GB
- **Hourly aggregated**: ~500MB
- **Final dataset**: ~300MB (after NaN removal)

## Key Design Decisions

### Why Not Pre-Aggregated Hourly?
Pre-aggregated hourly candles from exchanges lose critical information:
- Intra-hour volatility patterns
- Volume distribution
- VWAP deviation
- Microstructure signals

### Why Union + Coverage Filter?
Alternative approaches and their issues:
- **Intersection**: Loses massive data for newer coins (2018-2025 span)
- **Union + Forward Fill**: Includes low-quality data for newly listed coins
- **Union + Coverage Filter**: Best balance of data retention and quality

### Why Forward Fill Only?
- Maintains point-in-time integrity
- Prevents look-ahead bias
- Reflects real trading conditions (use last known value)

## Limitations and Future Improvements

### Current Limitations
1. **Taker Buy Ratio**: Only last 7 days available via REST API
2. **Open Interest**: Only current snapshot, no historical data
3. **Liquidations**: Only last 30 days available
4. **Tick Data**: Not available via REST API

### Recommended Improvements
1. **Historical Trade Data**: Use Tardis.dev or Kaiko for full history
2. **Open Interest**: Collect snapshots continuously or use Coinglass API
3. **Liquidations**: Use Coinglass aggregated liquidation data
4. **Basis Data**: Add spot-futures basis for arbitrage signals
5. **Order Book**: Add bid-ask spread and depth imbalance

## Integration with AlphaQCM

The final dataset can be loaded into AlphaQCM using the modified `CryptoData` class:

```python
from alphagen_qlib.crypto_data import CryptoData

data = CryptoData(
    data_dir='AlphaQCM_data/final_dataset',
    train_start='2020-01-01',
    train_end='2023-12-31',
    valid_start='2024-01-01',
    valid_end='2024-06-30',
    test_start='2024-07-01',
    test_end='2025-02-15'
)
```

## References

- Parkinson Volatility: Parkinson (1980)
- Garman-Klass Volatility: Garman & Klass (1980)
- Microstructure: Hasbrouck (2007) "Empirical Market Microstructure"
- Point-in-Time Data: Lopez de Prado (2018) "Advances in Financial Machine Learning"
