# Data Pipeline Status

## Current Status: IN PROGRESS

### Active Downloads

**1-minute OHLCV Data Download**
- Status: RUNNING (Background process)
- Process ID: 2576081
- Log file: `/tmp/1min_download.log`
- Output file: `/tmp/claude-1000/-home-ppw-CryptoQuant/tasks/b6a587c.output`
- Current progress: BTC/USDT:USDT at October 2020
- Estimated completion: 10-20+ hours (downloading 5+ years × 99 symbols)

### Completed Steps

1. ✅ Created 1-minute data download script (`fetch_1min_data.py`)
2. ✅ Created microstructure aggregation script (`aggregate_microstructure.py`)
3. ✅ Created trade data download script (`fetch_trades_data.py`)
4. ✅ Created final dataset builder (`build_final_dataset.py`)
5. ✅ Created master pipeline orchestrator (`run_full_pipeline.py`)
6. ✅ Created comprehensive documentation (`README_PIPELINE.md`)
7. ✅ Started 1-minute data download

### Pending Steps

1. ⏳ Complete 1-minute data download (IN PROGRESS)
2. ⏸️ Run microstructure aggregation
3. ⏸️ Download trade data (optional, last 7 days only)
4. ⏸️ Build final dataset with cross-sectional alignment

## Monitoring Progress

Check download progress:
```bash
tail -f /tmp/claude-1000/-home-ppw-CryptoQuant/tasks/b6a587c.output
```

Check completed files:
```bash
ls -lh AlphaQCM_data/crypto_1min/ | wc -l
```

Check process status:
```bash
ps aux | grep fetch_1min_data.py | grep -v grep
```

## Next Steps After Download Completes

Once the 1-minute data download finishes, run:

```bash
cd /home/ppw/CryptoQuant/AlphaQCM/data_collection

# Aggregate to hourly with microstructure features
python3 aggregate_microstructure.py

# Build final dataset
python3 build_final_dataset.py
```

Or run the full remaining pipeline:
```bash
python3 run_full_pipeline.py --skip-1min
```

## Data Pipeline Architecture

### Input Data Sources
1. **1-minute OHLCV**: Binance perpetual futures (2020-2025)
2. **Funding Rate**: 8-hour intervals (already downloaded)
3. **Trade Data**: Individual trades for taker buy ratio (optional)

### Microstructure Features
- VWAP deviation: `(close - vwap) / vwap`
- Realized volatility: `std(60 1-min returns) * sqrt(60)`
- Volume CV: `std(60 1-min volumes) / mean(60 1-min volumes)`
- Taker buy ratio: `taker_buy_volume / total_volume`

### Cross-Sectional Factors
- Funding pressure: `funding_rate - cross_sectional_median`
- Relative volume: `volume / cross_sectional_mean`
- Turnover ratio: `volume / rolling_mean(24h)`
- Price efficiency: `realized_vol / (ATR / close)`
- Momentum: 1h, 24h, 168h returns
- Volatility regime: `realized_vol / rolling_mean(168h)`

### Output Structure
```
AlphaQCM_data/
├── crypto_1min/              # Raw 1-minute OHLCV
├── crypto_trades/            # Trade-level data (optional)
├── crypto_hourly_micro/      # Hourly with microstructure features
└── final_dataset/            # Production-ready aligned dataset
```

## Estimated Data Sizes

- 1-minute data: ~10-20GB (99 symbols × 5 years)
- Hourly aggregated: ~500MB
- Final dataset: ~300MB (after NaN removal)

## Important Notes

1. **Point-in-Time Integrity**: All factors maintain T+1ms rule, no look-ahead bias
2. **Dynamic Universe**: Uses union + 50% coverage filter for cross-sectional alignment
3. **Trade Data Limitation**: Only last 7 days available via REST API
4. **Open Interest**: Only current snapshot available, no historical data
5. **Liquidations**: Only last 30 days available

## Alternative Data Sources

For historical data not available via Binance API:
- **Trade history**: Tardis.dev, Kaiko
- **Open interest**: Coinglass API
- **Liquidations**: Coinglass aggregated data
- **Order book**: Tardis.dev tick data

## Last Updated
2026-02-15 22:30 UTC
