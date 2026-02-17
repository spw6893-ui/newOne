"""
Smart download strategy: Test first, then decide
"""
import os
from download_binance_efficient import download_symbol_range
from datetime import datetime

# Test symbols
test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

# Date range: 2024 only (1 year)
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

print(f"Testing download for {len(test_symbols)} symbols")
print(f"Date range: {start_date.date()} to {end_date.date()}")
print()

for symbol in test_symbols:
    print(f"Downloading {symbol}...")
    download_symbol_range(symbol, start_date, end_date, 'metrics',
                         'AlphaQCM_data/binance_metrics')
    print()

# Check total size
import subprocess
result = subprocess.run(['du', '-sh', 'AlphaQCM_data/binance_metrics'],
                       capture_output=True, text=True)
print(f"Total size: {result.stdout.strip()}")
print("\nEstimated size for all 92 symbols: ~{} MB".format(
    int(result.stdout.split()[0].replace('M', '')) * 92 // 3
))
