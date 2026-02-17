"""
Master pipeline for cryptocurrency data collection and processing
"""
import os
import sys
import subprocess

def run_pipeline(
    download_1min=True,
    download_trades=False,  # Trade data only available for last 7 days
    aggregate_micro=True,
    aggregate_momentum=True,
    aggregate_volatility=True,
    build_final=True
):
    """
    Run full data pipeline:
    1. Download 1-minute OHLCV data
    2. Download trade data for taker buy ratio (optional, limited history)
    3. Aggregate to hourly with microstructure features
    4. Build final dataset with cross-sectional alignment
    """

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    alphaqcm_root = os.path.dirname(scripts_dir)
    python_cmd = 'python3' if os.system('which python3 > /dev/null 2>&1') == 0 else 'python'

    # Step 1: Download 1-minute data
    if download_1min:
        print("\n" + "="*60)
        print("STEP 1: Downloading 1-minute OHLCV data")
        print("="*60)
        script = os.path.join('data_collection', 'fetch_1min_data.py')
        result = subprocess.run([python_cmd, script], cwd=alphaqcm_root)
        if result.returncode != 0:
            print("ERROR: 1-minute data download failed")
            return False

    # Step 2: Download trade data (optional)
    if download_trades:
        print("\n" + "="*60)
        print("STEP 2: Downloading trade data for taker buy ratio")
        print("="*60)
        script = os.path.join('data_collection', 'fetch_trades_data.py')
        result = subprocess.run([sys.executable, script], cwd=alphaqcm_root)
        if result.returncode != 0:
            print("WARNING: Trade data download failed (continuing anyway)")

    # Step 3: Aggregate to hourly with microstructure features
    if aggregate_micro:
        print("\n" + "="*60)
        print("STEP 3: Aggregating to hourly with microstructure features")
        print("="*60)
        script = os.path.join('data_collection', 'aggregate_microstructure.py')
        result = subprocess.run([sys.executable, script], cwd=alphaqcm_root)
        if result.returncode != 0:
            print("ERROR: Aggregation failed")
            return False

    # Step 3b: Aggregate advanced momentum factors (from 1m OHLCV)
    if aggregate_momentum:
        print("\n" + "="*60)
        print("STEP 3b: Aggregating advanced momentum factors (seg/vol/QRS)")
        print("="*60)
        script = os.path.join('data_collection', 'aggregate_momentum_factors.py')
        result = subprocess.run([sys.executable, script], cwd=alphaqcm_root)
        if result.returncode != 0:
            print("ERROR: Momentum aggregation failed")
            return False

    # Step 3c: Aggregate volatility factors (from 1m OHLCV)
    if aggregate_volatility:
        print("\n" + "="*60)
        print("STEP 3c: Aggregating volatility factors (vol/range/up-down)")
        print("="*60)
        script = os.path.join('data_collection', 'aggregate_volatility_factors.py')
        result = subprocess.run([sys.executable, script], cwd=alphaqcm_root)
        if result.returncode != 0:
            print("ERROR: Volatility aggregation failed")
            return False

    # Step 4: Build final dataset
    if build_final:
        print("\n" + "="*60)
        print("STEP 4: Building final dataset with cross-sectional alignment")
        print("="*60)
        script = os.path.join('data_collection', 'build_final_dataset.py')
        result = subprocess.run([sys.executable, script], cwd=alphaqcm_root)
        if result.returncode != 0:
            print("ERROR: Final dataset build failed")
            return False

    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    return True

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run cryptocurrency data pipeline')
    parser.add_argument('--skip-1min', action='store_true', help='Skip 1-minute data download')
    parser.add_argument('--download-trades', action='store_true', help='Download trade data (last 7 days only)')
    parser.add_argument('--skip-aggregate', action='store_true', help='Skip aggregation step')
    parser.add_argument('--skip-momentum', action='store_true', help='Skip momentum factor aggregation step')
    parser.add_argument('--skip-volatility', action='store_true', help='Skip volatility factor aggregation step')
    parser.add_argument('--skip-final', action='store_true', help='Skip final dataset build')

    args = parser.parse_args()

    success = run_pipeline(
        download_1min=not args.skip_1min,
        download_trades=args.download_trades,
        aggregate_micro=not args.skip_aggregate,
        aggregate_momentum=not args.skip_momentum,
        aggregate_volatility=not args.skip_volatility,
        build_final=not args.skip_final
    )

    sys.exit(0 if success else 1)
