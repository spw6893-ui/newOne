"""
Test script to verify crypto data setup
Run this before training to ensure everything is configured correctly
"""
import os
import sys
import torch

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import ccxt
        print("  ✓ ccxt installed")
    except ImportError:
        print("  ✗ ccxt not found. Install with: pip install ccxt")
        return False

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__} installed")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available, will use CPU (slower)")
    except ImportError:
        print("  ✗ PyTorch not found")
        return False

    try:
        import pandas as pd
        import numpy as np
        import yaml
        print("  ✓ pandas, numpy, yaml installed")
    except ImportError as e:
        print(f"  ✗ Missing package: {e}")
        return False

    return True

def test_data_directory():
    """Check if data directory exists"""
    print("\nTesting data directory...")
    data_dir = 'AlphaQCM_data/crypto_data'

    if not os.path.exists(data_dir):
        print(f"  ✗ Data directory not found: {data_dir}")
        print("  → Run: python data_collection/fetch_crypto_data.py")
        return False

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"  ✗ No CSV files found in {data_dir}")
        print("  → Run: python data_collection/fetch_crypto_data.py")
        return False

    print(f"  ✓ Found {len(csv_files)} data files")
    print(f"  Symbols: {', '.join([f.split('_')[0] for f in csv_files[:5]])}...")
    return True

def test_crypto_data_loader():
    """Test if CryptoData class works"""
    print("\nTesting CryptoData loader...")
    try:
        from alphagen_qlib.crypto_data import CryptoData, FeatureType
        print("  ✓ CryptoData imported successfully")

        # Try loading a small dataset
        try:
            data = CryptoData(
                symbols='top10',
                start_time='2023-01-01',
                end_time='2023-01-31',
                timeframe='1h',
                data_dir='AlphaQCM_data/crypto_data',
                device=torch.device('cpu')
            )
            print(f"  ✓ Loaded data: {data.n_stocks} symbols, {data.n_days} periods")
            return True
        except FileNotFoundError as e:
            print(f"  ✗ Data loading failed: {e}")
            print("  → Run: python data_collection/fetch_crypto_data.py")
            return False
        except Exception as e:
            print(f"  ✗ Error loading data: {e}")
            return False

    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_config_files():
    """Check if config files exist"""
    print("\nTesting config files...")
    configs = ['qcm_config/qrdqn.yaml', 'qcm_config/iqn.yaml', 'qcm_config/fqf.yaml']

    all_exist = True
    for config in configs:
        if os.path.exists(config):
            print(f"  ✓ {config}")
        else:
            print(f"  ✗ {config} not found")
            all_exist = False

    return all_exist

def main():
    print("=" * 50)
    print("AlphaQCM Crypto Setup Test")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Data Directory", test_data_directory),
        ("Config Files", test_config_files),
        ("CryptoData Loader", test_crypto_data_loader),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r for _, r in results)

    if all_passed:
        print("\n✓ All tests passed! Ready to train.")
        print("\nQuick start:")
        print("  ./run_crypto_experiment.sh qrdqn top10 1h 20")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
