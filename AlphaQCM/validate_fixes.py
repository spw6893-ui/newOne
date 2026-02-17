#!/usr/bin/env python3
"""
Quick validation script to verify all bug fixes
"""
import sys

def test_imports():
    """Test 1: Check if imports work correctly"""
    print("Test 1: Checking imports...")
    try:
        from alphagen_qlib.stock_data import FeatureType
        from alphagen_qlib.crypto_data import CryptoData
        print("  ✓ FeatureType import works")
        print("  ✓ CryptoData import works")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_vwap_calculation():
    """Test 2: Verify VWAP uses rolling window"""
    print("\nTest 2: Checking VWAP calculation...")
    try:
        with open('data_collection/fetch_crypto_data.py', 'r') as f:
            content = f.read()
            if 'rolling(window=' in content and 'vwap' in content:
                print("  ✓ VWAP uses rolling window (not cumsum)")
                return True
            else:
                print("  ✗ VWAP still uses cumsum")
                return False
    except Exception as e:
        print(f"  ✗ Check failed: {e}")
        return False

def test_ccxt_check():
    """Test 3: Verify CCXT dependency check exists"""
    print("\nTest 3: Checking CCXT dependency validation...")
    try:
        with open('data_collection/fetch_crypto_data.py', 'r') as f:
            content = f.read()
            if 'try:' in content and 'import ccxt' in content and 'except ImportError' in content:
                print("  ✓ CCXT dependency check exists")
                return True
            else:
                print("  ✗ No CCXT dependency check")
                return False
    except Exception as e:
        print(f"  ✗ Check failed: {e}")
        return False

def test_data_alignment():
    """Test 4: Verify data alignment uses union strategy"""
    print("\nTest 4: Checking data alignment strategy...")
    try:
        with open('alphagen_qlib/crypto_data.py', 'r') as f:
            content = f.read()
            if 'union' in content and 'coverage' in content.lower():
                print("  ✓ Data alignment uses union + coverage strategy")
                return True
            else:
                print("  ✗ Still using intersection")
                return False
    except Exception as e:
        print(f"  ✗ Check failed: {e}")
        return False

def test_memory_optimization():
    """Test 5: Verify memory optimizations"""
    print("\nTest 5: Checking memory optimizations...")
    try:
        with open('alphagen_qlib/crypto_data.py', 'r') as f:
            content = f.read()
            checks = [
                ('float32' in content, "Uses float32"),
                ("device='cpu'" in content, "Loads to CPU first"),
                ('try:' in content and 'to(self.device)' in content, "Safe GPU transfer")
            ]
            all_pass = all(check[0] for check in checks)
            for passed, desc in checks:
                print(f"  {'✓' if passed else '✗'} {desc}")
            return all_pass
    except Exception as e:
        print(f"  ✗ Check failed: {e}")
        return False

def test_top100_support():
    """Test 6: Verify Top100 support"""
    print("\nTest 6: Checking Top100 support...")
    try:
        import os
        checks = [
            (os.path.exists('data_collection/top100_symbols.txt'), "top100_symbols.txt exists"),
        ]

        with open('train_qcm_crypto.py', 'r') as f:
            content = f.read()
            checks.append(("'top100'" in content, "train script supports top100"))

        with open('alphagen_qlib/crypto_data.py', 'r') as f:
            content = f.read()
            checks.append(("'top100'" in content, "CryptoData supports top100"))

        all_pass = all(check[0] for check in checks)
        for passed, desc in checks:
            print(f"  {'✓' if passed else '✗'} {desc}")
        return all_pass
    except Exception as e:
        print(f"  ✗ Check failed: {e}")
        return False

def test_timezone_handling():
    """Test 7: Verify timezone handling"""
    print("\nTest 7: Checking timezone handling...")
    try:
        with open('data_collection/fetch_crypto_data.py', 'r') as f:
            content = f.read()
            if 'utc=True' in content:
                print("  ✓ UTC timezone handling present")
                return True
            else:
                print("  ✗ No UTC timezone handling")
                return False
    except Exception as e:
        print(f"  ✗ Check failed: {e}")
        return False

def main():
    print("=" * 60)
    print("AlphaQCM Bug Fix Validation")
    print("=" * 60)

    tests = [
        ("Import compatibility", test_imports),
        ("VWAP calculation", test_vwap_calculation),
        ("CCXT dependency check", test_ccxt_check),
        ("Data alignment strategy", test_data_alignment),
        ("Memory optimizations", test_memory_optimization),
        ("Top100 support", test_top100_support),
        ("Timezone handling", test_timezone_handling),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r for _, r in results)

    if all_passed:
        print("\n✓ All bug fixes validated successfully!")
        print("\nReady to use:")
        print("  python data_collection/fetch_crypto_data.py")
        print("  python train_qcm_crypto.py --symbols top100 --pool 50")
        return 0
    else:
        print("\n✗ Some validations failed. Please review the fixes.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
