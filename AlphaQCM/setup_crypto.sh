#!/bin/bash
set -e

echo "=========================================="
echo "AlphaQCM Crypto Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements_crypto.txt

# Check CUDA
echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Create data directory
echo ""
echo "Creating data directories..."
mkdir -p AlphaQCM_data/crypto_data
mkdir -p AlphaQCM_data/crypto_logs

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download data: python3 data_collection/fetch_crypto_data.py"
echo "2. Start training: python3 train_qcm_crypto.py --symbols top10"
