#!/bin/bash

# AlphaQCM Crypto Experiment Runner
# Quick script to run multiple experiments in sequence

set -e

echo "=========================================="
echo "AlphaQCM Cryptocurrency Factor Mining"
echo "=========================================="
echo ""

# Check if data exists
if [ ! -d "AlphaQCM_data/crypto_data" ]; then
    echo "Error: Crypto data not found!"
    echo "Please run: python data_collection/fetch_crypto_data.py"
    exit 1
fi

# Default parameters
MODEL=${1:-qrdqn}
SYMBOLS=${2:-top10}
TIMEFRAME=${3:-1h}
POOL=${4:-20}

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Symbols: $SYMBOLS"
echo "  Timeframe: $TIMEFRAME"
echo "  Pool Size: $POOL"
echo ""

# Run training
echo "Starting training..."
python train_qcm_crypto.py \
    --model $MODEL \
    --symbols $SYMBOLS \
    --timeframe $TIMEFRAME \
    --pool $POOL \
    --std-lam 1.0 \
    --seed 0

echo ""
echo "Training completed!"
echo "Check results in: AlphaQCM_data/crypto_logs/${SYMBOLS}_${TIMEFRAME}/"
