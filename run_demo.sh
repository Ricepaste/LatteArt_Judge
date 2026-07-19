#!/bin/bash
# ==============================================================================
# Local Statistics Sparse SSL - Quick Start Demo Script
# ==============================================================================
# This script demonstrates how to run the Hebbian Sparse SSL pretraining 
# and the subsequent downstream evaluation on a small scale.
# 
# It runs a quick 2-epoch pretraining on CIFAR-10 at 80% sparsity, 
# followed by KNN and Linear Probing evaluation.
# ==============================================================================

set -e

echo "============================================================"
echo "🚀 Starting Local Statistics Sparse SSL Quick Demo"
echo "============================================================"

# Switch to the main directory
cd main

# ---------------------------------------------------------
# 1. Pretraining Stage
# ---------------------------------------------------------
echo "\n[1/2] 🏋️‍♂️ Running Hebbian Sparse Pretraining (2 Epochs)..."

export TARGET_DATASET="cifar10"
export TARGET_SPARSITY=0.80
export NUM_EPOCHS=2
export RUN_SEED=42
export DATALOADER_WORKERS=4

# Run the pretraining script
python Hebbian.py

echo "✅ Pretraining completed!"

# Find the newly created checkpoint folder
# (It will look something like runs/Hebbian_SSL_s80_seed42_c10_...)
LATEST_RUN=$(ls -td runs/Hebbian_SSL_* | head -1)
ENCODER_PATH="${LATEST_RUN}/last.pt"

echo "📂 Checkpoint saved at: ${ENCODER_PATH}"

# ---------------------------------------------------------
# 2. Evaluation Stage
# ---------------------------------------------------------
echo "\n[2/2] 📊 Running Downstream Evaluation (KNN & Linear Probing)..."

export ENCODER_PATH=$ENCODER_PATH
export METHOD="hebbian"
# For the demo, we also do a very short 5-epoch linear probe
export NUM_EPOCHS=5 

# Run the evaluation script
python evaluate_model.py

echo "============================================================"
echo "🎉 Demo completed successfully! "
echo "You can check the evaluation results in the logs or console output above."
echo "============================================================"
