#!/bin/bash
set -euo pipefail
set -x

# ============================================================
# 1. 工作目錄 & 環境設定 (對齊您在 TWCC 的路徑)
# ============================================================
WORKDIR="/home/sharpaste236/LatteArt_Judge"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ============================================================
# 2. 執行隨機生長（SET）對照組實驗
# ============================================================
TARGET_SPARSITY_VAL="${TARGET_SPARSITY:-0.8}"
RUN_SEED_VAL="${RUN_SEED:-42}"
SPARSITY_PERCENT=$(python3 -c "print(int(float('${TARGET_SPARSITY_VAL}') * 100))")

echo "============================================================"
echo "🚀 Starting Standardized SET (Random Growth) Experiment @ ${SPARSITY_PERCENT}% Sparsity"
echo "🚀 Target Dataset: CIFAR-100 | Target Sparsity: ${TARGET_SPARSITY_VAL} | Seed: ${RUN_SEED_VAL}"
echo "============================================================"

# 我們將執行剛寫好的自動化預訓練與評估腳本
# 它會預訓練 400 epochs，隨後自動進行 50 epochs 的線性探測與 KNN 評估
python main/run_set_ablation.py

echo "============================================================"
echo "✅ SET Experiment Completed successfully! $(date)"
echo "============================================================"
