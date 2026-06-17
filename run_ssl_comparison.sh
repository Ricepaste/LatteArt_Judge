#!/bin/bash
set -euo pipefail
set -x

# ============================================================
# 1. 工作目錄 & 環境設定 (對齊您在 TWCC 的路徑)
# ============================================================
WORKDIR="/home/sharpaste/repo/LatteArt_Judge"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ============================================================
# 2. 執行 80% 稀疏度隨機生長（SET）對照組實驗
# ============================================================
echo "============================================================"
echo "🚀 Starting Standardized SET (Random Growth) Experiment @ 80% Sparsity"
echo "🚀 Target Dataset: CIFAR-100 | Target Sparsity: 0.8"
echo "============================================================"

# 我們將執行剛寫好的自動化預訓練與評估腳本
# 它會預訓練 400 epochs，隨後自動進行 50 epochs 的線性探測與 KNN 評估
python main/run_set_ablation.py

echo "============================================================"
echo "✅ SET Experiment Completed successfully! $(date)"
echo "============================================================"
