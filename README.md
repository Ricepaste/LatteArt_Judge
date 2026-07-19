# Gradient-Free Dynamic Sparse Training via Local Statistics for Self-Supervised Convolutional Networks
*(基於局部統計量之無梯度動態稀疏訓練研究：以自監督式卷積網路為例)*

This repository contains the official PyTorch implementation for exploring biologically-inspired Hebbian rules in the context of Sparse Self-Supervised Learning (SSL). By dynamically pruning and growing network connections based on Anti-Hebbian principles, the network naturally forms highly structured feature clusters (Motifs) and achieves superior feature decoupling compared to random sparse topologies.

## Repository Structure

The repository has been streamlined to highlight the core contributions:

- `main/Hebbian.py`: The entry point for our proposed Hebbian Sparse SSL pretraining.
- `main/SimSiam.py`: The entry point for standard dense SimSiam pretraining (Baseline).
- `main/Dense.py`: The entry point for standard dense supervised learning (Baseline).
- `main/evaluate_model.py`: The standardized evaluation script (KNN and Linear Probing) for downstream tasks.
- `main/compare_mask_topology_local.py`: The explainability script used to compute topological differences, motif formations (Kurtosis), and feature decoupling (Jaccard similarity).
- `main/src/`: Contains all the core neural network layers, custom Hebbian routing logic, and data processing utilities.
- `legacy_experiments/`: An archive containing all previous ablation studies, memory profiling, and alternative baseline scripts used during the research phase.

## Environment Setup

You can build the environment using the provided Dockerfile or install the dependencies manually.

### Using Requirements
```bash
pip install -r requirements.txt
```

### Using Docker / Singularity
We provide a `Dockerfile` for containerized execution. You can build the image or convert it to a Singularity `.sif` file if you are working on HPC clusters (like TWCC).

## Quick Start Guide

### 1. Pretraining
To run the sparse pretraining using our proposed Hebbian method at 90% sparsity on CIFAR-100:
```bash
cd main
export TARGET_DATASET="cifar100"
export TARGET_SPARSITY=0.90
export NUM_EPOCHS=400
export RUN_SEED=42
export DATALOADER_WORKERS=8

python Hebbian.py
```
This will create a new folder under `main/runs/` containing the trained weights (e.g., `last.pt`).

### 2. Evaluation
To evaluate the pretrained weights using KNN and Linear Probing:
```bash
export ENCODER_PATH="runs/Hebbian_SSL_s90_seed42_c100_XXXXXXXX-XXXXXX/last.pt"
export METHOD="hebbian"
export TARGET_DATASET="cifar100"
export NUM_EPOCHS=50

python evaluate_model.py
```

### 3. Topological Analysis
To reproduce the topological motif analysis presented in the paper, you can compare the masks of our Hebbian method against a random sparse baseline (e.g., SET/RigL).

```bash
python compare_mask_topology_local.py \
    --ours_path "runs/Hebbian_SSL_.../last.pt" \
    --rigl_path "runs/SET_SSL_.../last.pt" \
    --save_dir "topology_results"
```
This will output a comprehensive Markdown report detailing the Jaccard similarity distributions, peak Kurtosis (proving small-world motifs), and a KDE visualization chart.

## Author & Contact

**Jia-Rong Kuo (郭家榕)**  
Master's Thesis, June 2026  
Department of Computer Science and Information Engineering  
National Central University (國立中央大學)  
Advisor: Dr. Hung-Hsuan Chen (陳弘軒 博士)

If you have any questions about the code, the paper, or the methodology, feel free to reach out:  
📧 Email: [n22126@gmail.com](mailto:n22126@gmail.com)
