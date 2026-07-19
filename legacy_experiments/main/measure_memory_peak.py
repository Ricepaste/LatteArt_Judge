# main/measure_memory_peak.py
import os
import sys
import torch
import torchvision.models as models
import torch.nn.functional as F

# 確保在 container 內 /app 能被 import 讀取到
sys.path.insert(0, "/app")

from src.module.hebbian_SimSiam_Module import Hebbian_SimSiam
import src.module.SimSiam_Module as SimSiam_Module
import rigl_torch.RigL as rigl_baseline_module

def measure_hebbian_peak(device, batch_size=128):
    print("\n--- Measuring Hebbian (Ours) Update Memory Peak ---")
    backbone = models.resnet18(weights=None)
    model = Hebbian_SimSiam(
        backbone,
        model_type='resnet',
        encoder_output_dim=512,
        projector_inner_dim=2048,
        target_sparsity=0.99
    ).to(device)
    
    # 模擬輸入
    x1 = torch.randn(batch_size, 3, 224, 224, device=device)
    x2 = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # 清空記憶體統計
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    # 1. 啟用 Hebbian 統計
    model.set_hebbian_enable(True)
    
    # 2. 前向傳播 (計算 Pearson/Anti-Hebbian 相關度)
    # Hebbian 不需要保留計算圖來做結構更新
    with torch.no_grad():
        p1, p2, z1, z2 = model(x1, x2)
    
    # 3. 執行拓撲更新 (Prune & Grow)
    topology_changes = model.update_topology(grow_ratio=0.2)
    
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    final_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    print(f"Hebbian Initial Memory: {initial_memory:.2f} MB")
    print(f"Hebbian Peak Memory during Update: {peak_memory:.2f} MB")
    print(f"Hebbian Net Peak Overhead: {peak_memory - initial_memory:.2f} MB")
    print(f"Hebbian Final Memory: {final_memory:.2f} MB")
    return peak_memory - initial_memory

def measure_rigl_peak(device, batch_size=128):
    print("\n--- Measuring RigL (Baseline) Update Memory Peak ---")
    backbone = models.resnet18(weights=None)
    model = SimSiam_Module.SimSiam(
        backbone,
        model_type='resnet',
        encoder_output_dim=512,
        projector_inner_dim=2048
    ).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    criterion = SimSiam_Module.SimSiamLoss()
    
    # 初始化 RigL Scheduler
    pruner = rigl_baseline_module.RigLScheduler(
        model=model,
        optimizer=optimizer,
        dense_allocation=0.01, # 99% sparsity
        sparsity_distribution="uniform",
        T_end=1000,
        delta=100,
        alpha=0.3,
        ignore_linear_layers=False
    )
    
    x1 = torch.randn(batch_size, 3, 224, 224, device=device)
    x2 = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # 清空記憶體統計
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    # 1. 設定 step 為更新前一步，使得 backward pass 會觸發 dense gradient 的計算與累積
    pruner.step = pruner.delta_T - 1
    
    # 2. 前向傳播
    p1, p2, z1, z2 = model(x1, x2)
    loss = criterion(p1, p2, z1, z2)
    
    # 3. 反向傳播 (這會觸發 IndexMaskHook 進行 dense_grad 累積)
    loss.backward()
    
    # 4. 模擬觸發 RigL 更新步
    # pruner() 內部會加 step 並呼叫 _rigl_step()
    pruner() 
    
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    final_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    print(f"RigL Initial Memory: {initial_memory:.2f} MB")
    print(f"RigL Peak Memory during Update: {peak_memory:.2f} MB")
    print(f"RigL Net Peak Overhead: {peak_memory - initial_memory:.2f} MB")
    print(f"RigL Final Memory: {final_memory:.2f} MB")
    return peak_memory - initial_memory

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available. Peak memory measurement requires GPU.")
        return
        
    print("="*60)
    print("🔬 GPU Memory Peak Profiler for Sparse Topology Update")
    print("="*60)
    
    hebbian_overhead = measure_hebbian_peak(device)
    rigl_overhead = measure_rigl_peak(device)
    
    print("\n" + "="*60)
    print("📊 LIVE COMPARISON SUMMARY (Net Peak Memory Overhead)")
    print("="*60)
    print(f"Hebbian (Ours, Forward Correlation): {hebbian_overhead:.2f} MB")
    print(f"RigL (Baseline, Backward Dense Grad): {rigl_overhead:.2f} MB")
    ratio = rigl_overhead / hebbian_overhead if hebbian_overhead > 0 else 0
    print(f">> RigL requires {ratio:.2f}x more peak memory during structural update!")
    print("="*60)
    
    print("\n" + "="*60)
    print("📌 REFERENCE HISTORICAL BASELINE (RTX 4090 - 2026-06-11)")
    print("="*60)
    print("Hebbian (Ours, Forward Correlation): 5179.61 MB")
    print("RigL (Baseline, Backward Dense Grad): 5445.12 MB")
    print(">> RigL requires 1.05x more peak memory during structural update!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
