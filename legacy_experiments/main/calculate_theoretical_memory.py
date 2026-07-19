# main/calculate_theoretical_memory.py
import torch
import torchvision.models as models
import torch.nn as nn
import os
import csv
import sys

class MemoryProfiler:
    def __init__(self, model):
        self.model = model
        self.layer_info = []
        self.hooks = []
        self._register_hooks()

    def _hook_fn(self, module, input, output):
        # 取得輸入與輸出的 Element 數量
        in_tensor = input[0]
        out_tensor = output
        
        in_elements = in_tensor.numel()
        out_elements = out_tensor.numel()
        
        # 以 Float32 (4 bytes) 計算記憶體大小 (MB)
        in_mb = (in_elements * 4) / (1024 ** 2)
        out_mb = (out_elements * 4) / (1024 ** 2)
        
        self.layer_info.append({
            "name": module.__class__.__name__,
            "in_shape": list(in_tensor.shape),
            "out_shape": list(out_tensor.shape),
            "in_mb": in_mb,
            "out_mb": out_mb
        })

    def _register_hooks(self):
        # 我們主要統計 Conv2d 和 Linear 層，因為它們是記憶體消耗的大頭
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def profile_model(model_name, batch_size=128):
    if model_name == "resnet152":
        model = models.resnet152(weights=None)
    else:
        model = models.resnet18(weights=None)
        
    profiler = MemoryProfiler(model)
    x = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        _ = model(x)
        
    profiler.remove_hooks()
    
    # 計算模型參數量 (包含不需梯度的)
    num_params = sum(p.numel() for p in model.parameters())
    # 計算需要梯度的參數量
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 權重佔用 (Float32: 4 bytes)
    weight_mb = (num_params * 4) / (1024 ** 2)
    # 梯度佔用 (每個可訓練參數對應一個梯度)
    grad_mb = (num_trainable_params * 4) / (1024 ** 2)
    # 優化器狀態佔用 (以 SGD + Momentum 為基準，動量緩衝區大小與可訓練參數相同)
    opt_sgd_mb = (num_trainable_params * 4) / (1024 ** 2)
    # 以 Adam 為基準的優化器狀態 (Adam 保留兩個動量緩衝區，為 2x 參數大小)
    opt_adam_mb = (num_trainable_params * 8) / (1024 ** 2)
    
    return {
        "layer_info": profiler.layer_info,
        "num_params": num_params,
        "weight_mb": weight_mb,
        "grad_mb": grad_mb,
        "opt_sgd_mb": opt_sgd_mb,
        "opt_adam_mb": opt_adam_mb
    }

def save_to_csv(layer_info, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Layer Type", "Input Shape", "Output Shape", "Input (MB)", "Output (MB)", "RigL Cumulative Input (MB)", "Hebbian Local (MB)"])
        
        rigl_accum = 0.0
        for idx, layer in enumerate(layer_info):
            rigl_accum += layer["in_mb"]
            hebbian_local = layer["in_mb"] + layer["out_mb"]
            writer.writerow([
                idx,
                layer["name"],
                str(layer["in_shape"]),
                str(layer["out_shape"]),
                f"{layer['in_mb']:.4f}",
                f"{layer['out_mb']:.4f}",
                f"{rigl_accum:.4f}",
                f"{hebbian_local:.4f}"
            ])
    print(f"Successfully saved detailed step-by-step table to: {filepath}")

def print_summary(model_name, profile_results):
    layer_info = profile_results["layer_info"]
    rigl_accum = 0.0
    hebbian_peaks = []
    
    for layer in layer_info:
        rigl_accum += layer["in_mb"]
        hebbian_peaks.append(layer["in_mb"] + layer["out_mb"])
        
    hebbian_peak = max(hebbian_peaks)
    bottleneck_layer_idx = hebbian_peaks.index(hebbian_peak)
    bottleneck_layer = layer_info[bottleneck_layer_idx]
    
    w_mb = profile_results["weight_mb"]
    g_mb = profile_results["grad_mb"]
    opt_sgd = profile_results["opt_sgd_mb"]
    opt_adam = profile_results["opt_adam_mb"]
    
    # 預估 CUDA Context 基本開銷為 300 MB
    cuda_context = 300.0
    
    # 計算總體理論顯存佔用 (以 SGD 訓練為例)
    total_rigl_sgd = cuda_context + w_mb + g_mb + opt_sgd + rigl_accum
    total_hebbian_sgd = cuda_context + w_mb + g_mb + opt_sgd + hebbian_peak
    
    print("=" * 100)
    print(f"💡 THEORETICAL VRAM BREAKDOWN: {model_name.upper()} (BS=128, float32, SGD w/ Momentum)")
    print("=" * 100)
    print(f"1. CUDA Context (Est. Baseline):          {cuda_context:8.2f} MB")
    print(f"2. Model Weights (Parameters):            {w_mb:8.2f} MB  ({profile_results['num_params']/1e6:.2f}M params)")
    print(f"3. Gradients:                             {g_mb:8.2f} MB")
    print(f"4. Optimizer States (SGD Momentum):       {opt_sgd:8.2f} MB  (Adam would be {opt_adam:.2f} MB)")
    print("-" * 100)
    print(f"5. Activation Memory (RigL / Backward):   {rigl_accum:8.2f} MB")
    print(f"5. Activation Memory (Hebbian / Peak):    {hebbian_peak:8.2f} MB")
    print("-" * 100)
    print(f"🔥 TOTAL ESTIMATED VRAM REQUIREMENT:")
    print(f"   - RigL Training Peak:                  {total_rigl_sgd:8.2f} MB")
    print(f"   - Hebbian Training Peak:               {total_hebbian_sgd:8.2f} MB")
    print(f"   - Hebbian saves:                       {total_rigl_sgd - total_hebbian_sgd:8.2f} MB ({(1 - total_hebbian_sgd/total_rigl_sgd)*100:.1f}% reduction)")
    print("-" * 100)
    
    # 激活值佔比分析
    rigl_act_ratio = (rigl_accum / total_rigl_sgd) * 100
    hebbian_act_ratio = (hebbian_peak / total_hebbian_sgd) * 100
    print(f">> Activations represent {rigl_act_ratio:.1f}% of total VRAM in RigL.")
    print(f">> Activations represent {hebbian_act_ratio:.1f}% of total VRAM in Hebbian.")
    print("=" * 100 + "\n")

def main():
    main_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(main_dir, "memory_profile_results")
    
    print("🔬 Running Theoretical VRAM Breakdown Profiler for BOTH ResNet-18 & ResNet-152...")
    print("(This process runs entirely on CPU to avoid GPU VRAM impact.)\n")
    
    # 1. Profile ResNet-18
    print("Profiling ResNet-18...")
    r18_results = profile_model("resnet18")
    r18_csv = os.path.join(output_dir, "resnet18_theoretical_memory.csv")
    save_to_csv(r18_results["layer_info"], r18_csv)
    print_summary("resnet18", r18_results)
    
    # 2. Profile ResNet-152
    print("Profiling ResNet-152...")
    r152_results = profile_model("resnet152")
    r152_csv = os.path.join(output_dir, "resnet152_theoretical_memory.csv")
    save_to_csv(r152_results["layer_info"], r152_csv)
    print_summary("resnet152", r152_results)

if __name__ == "__main__":
    main()
