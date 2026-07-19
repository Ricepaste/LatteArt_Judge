# main/measure_physical_activation_ratio.py
import torch
import torchvision.models as models
import torch.nn as nn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU to measure physical VRAM.")
        return
        
    batch_size = 128
    print("=" * 60)
    print(f"🔬 Physical GPU Activation Memory Profiler (ResNet-18, BS={batch_size})")
    print("=" * 60)
    
    # Note: torch.cuda.memory_allocated() only tracks VRAM allocated by the *current* PyTorch process.
    # It will NOT include VRAM used by other processes running on the same GPU.
    torch.cuda.empty_cache()
    
    # 1. Initialize model and optimizer
    model = models.resnet18(weights=None).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Run a warm-up forward/backward pass so PyTorch allocates internal workspaces (e.g. cuDNN convolution kernels)
    x_dummy = torch.randn(batch_size, 3, 224, 224, device=device)
    y_dummy = torch.randint(0, 1000, (batch_size,), device=device)
    out = model(x_dummy)
    loss = criterion(out, y_dummy)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    torch.cuda.empty_cache()
    
    # ---------------- CORE MEASUREMENT START ----------------
    # Step 1: Measure VRAM before forward pass (contains weights + gradients + optimizer states)
    m_init = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    # Step 2: Perform forward pass (activations are stored here)
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 1000, (batch_size,), device=device)
    
    out = model(x)
    loss = criterion(out, y)
    
    # Step 3: Measure VRAM after forward pass (contains weights + gradients + optimizer + saved activations)
    m_fwd = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    # Step 4: Measure Peak VRAM during backward pass
    torch.cuda.reset_peak_memory_stats(device)
    loss.backward()
    m_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    # Step 5: Clean up gradients to prepare for output
    optimizer.zero_grad()
    
    # ---------------- CALCULATE RATIOS ----------------
    physical_activations = m_fwd - m_init
    activation_ratio = (physical_activations / m_fwd) * 100
    
    print(f"1. VRAM before Forward (Weights + Opt):    {m_init:8.2f} MB")
    print(f"2. VRAM after Forward (Weights + Opt + Act): {m_fwd:8.2f} MB")
    print(f"3. Peak VRAM during Backward:              {m_peak:8.2f} MB")
    print("-" * 60)
    print(f"💡 Measured Physical Activation VRAM:      {physical_activations:8.2f} MB")
    print(f"💡 Activations represent {activation_ratio:.1f}% of total forward VRAM.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
