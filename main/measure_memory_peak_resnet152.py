# main/measure_memory_peak_resnet152.py
"""
🔬 ResNet-152 GPU Memory Peak Profiler for Sparse Topology Update (Hebbian vs. RigL)

This script measures the peak VRAM consumption during the structural update step
for both our Hebbian method and the RigL baseline using a ResNet-152 backbone.

Usage on another server:
1. Copy the project repository to the new server.
2. Build the Docker container (see _DART_docker_build_notes.md) or run within a local PyTorch environment.
3. Run the script:
   python main/measure_memory_peak_resnet152.py --batch-size 128 --device cuda:0
"""

import os
import sys
import argparse
import torch
import torchvision.models as models
import torch.nn.functional as F

# Ensure the root /app or current repo path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/app")

try:
    from src.module.hebbian_SimSiam_Module import Hebbian_SimSiam
    import src.module.SimSiam_Module as SimSiam_Module
    import rigl_torch.RigL as rigl_baseline_module
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please make sure you are running this script from the workspace root or inside the Docker container.")
    sys.exit(1)

def measure_hebbian_peak(device, batch_size):
    print(f"\n--- Measuring Hebbian (Ours) Update Memory Peak (ResNet-152, BS={batch_size}) ---")
    
    # ResNet-152 Backbone
    backbone = models.resnet152(weights=None)
    
    # Note: ResNet-152 has a final feature map of 2048 channels, so encoder_output_dim MUST be 2048.
    model = Hebbian_SimSiam(
        backbone,
        model_type='resnet',
        encoder_output_dim=2048, 
        projector_inner_dim=2048,
        target_sparsity=0.99
    ).to(device)
    
    # Generate dummy input
    x1 = torch.randn(batch_size, 3, 224, 224, device=device)
    x2 = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Clean up CUDA memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    # 1. Enable Hebbian statistics hook
    model.set_hebbian_enable(True)
    
    print("Executing Forward Pass under no_grad()...")
    # 2. Forward pass (computes Pearson correlation layer-by-layer)
    with torch.no_grad():
        p1, p2, z1, z2 = model(x1, x2)
    
    print("Executing Topology Update (unfold + correlation calculation)...")
    # 3. Perform topology update (Prune & Grow)
    topology_changes = model.update_topology(grow_ratio=0.2)
    
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    final_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    net_overhead = peak_memory - initial_memory
    print(f"Hebbian Initial Memory: {initial_memory:.2f} MB")
    print(f"Hebbian Peak Memory during Update: {peak_memory:.2f} MB")
    print(f"Hebbian Net Peak Overhead: {net_overhead:.2f} MB")
    print(f"Hebbian Final Memory: {final_memory:.2f} MB")
    return net_overhead

def measure_rigl_peak(device, batch_size):
    print(f"\n--- Measuring RigL (Baseline) Update Memory Peak (ResNet-152, BS={batch_size}) ---")
    
    # ResNet-152 Backbone
    backbone = models.resnet152(weights=None)
    
    # Note: ResNet-152 has a final feature map of 2048 channels, so encoder_output_dim MUST be 2048.
    model = SimSiam_Module.SimSiam(
        backbone,
        model_type='resnet',
        encoder_output_dim=2048,
        projector_inner_dim=2048
    ).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    criterion = SimSiam_Module.SimSiamLoss()
    
    # Initialize RigL Scheduler
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
    
    # Generate dummy input
    x1 = torch.randn(batch_size, 3, 224, 224, device=device)
    x2 = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Clean up CUDA memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    # 1. Set the step right before the update step to trigger dense gradients logging in the backward hook
    pruner.step = pruner.delta_T - 1
    
    print("Executing Forward Pass...")
    # 2. Forward pass
    p1, p2, z1, z2 = model(x1, x2)
    loss = criterion(p1, p2, z1, z2)
    
    print("Executing Backward Pass (accumulating dense gradients)...")
    # 3. Backward pass (triggers IndexMaskHook to store dense gradients for all sparsified layers)
    loss.backward()
    
    print("Executing RigL Update Step (prune & grow based on gradients)...")
    # 4. Trigger RigL update step
    pruner() 
    
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    final_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    net_overhead = peak_memory - initial_memory
    print(f"RigL Initial Memory: {initial_memory:.2f} MB")
    print(f"RigL Peak Memory during Update: {peak_memory:.2f} MB")
    print(f"RigL Net Peak Overhead: {net_overhead:.2f} MB")
    print(f"RigL Final Memory: {final_memory:.2f} MB")
    return net_overhead

def main():
    parser = argparse.ArgumentParser(description="GPU Peak Memory Profiler for ResNet-152")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for profiling (default: 128)")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (default: cuda)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and "cuda" in args.device:
        print("⚠️ CUDA requested but not available. Falling back to CPU, but peak VRAM stats will not be recorded.")
        device = torch.device("cpu")
        
    print("=" * 70)
    print(f"🔬 GPU Memory Peak Profiler (ResNet-152, BS={args.batch_size})")
    print("=" * 70)
    print(f"Device: {device}")
    
    hebbian_overhead = 0.0
    rigl_overhead = 0.0
    
    # 1. Measure Hebbian
    try:
        hebbian_overhead = measure_hebbian_peak(device, args.batch_size)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ Hebbian update step ran Out of Memory (OOM)!")
            print("Try reducing --batch-size.")
        else:
            raise e
            
    # 2. Measure RigL
    try:
        rigl_overhead = measure_rigl_peak(device, args.batch_size)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ RigL update step ran Out of Memory (OOM)!")
            print("Try reducing --batch-size.")
        else:
            raise e

    # Summary Comparison
    if hebbian_overhead > 0 and rigl_overhead > 0:
        print("\n" + "=" * 70)
        print("📊 MEASUREMENT SUMMARY (ResNet-152, Net Peak Memory Overhead)")
        print("=" * 70)
        print(f"Hebbian (Ours, Forward Correlation): {hebbian_overhead:10.2f} MB")
        print(f"RigL (Baseline, Backward Dense Grad): {rigl_overhead:10.2f} MB")
        ratio = rigl_overhead / hebbian_overhead
        diff = rigl_overhead - hebbian_overhead
        print(f">> RigL requires {ratio:.2f}x more peak memory during structural update (+{diff:.2f} MB)!")
        print("=" * 70 + "\n")
        
        # Add guidance on deep vs shallow scaling
        print("💡 Scaling Context:")
        print("   - For ResNet-18, the difference in net overhead was small (~1.05x).")
        print("   - For ResNet-152, since RigL accumulates activation gradients across all 152 layers")
        print("     while Hebbian releases activations layer-by-layer, the gap should widen dramatically.")
    else:
        print("\n⚠️ Profile did not complete successfully due to OOM or errors on one of the runs.")

if __name__ == "__main__":
    main()
