import torch
import glob
import os

print("--- Sparsity Check for Local Model Weights ---")
paths = glob.glob("main/runs/*/last.pt") + glob.glob("runs/*/last.pt")
# De-duplicate paths
paths = list(set([os.path.abspath(p) for p in paths]))
paths.sort()

for path in paths:
    if not os.path.exists(path):
        continue
    try:
        state_dict = torch.load(path, map_location="cpu")
        total_elements = 0
        zero_elements = 0
        
        # Check all weight parameters in the encoder
        for name, tensor in state_dict.items():
            # Standard way to calculate weight sparsity in ResNet/ShuffleNet modules
            if "weight" in name and tensor.is_floating_point():
                total_elements += tensor.numel()
                zero_elements += (tensor == 0).sum().item()
                
        if total_elements > 0:
            sparsity = zero_elements / total_elements
            print(f"{os.path.basename(os.path.dirname(path))}: sparsity = {sparsity*100:.2f}% (zeros: {zero_elements}/{total_elements})")
        else:
            print(f"{os.path.basename(os.path.dirname(path))}: No float weight tensors found")
    except Exception as e:
        print(f"Error loading {path}: {e}")
