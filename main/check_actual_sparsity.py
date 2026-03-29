import torch
import os

def calculate_actual_sparsity(model_path):
    print(f"Loading checkpoint from: {model_path}")
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return
        
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 處理 checkpoint 的可能格式
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    total_params = 0
    zero_params = 0
    
    print(f"{'Layer Name':<55} | {'Shape':<20} | {'Sparsity'}")
    print("-" * 95)
    
    for name, param in state_dict.items():
        # 篩選我們關心的權重，排除 BatchNorm 以及 bias 等通常不剪枝的部分
        # downsample.1 通常是 ResNet 的 BN 層
        if 'weight' in name and 'bn' not in name and 'downsample.1' not in name and param.dim() > 1:
            param_numel = param.numel()
            # 計算數值為 0 的數量 (預防浮點數誤差，取絕對值小於 1e-7)
            param_zeros = (param.abs() < 1e-7).sum().item()
            
            total_params += param_numel
            zero_params += param_zeros
            
            layer_sparsity = param_zeros / param_numel if param_numel > 0 else 0
            print(f"{name:<55} | {str(list(param.shape)):<20} | {layer_sparsity:.4f}")

    global_sparsity = zero_params / total_params if total_params > 0 else 0
    print("-" * 95)
    print(f"Total parameters computed: {total_params:,}")
    print(f"Total zero parameters:     {zero_params:,}")
    print(f"Global Actual Sparsity:    {global_sparsity:.4f} ({(global_sparsity*100):.2f}%)")

if __name__ == "__main__":
    # 使用完整的絕對路徑來確保一定能讀到
    OLD_MODEL_PATH = "./runs/Hebbian_SSL_20260326-174032/last.pt" 
    calculate_actual_sparsity(OLD_MODEL_PATH)
