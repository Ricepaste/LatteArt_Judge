# main/visualize_kernel_pruning.py
import os
import argparse
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

# 固定隨機種子
torch.manual_seed(42)

def load_sparse_model(method, encoder_path, target_sparsity=0.99, use_erk=True, protect_highway=False):
    """
    依照 evaluate_model.py 的邏輯加載模型並注入權重
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if method == "hebbian":
        from src.training.Hebbian_train import Hebbian_SSL_Trainer
        dummy_trainer = Hebbian_SSL_Trainer(
            pretrained_model_class=models.resnet18,
            target_sparsity=target_sparsity,
            use_erk=use_erk,
            protect_highway=protect_highway
        )
        simsiam_model = dummy_trainer.model.to(device)
    else:
        import src.training.SimSiam_train as SimSiam_train
        dummy_trainer = SimSiam_train.SimSiam_Model(
            pretrained_model=models.resnet18,
        )
        simsiam_model = dummy_trainer.model.to(device)
        
    print(f"Loading weights from {encoder_path}...")
    state_dict = torch.load(encoder_path, map_location=device, weights_only=True)
    simsiam_model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully!")
    
    # 關閉 hebbian 統計
    if hasattr(simsiam_model, 'set_hebbian_enable'):
        simsiam_model.set_hebbian_enable(False)
        
    return simsiam_model.encoder, device

def analyze_and_plot_kernels(encoder, model_title, output_dir="runs/visualizations"):
    """
    遍歷 Encoder 中的卷積層，計算每個卷積核 (Kernel) 是否被完全剪枝，並繪製熱力圖
    """
    os.makedirs(output_dir, exist_ok=True)
    conv_layers = []
    
    # 收集有權重且維度為 4 (Conv2d) 的層
    for name, module in encoder.named_modules():
        # 如果是 HebbianSparseLayer，其卷積層包在 module.layer 中
        target_layer = module
        if hasattr(module, 'layer'):
            target_layer = module.layer
            
        if isinstance(target_layer, nn.Conv2d):
            # 排除 1x1 卷積，因為 1x1 卷積核只有 1x1，不易以熱力圖展示感受野剪枝，且通常會被保護
            if target_layer.kernel_size != (1, 1) and target_layer.kernel_size != 1:
                conv_layers.append((name, target_layer))
                
    print(f"Found {len(conv_layers)} standard Conv2d layers (excluding 1x1). Generating heatmaps...")
    
    # 挑選 ResNet-18 四個不同 Stage 具代表性的 3x3 卷積層進行對比
    # 例如：conv1 (Stem), layer1.0.conv1, layer2.0.conv1, layer3.0.conv1, layer4.0.conv1
    selected_layers = []
    candidates = ["conv1", "layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]
    
    for name, layer in conv_layers:
        # Hebbian 層的名字可能是 'encoder.layer1.0.conv1'，因此用 in 判定
        if any(c in name for c in candidates):
            selected_layers.append((name, layer))
            
    if not selected_layers:
        selected_layers = conv_layers[:4] # 若無匹配，預設取前四層
        
    num_selected = len(selected_layers)
    fig, axes = plt.subplots(1, num_selected, figsize=(5 * num_selected, 4.5))
    if num_selected == 1:
        axes = [axes]
        
    # 設定圖表風格：深色高亮風格 (e.g. viridis 或是 hot)
    cmap = "inferno" # inferno 黑色背景，高亮為黃色/白色，非常適合呈現「被剪掉的高亮點」
    
    for idx, (name, layer) in enumerate(selected_layers):
        weight = layer.weight.detach().cpu() # Shape: (C_out, C_in, Kh, Kw)
        C_out, C_in, Kh, Kw = weight.shape
        
        # 計算每個卷積核 (Kh x Kw) 的 L1 Norm
        # 若整個卷積核的權重絕對值之和小於 1e-7，代表該連接被完全剪枝 (Pruned)
        kernel_norms = weight.abs().sum(dim=(2, 3)) # Shape: (C_out, C_in)
        
        # 建立高亮矩陣：被完全剪除的 Kernel 顯示為 1.0 (高亮)，未被剪除的顯示為 0.0 (暗色)
        pruned_heatmap = (kernel_norms < 1e-7).float().numpy()
        
        pruned_ratio = np.mean(pruned_heatmap) * 100.0
        print(f"  Layer {name} | Size: {C_out}x{C_in} | Fully Pruned Kernels: {pruned_ratio:.2f}%")
        
        ax = axes[idx]
        im = ax.imshow(pruned_heatmap, cmap=cmap, aspect='auto', interpolation='nearest')
        
        ax.set_title(f"{name}\nFully Pruned: {pruned_ratio:.1f}%", fontsize=11, fontweight='bold')
        ax.set_ylabel("Output Channels", fontsize=9)
        ax.set_xlabel("Input Channels", fontsize=9)
        
        # 標註尺度
        if C_out > 128 or C_in > 128:
            # 減少標籤密度避免擁擠
            ax.set_xticks(np.arange(0, C_in, C_in // 4))
            ax.set_yticks(np.arange(0, C_out, C_out // 4))
            
    fig.suptitle(f"Kernel-Level Pruning Map ({model_title})\nHighlit points = Fully Pruned Kernels (All weights are 0)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 存檔
    save_path = os.path.join(output_dir, f"kernel_pruning_{model_title.replace(' ', '_').lower()}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Heatmap successfully saved to: {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Kernel-level Pruning Map using 2D Heatmaps")
    parser.add_argument("--hebbian_path", type=str, default="", help="Path to Hebbian model checkpoint (last.pt)")
    parser.add_argument("--rigl_path", type=str, default="", help="Path to RigL model checkpoint (last.pt)")
    parser.add_argument("--sparsity", type=float, default=0.99, help="Target global sparsity (default: 0.99)")
    parser.add_argument("--out_dir", type=str, default="runs/visualizations", help="Output directory for heatmaps")
    
    # 也支援從環境變數讀取 (對齊 evaluate_model.py)
    args = parser.parse_args()
    
    # 1. 處理 Hebbian (Ours)
    hebbian_path = args.hebbian_path or os.environ.get("HEBBIAN_PATH", "")
    if hebbian_path and os.path.exists(hebbian_path):
        print("\n" + "="*50)
        print("🔍 Analyzing Hebbian (Ours) Model Structure...")
        print("="*50)
        try:
            encoder, _ = load_sparse_model("hebbian", hebbian_path, target_sparsity=args.sparsity)
            analyze_and_plot_kernels(encoder, f"Hebbian Ours (Sparsity {args.sparsity})", args.out_dir)
        except Exception as e:
            print(f"❌ Failed to visualize Hebbian model: {e}")
            
    # 2. 處理 RigL
    rigl_path = args.rigl_path or os.environ.get("RIGL_PATH", "")
    if rigl_path and os.path.exists(rigl_path):
        print("\n" + "="*50)
        print("🔍 Analyzing RigL Model Structure...")
        print("="*50)
        try:
            encoder, _ = load_sparse_model("rigl", rigl_path, target_sparsity=args.sparsity)
            analyze_and_plot_kernels(encoder, f"RigL Baseline (Sparsity {args.sparsity})", args.out_dir)
        except Exception as e:
            print(f"❌ Failed to visualize RigL model: {e}")
            
    if not hebbian_path and not rigl_path:
        print("\n⚠️ No model path provided. Please specify model path via arguments or environment variables.")
        print("Usage Example:")
        print("  python visualize_kernel_pruning.py --hebbian_path runs/Hebbian_SSL_20260508-094600/last.pt --sparsity 0.96")
        print("  python visualize_kernel_pruning.py --rigl_path runs/shuffleNet_v05_SimSiam__5/last.pt --sparsity 0.99")
