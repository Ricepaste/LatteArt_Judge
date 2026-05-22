# main/visualize_kernel_pruning.py
import os
import argparse
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import traceback

# 固定隨機種子
torch.manual_seed(42)

def load_sparse_model(method, encoder_path, target_sparsity=0.99, use_erk=True, protect_highway=True):
    """
    依照 evaluate_model.py 的邏輯加載模型並注入權重，支援動態偵測 Backbone (ResNet18 vs ShuffleNetV2)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Detecting backbone for {encoder_path}...")
    state_dict = torch.load(encoder_path, map_location=device, weights_only=True)
    # 移出 module. 前綴以確保相容性
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    is_shufflenet = False
    
    # 1. 優先檢查 projector 的第一層 Linear 的輸入維度
    for key, param in state_dict.items():
        if "projector.1" in key and "weight" in key:
            in_features = param.shape[1]
            if in_features == 1024:
                is_shufflenet = True
            elif in_features == 512:
                is_shufflenet = False
            break
    else:
        # 2. 如果沒有 projector 相關的權重，則檢查是否有 branch (ShuffleNet 獨有)
        for key in state_dict.keys():
            if "branch" in key:
                is_shufflenet = True
                break
        else:
            # 3. 根據路徑名稱進行備份偵測
            if "shuffle" in encoder_path.lower() or "shufflenet" in encoder_path.lower():
                is_shufflenet = True
                
    model_class = models.shufflenet_v2_x0_5 if is_shufflenet else models.resnet18
    backbone_name = "ShuffleNetV2_0.5" if is_shufflenet else "ResNet18"
    print(f"Detected backbone: {backbone_name}")
    
    if method == "hebbian":
        from src.training.Hebbian_train import Hebbian_SSL_Trainer
        dummy_trainer = Hebbian_SSL_Trainer(
            pretrained_model_class=model_class,
            target_sparsity=target_sparsity,
            use_erk=use_erk,
            protect_highway=protect_highway
        )
        simsiam_model = dummy_trainer.model.to(device)
    else:
        import src.training.SimSiam_train as SimSiam_train
        dummy_trainer = SimSiam_train.SimSiam_Model(
            pretrained_model=model_class,
        )
        simsiam_model = dummy_trainer.model.to(device)
        
    print(f"Loading weights into the model...")
    simsiam_model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully!")
    
    # 關閉 hebbian 統計
    if hasattr(simsiam_model, 'set_hebbian_enable'):
        simsiam_model.set_hebbian_enable(False)
        
    return simsiam_model.encoder, device, backbone_name

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
                
    print(f"Found {len(conv_layers)} standard Conv2d layers (excluding 1x1). Selecting representative layers...")
    
    # 挑選不同 Stage 具代表性的 3x3 卷積層進行對比
    # 對於 ResNet-18，名稱可能是 "0" (Stem) 或 "4.0.conv1", "5.0.conv1", "6.0.conv1", "7.0.conv1"
    # 對於 ShuffleNet-V2，名稱可能是 "0.0" (Stem) 或 "2.0.branch2.2", "3.0.branch2.2", "4.0.branch2.2"
    selected_layers = []
    
    # 1. 找 Stem 第一個卷積層
    for name, layer in conv_layers:
        if name == "0" or name == "0.0" or name == "conv1" or name.endswith(".conv1") or (name.endswith(".0") and not ("." in name[:-2])):
            selected_layers.append((name, layer))
            break
            
    # 2. 找其他代表性的中間層
    candidates = [
        # ResNet-18 standard block layers
        "4.0.conv1", "5.0.conv1", "6.0.conv1", "7.0.conv1",
        # ShuffleNet-V2 stage blocks
        "2.0.branch2.2", "3.0.branch2.2", "4.0.branch2.2",
        "stage2.0.branch2.2", "stage3.0.branch2.2", "stage4.0.branch2.2"
    ]
    for name, layer in conv_layers:
        if any(c in name for c in candidates):
            # 避免重複添加第一層
            if not any(selected[0] == name for selected in selected_layers):
                selected_layers.append((name, layer))
                
    if not selected_layers:
        selected_layers = conv_layers[:4] # 若無匹配，預設取前四層
        
    num_selected = len(selected_layers)
    print(f"Plotting {num_selected} selected layers: {[n for n, _ in selected_layers]}")
    
    fig, axes = plt.subplots(1, num_selected, figsize=(5 * num_selected, 4.5))
    if num_selected == 1:
        axes = [axes]
        
    # 設定圖表風格：深色高亮風格 (e.g. viridis 或是 hot)
    cmap = "inferno" # inferno 黑色背景，高亮為黃色/白色，非常適合呈現「被剪掉的高亮點」
    
    for idx, (name, layer) in enumerate(selected_layers):
        weight = layer.weight.detach().cpu() # Shape: (C_out, C_in_divided_by_groups, Kh, Kw)
        C_out, C_in_g, Kh, Kw = weight.shape
        
        # 計算每個卷積核 (Kh x Kw) 的 L1 Norm
        # 若整個卷積核的權重絕對值之和小於 1e-7，代表該連接被完全剪枝 (Pruned)
        kernel_norms = weight.abs().sum(dim=(2, 3)) # Shape: (C_out, C_in_g)
        
        # 建立高亮矩陣：被完全剪除的 Kernel 顯示為 1.0 (高亮)，未被剪除的顯示為 0.0 (暗色)
        pruned_heatmap = (kernel_norms < 1e-7).float().numpy()
        
        pruned_ratio = np.mean(pruned_heatmap) * 100.0
        in_ch = layer.in_channels
        out_ch = layer.out_channels
        print(f"  Layer {name} | Size: {out_ch}x{in_ch} | Fully Pruned Kernels: {pruned_ratio:.2f}%")
        
        ax = axes[idx]
        im = ax.imshow(pruned_heatmap, cmap=cmap, aspect='auto', interpolation='nearest')
        
        ax.set_title(f"{name}\n({out_ch}x{in_ch})\nFully Pruned: {pruned_ratio:.1f}%", fontsize=10, fontweight='bold')
        ax.set_ylabel("Output Channels", fontsize=9)
        ax.set_xlabel("Weight Channels (C_in/groups)", fontsize=9)
        
        # 標註尺度
        if C_out > 128 or C_in_g > 128:
            x_step = max(1, C_in_g // 4)
            y_step = max(1, C_out // 4)
            ax.set_xticks(np.arange(0, C_in_g, x_step))
            ax.set_yticks(np.arange(0, C_out, y_step))
        else:
            x_step = max(1, C_in_g // 4) if C_in_g > 4 else 1
            y_step = max(1, C_out // 8) if C_out > 8 else 1
            ax.set_xticks(np.arange(0, C_in_g, x_step))
            ax.set_yticks(np.arange(0, C_out, y_step))
            
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
    
    # Hebbian 設定相關參數
    parser.add_argument("--use_erk", action="store_true", default=True, help="Use ERK sparsity distribution")
    parser.add_argument("--no_erk", action="store_false", dest="use_erk", help="Disable ERK sparsity distribution")
    parser.add_argument("--protect_highway", action="store_true", default=True, help="Protect 1x1/highway layers")
    parser.add_argument("--no_protect_highway", action="store_false", dest="protect_highway", help="Do not protect 1x1/highway layers")
    
    args = parser.parse_args()
    
    # 1. 處理 Hebbian (Ours)
    hebbian_path = args.hebbian_path or os.environ.get("HEBBIAN_PATH", "")
    if hebbian_path and os.path.exists(hebbian_path):
        print("\n" + "="*50)
        print("🔍 Analyzing Hebbian (Ours) Model Structure...")
        print("="*50)
        try:
            encoder, _, backbone_name = load_sparse_model(
                "hebbian", 
                hebbian_path, 
                target_sparsity=args.sparsity,
                use_erk=args.use_erk,
                protect_highway=args.protect_highway
            )
            analyze_and_plot_kernels(encoder, f"Hebbian Ours ({backbone_name}, Sparsity {args.sparsity})", args.out_dir)
        except Exception as e:
            traceback.print_exc()
            print(f"❌ Failed to visualize Hebbian model: {e}")
            
    # 2. 處理 RigL
    rigl_path = args.rigl_path or os.environ.get("RIGL_PATH", "")
    if rigl_path and os.path.exists(rigl_path):
        print("\n" + "="*50)
        print("🔍 Analyzing RigL Model Structure...")
        print("="*50)
        try:
            encoder, _, backbone_name = load_sparse_model(
                "rigl", 
                rigl_path, 
                target_sparsity=args.sparsity,
                use_erk=args.use_erk,
                protect_highway=args.protect_highway
            )
            analyze_and_plot_kernels(encoder, f"RigL Baseline ({backbone_name}, Sparsity {args.sparsity})", args.out_dir)
        except Exception as e:
            traceback.print_exc()
            print(f"❌ Failed to visualize RigL model: {e}")
            
    if not hebbian_path and not rigl_path:
        print("\n⚠️ No model path provided. Please specify model path via arguments or environment variables.")
        print("Usage Example:")
        print("  python visualize_kernel_pruning.py --hebbian_path runs/Hebbian_SSL_20260508-094600/last.pt --sparsity 0.96")
        print("  python visualize_kernel_pruning.py --rigl_path runs/shuffleNet_v05_SimSiam__4/last.pt --sparsity 0.99")
