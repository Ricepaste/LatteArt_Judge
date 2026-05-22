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

def get_clean_name(name, backbone):
    """
    將原始的模組名稱轉換為論文中易讀的簡潔名稱 (Paper-friendly names)
    """
    if "resnet" in backbone.lower():
        if name == "0" or "conv1" in name:
            return "Stem Conv (7x7)"
        elif "4.0.conv1" in name:
            return "Stage 1 Block 0 Conv1"
        elif "5.0.conv1" in name:
            return "Stage 2 Block 0 Conv1"
        elif "6.0.conv1" in name:
            return "Stage 3 Block 0 Conv1"
        elif "7.0.conv1" in name:
            return "Stage 4 Block 0 Conv1"
    else: # shufflenet
        if name == "0.0" or "conv1" in name:
            return "Stem Conv (3x3)"
        elif "2.0.branch2.3" in name:
            return "Stage 2 Block 0 DW-Conv"
        elif "3.0.branch2.3" in name:
            return "Stage 3 Block 0 DW-Conv"
        elif "4.0.branch2.3" in name:
            return "Stage 4 Block 0 DW-Conv"
    return name

def analyze_and_plot_kernels(encoder, model_title, backbone_name, output_dir="runs/visualizations", threshold=1e-7):
    """
    遍歷 Encoder 中的卷積層，計算每個卷積核 (Kernel) 是否被剪枝，並繪製論文等級的 2D 對比熱力圖。
    提供雙排對比圖：
      - 第一排：二值剪枝圖 (Binary Map: 亮色 = 被剪枝, 暗色 = 未剪枝)
      - 第二排：卷積核 L1-Norm 的連續強度圖 (Log-scaled Magnitude Map: 亮色 = 強連接, 暗色 = 弱/剪枝連接)
    """
    os.makedirs(output_dir, exist_ok=True)
    conv_layers = []
    
    # 收集有權重且維度為 4 (Conv2d) 的層
    for name, module in encoder.named_modules():
        target_layer = module
        if hasattr(module, 'layer'):
            target_layer = module.layer
            
        if isinstance(target_layer, nn.Conv2d):
            # 排除 1x1 卷積
            if target_layer.kernel_size != (1, 1) and target_layer.kernel_size != 1:
                conv_layers.append((name, target_layer))
                
    print(f"\n--- Conv2d Layers found in {backbone_name} ---")
    for n, l in conv_layers:
        print(f"  - {n} | Shape: {list(l.weight.shape)} | Kernel Size: {l.kernel_size}")
        
    print(f"\nSelecting representative layers for visualization...")
    
    selected_layers = []
    
    # 1. 挑選 Stem
    for name, layer in conv_layers:
        if name == "0" or name == "0.0" or name == "conv1" or name.endswith(".conv1") or (name.endswith(".0") and not ("." in name[:-2])):
            selected_layers.append((name, layer))
            break
            
    # 2. 挑選不同 Stage 的代表性 3x3 卷積層
    candidates = [
        # ResNet-18
        "4.0.conv1", "5.0.conv1", "6.0.conv1", "7.0.conv1",
        # ShuffleNet-V2 stage blocks (dw-conv at index 3 of branch2)
        "2.0.branch2.3", "3.0.branch2.3", "4.0.branch2.3",
        "stage2.0.branch2.3", "stage3.0.branch2.3", "stage4.0.branch2.3"
    ]
    for name, layer in conv_layers:
        if any(c in name for c in candidates):
            if not any(selected[0] == name for selected in selected_layers):
                selected_layers.append((name, layer))
                
    if not selected_layers:
        selected_layers = conv_layers[:4]
        
    num_selected = len(selected_layers)
    print(f"Selected {num_selected} layers: {[n for n, _ in selected_layers]}")
    
    # 使用論文標準格式美化畫布 (Row 1: Binary, Row 2: Log Magnitude)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    fig, axes = plt.subplots(2, num_selected, figsize=(4.2 * num_selected, 7.5), dpi=300)
    if num_selected == 1:
        axes = np.expand_dims(axes, axis=1) # 確保為 2D 陣列 (2, 1)
        
    for idx, (name, layer) in enumerate(selected_layers):
        weight = layer.weight.detach().cpu() # Shape: (C_out, C_in_g, Kh, Kw)
        C_out, C_in_g, Kh, Kw = weight.shape
        in_ch = layer.in_channels
        out_ch = layer.out_channels
        
        # 計算每個卷積核 (Kh x Kw) 的 L1 Norm
        kernel_norms = weight.abs().sum(dim=(2, 3)).numpy()
        
        # 1. 二值剪枝矩陣：低於 threshold 視為被剪枝 (1.0 = Pruned, 0.0 = Active)
        pruned_heatmap = (kernel_norms < threshold).astype(float)
        pruned_ratio = np.mean(pruned_heatmap) * 100.0
        print(f"  Layer {name} | Size: {out_ch}x{in_ch} | Pruned Kernels (threshold {threshold}): {pruned_ratio:.2f}%")
        
        # 2. 連續強度矩陣 (Log10 縮放以便清晰顯示微小權重差別)
        log_norms = np.log10(kernel_norms + 1e-10)
        
        paper_name = get_clean_name(name, backbone_name)
        
        # --- Row 1: Binary Pruning Heatmap (Inferno: 亮黃色為完全剪除，暗色為保留) ---
        ax0 = axes[0, idx]
        im0 = ax0.imshow(pruned_heatmap, cmap="inferno", aspect='auto', interpolation='nearest')
        ax0.set_title(f"{paper_name}\nBinary Pruning Map\n(Pruned: {pruned_ratio:.1f}%)", fontsize=10, fontweight='bold', pad=8)
        ax0.set_ylabel("Output Channels", fontsize=8)
        ax0.set_xlabel("Weight Channels (C_in/groups)", fontsize=8)
        ax0.tick_params(labelsize=7)
        
        # 設定刻度，防範單通道/少通道時的除零錯誤
        x_step = max(1, C_in_g // 4) if C_in_g > 4 else 1
        y_step = max(1, C_out // 8) if C_out > 8 else 1
        ax0.set_xticks(np.arange(0, C_in_g, x_step))
        ax0.set_yticks(np.arange(0, C_out, y_step))
        
        # 標示圖例或小字
        ax0.text(0.02, 0.98, f"Thresh: {threshold:.0e}", transform=ax0.transAxes, color="white",
                 fontsize=7, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
        
        # --- Row 2: Continuous Log L1-Norm Magnitude (Viridis: 亮黃色為大權重，暗紫色為被剪除) ---
        ax1 = axes[1, idx]
        im1 = ax1.imshow(log_norms, cmap="viridis", aspect='auto', interpolation='nearest')
        ax1.set_title("Kernel L1-Norm Strength\n(Log10 Magnitude)", fontsize=10, fontweight='bold', pad=8)
        ax1.set_ylabel("Output Channels", fontsize=8)
        ax1.set_xlabel("Weight Channels (C_in/groups)", fontsize=8)
        ax1.tick_params(labelsize=7)
        
        ax1.set_xticks(np.arange(0, C_in_g, x_step))
        ax1.set_yticks(np.arange(0, C_out, y_step))
        
        # 加入美觀的 colorbar
        cbar = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("log10(L1 Norm)", fontsize=7)

    fig.suptitle(f"Kernel-Level Pruning & Weight Strength Visualizations ({model_title})\n"
                 f"Row 1 (Binary): Bright Yellow = Fully Pruned Kernels  |  Row 2 (Continuous): Bright Yellow = Strong Active Connections", 
                 fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 儲存高品質的 PNG (方便查看) 與 PDF (向量圖，方便直接插入 LaTeX 論文)
    save_name = f"kernel_pruning_{model_title.replace(' ', '_').lower()}"
    save_path_png = os.path.join(output_dir, f"{save_name}.png")
    save_path_pdf = os.path.join(output_dir, f"{save_name}.pdf")
    
    plt.savefig(save_path_png, bbox_inches='tight', dpi=300)
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.close()
    print(f"\n🎉 Scientific figures successfully saved to:")
    print(f"  - PNG (300 DPI): {save_path_png}")
    print(f"  - PDF (Vector):   {save_path_pdf}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Kernel-level Pruning Map using 2D Heatmaps")
    parser.add_argument("--hebbian_path", type=str, default="", help="Path to Hebbian model checkpoint (last.pt)")
    parser.add_argument("--rigl_path", type=str, default="", help="Path to RigL model checkpoint (last.pt)")
    parser.add_argument("--sparsity", type=float, default=0.99, help="Target global sparsity (default: 0.99)")
    parser.add_argument("--threshold", type=float, default=1e-7, help="Threshold below which a kernel is considered pruned (default: 1e-7)")
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
            analyze_and_plot_kernels(
                encoder, 
                f"Hebbian Ours (Sparsity {args.sparsity})", 
                backbone_name, 
                args.out_dir,
                threshold=args.threshold
            )
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
            analyze_and_plot_kernels(
                encoder, 
                f"RigL Baseline (Sparsity {args.sparsity})", 
                backbone_name, 
                args.out_dir,
                threshold=args.threshold
            )
        except Exception as e:
            traceback.print_exc()
            print(f"❌ Failed to visualize RigL model: {e}")
            
    if not hebbian_path and not rigl_path:
        print("\n⚠️ No model path provided. Please specify model path via arguments or environment variables.")
        print("Usage Example:")
        print("  python visualize_kernel_pruning.py --hebbian_path runs/Hebbian_SSL_20260508-094600/last.pt --sparsity 0.96 --threshold 1e-6")
        print("  python visualize_kernel_pruning.py --rigl_path runs/shuffleNet_v05_SimSiam__4/last.pt --sparsity 0.99 --threshold 1e-6")
