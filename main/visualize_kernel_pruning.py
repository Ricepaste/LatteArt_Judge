# main/visualize_kernel_pruning.py
import os
import argparse
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
            return "Stem Conv"
        elif "4.0.conv1" in name:
            return "Stage 1 Block 0 Conv"
        elif "5.0.conv1" in name:
            return "Stage 2 Block 0 Conv"
        elif "6.0.conv1" in name:
            return "Stage 3 Block 0 Conv"
        elif "7.0.conv1" in name:
            return "Stage 4 Block 0 Conv"
    else: # shufflenet
        if name == "0.0" or "conv1" in name:
            return "Stem Conv"
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
    橫軸為從前到後（由左至右）的網路層 (Layers)，縱軸為同一層內部不同卷積核/連接的索引。
    不同層之間的寬度/通道數差異以中性淺灰色 (NaN Padding) 填充，呈現階梯狀的整體拓樸圖。
    
    提供雙面圖表並排：
      - 左圖：二值剪枝圖 (Binary Map: 亮黃色 = 被剪枝, 暗黑色 = 未剪枝, 灰色 = 無此連接)
      - 右圖：卷積核 L1-Norm 的連續強度圖 (Log-scaled Magnitude Map: 亮色 = 強連接, 暗色 = 弱/剪枝連接, 灰色 = 無此連接)
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
    
    # 找出最大通道/卷積核數以進行對齊
    layer_flat_norms = []
    max_kernels = 0
    for name, layer in selected_layers:
        weight = layer.weight.detach().cpu()
        C_out, C_in_g, Kh, Kw = weight.shape
        flat_norms = weight.abs().sum(dim=(2, 3)).numpy().flatten()
        layer_flat_norms.append(flat_norms)
        if len(flat_norms) > max_kernels:
            max_kernels = len(flat_norms)
            
    # 建立對齊矩陣 (填充 NaN，Matplotlib 會渲染為 bad color)
    binary_matrix = np.full((max_kernels, num_selected), np.nan)
    magnitude_matrix = np.full((max_kernels, num_selected), np.nan)
    
    x_tick_labels = []
    for idx, (name, layer) in enumerate(selected_layers):
        flat_norms = layer_flat_norms[idx]
        L = len(flat_norms)
        
        # 計算二值剪枝
        binary_status = (flat_norms < threshold).astype(float)
        binary_matrix[:L, idx] = binary_status
        
        # 計算 Log 強度
        magnitude_matrix[:L, idx] = np.log10(flat_norms + 1e-10)
        
        # 計算剪枝率
        pruned_ratio = np.mean(binary_status) * 100.0
        paper_name = get_clean_name(name, backbone_name)
        x_tick_labels.append(f"{paper_name}\n({pruned_ratio:.1f}% pruned)")
        
        print(f"  Layer {name} | Total Kernels: {L} | Pruned: {pruned_ratio:.2f}%")
        
    # 使用論文標準格式美化畫布 (1 Row, 2 Columns)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 6.2), dpi=300)
    
    # 定義填充顏色：偏灰白色，既低調又與 inferno/viridis 區隔
    pad_color = "#e2e8f0"
    
    # --- 1. 左圖: Binary Pruning Heatmap (Inferno) ---
    cmap_binary = plt.colormaps["inferno"].copy()
    cmap_binary.set_bad(color=pad_color)
    
    im0 = ax0.imshow(binary_matrix, cmap=cmap_binary, aspect='auto', interpolation='nearest')
    ax0.set_title("Binary Kernel Pruning Profile\n(Bright Yellow = Fully Pruned Kernels)", fontsize=11, fontweight='bold', pad=10)
    
    # 建立客製化圖例
    legend_pruned = mpatches.Patch(color=cmap_binary(1.0), label='Pruned Connection')
    legend_active = mpatches.Patch(color=cmap_binary(0.0), label='Active Connection')
    legend_pad = mpatches.Patch(color=pad_color, label='Padded / Non-existent')
    ax0.legend(handles=[legend_pruned, legend_active, legend_pad], loc='upper right', fontsize=8, framealpha=0.9)
    
    # --- 2. 右圖: Continuous Log L1-Norm Strength (Viridis) ---
    cmap_mag = plt.colormaps["viridis"].copy()
    cmap_mag.set_bad(color=pad_color)
    
    im1 = ax1.imshow(magnitude_matrix, cmap=cmap_mag, aspect='auto', interpolation='nearest')
    ax1.set_title("Connection Strength Profile\n(Log10 L1-Norm Magnitude)", fontsize=11, fontweight='bold', pad=10)
    
    # 加上 Colorbar
    cbar = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("log10(Kernel L1-Norm)", fontsize=8)
    
    legend_pad_mag = mpatches.Patch(color=pad_color, label='Padded / Non-existent')
    ax1.legend(handles=[legend_pad_mag], loc='upper right', fontsize=8, framealpha=0.9)
    
    # --- 軸刻度與標籤美化 ---
    for ax in (ax0, ax1):
        ax.set_xticks(np.arange(num_selected))
        ax.set_xticklabels(x_tick_labels, fontsize=8, rotation=15, ha='right')
        ax.set_ylabel("Kernel Index (0 to Max)", fontsize=9)
        ax.set_xlabel("Layers (Input $\\rightarrow$ Output)", fontsize=9)
        ax.tick_params(labelsize=8)
        # 隱藏上方與右方的邊界線 (Spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    fig.suptitle(f"Global Kernel-Level Topology & Connection Strength Profile ({model_title})\n"
                 f"Backbone: {backbone_name} | Threshold: {threshold:.0e}", 
                 fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # 儲存高品質的 PNG 與 PDF (向量圖，方便直接插入 LaTeX 論文)
    save_name = f"global_kernel_profile_{model_title.replace(' ', '_').lower()}"
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
