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
    backbone_lower = backbone.lower()
    parts = name.split('.')
    
    # 支援 "stageX" 或 "X" 這種前綴
    stage_num = None
    if parts:
        first_part = parts[0]
        if first_part.startswith("stage") and first_part[5:].isdigit():
            stage_num = int(first_part[5:])
        elif first_part.isdigit():
            val = int(first_part)
            if 2 <= val <= 4:
                stage_num = val

    if "resnet" in backbone_lower:
        if name in ["0", "conv1"] or (name.endswith(".conv1") and "." not in name[:-6]):
            return "Stem Conv"
            
        layer_num = None
        if parts:
            first_part = parts[0]
            if first_part.startswith("layer") and first_part[5:].isdigit():
                layer_num = int(first_part[5:])
            elif first_part.isdigit():
                val = int(first_part)
                if 4 <= val <= 7:
                    layer_num = val - 3  # 4->1, 5->2, 6->3, 7->4
                    
        if layer_num is not None and len(parts) >= 3:
            block_num = parts[1]
            sub_name = ".".join(parts[2:])
            if sub_name == "conv1":
                return f"L{layer_num}.{block_num} C1"
            elif sub_name == "conv2":
                return f"L{layer_num}.{block_num} C2"
            elif sub_name == "downsample.0":
                return f"L{layer_num}.{block_num} DS"
        return name
    else: # shufflenet
        if name in ["0.0", "conv1", "0"]:
            return "Stem Conv"
        if name in ["5.0", "conv5"]:
            return "Conv5"
            
        if stage_num is not None and len(parts) >= 3:
            block_num = parts[1]
            branch_name = parts[2]
            sub_idx = parts[3] if len(parts) > 3 else ""
            
            if branch_name == "branch1":
                if sub_idx == "0":
                    return f"S{stage_num}.{block_num} B1 DW"
                elif sub_idx == "2":
                    return f"S{stage_num}.{block_num} B1 PW"
                else:
                    return f"S{stage_num}.{block_num} B1.{sub_idx}"
            elif branch_name == "branch2":
                if sub_idx == "0":
                    return f"S{stage_num}.{block_num} B2 PW1"
                elif sub_idx == "3":
                    return f"S{stage_num}.{block_num} B2 DW"
                elif sub_idx == "5":
                    return f"S{stage_num}.{block_num} B2 PW2"
                else:
                    return f"S{stage_num}.{block_num} B2.{sub_idx}"
        return name

def analyze_and_plot_comparison(models_info, backbone_name, output_dir="runs/visualizations", threshold=1e-7, layer_type="spatial"):
    """
    遍歷每個 Model 的 Encoder 中的卷積層，計算每個卷積核 (Kernel) 是否被剪枝，並繪製論文等級的 2D 對比熱力圖。
    支援單個模型（1x2 子圖）與雙模型（2x2 子圖對比）排版。
    """
    os.makedirs(output_dir, exist_ok=True)
    num_models = len(models_info)
    if num_models == 0:
        return
        
    # 1. 論文標準字型格式設定
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif", "serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    
    # 2. 確定要選取的網路層名稱列表 (以第一個 model 為基準，並在後續進行比對)
    # 我們需要對 nn.Conv2d 模組進行去重，因為 Hebbian 封裝層會導致同一個物理 nn.Conv2d 被 named_modules() 訪問兩次
    model_conv_layers = []
    for encoder, model_title in models_info:
        conv_layers = []
        visited_modules = set()
        for name, module in encoder.named_modules():
            target_layer = module
            if hasattr(module, 'layer'):
                target_layer = module.layer
            if isinstance(target_layer, nn.Conv2d):
                if target_layer not in visited_modules:
                    visited_modules.add(target_layer)
                    conv_layers.append((name, target_layer))
        model_conv_layers.append((encoder, model_title, conv_layers))
        
    first_conv_layers = model_conv_layers[0][2]
    print(f"\n--- Conv2d Layers found in {backbone_name} ---")
    for n, l in first_conv_layers:
        print(f"  - {n} | Shape: {list(l.weight.shape)} | Kernel Size: {l.kernel_size}")
        
    print(f"\nSelecting layers for visualization (Mode: {layer_type})...")
    selected_indices = []
    
    if layer_type == "representative":
        # 1. 挑選 Stem
        stem_idx = None
        for i, (name, layer) in enumerate(first_conv_layers):
            if layer.kernel_size != 1 and layer.kernel_size != (1, 1):
                if name in ["0", "0.0", "conv1"] or name.endswith(".conv1") or (name.endswith(".0") and not ("." in name[:-2])):
                    stem_idx = i
                    break
        if stem_idx is not None:
            selected_indices.append(stem_idx)
            
        # 2. 挑選不同 Stage 的代表性 3x3 卷積層
        candidates = [
            # ResNet-18
            "4.0.conv1", "5.0.conv1", "6.0.conv1", "7.0.conv1",
            # ShuffleNet-V2 stage blocks (dw-conv at index 3 of branch2)
            "2.0.branch2.3", "3.0.branch2.3", "4.0.branch2.3",
            "stage2.0.branch2.3", "stage3.0.branch2.3", "stage4.0.branch2.3"
        ]
        for i, (name, layer) in enumerate(first_conv_layers):
            if any(c in name for c in candidates):
                if i not in selected_indices:
                    selected_indices.append(i)
                    
        if not selected_indices:
            selected_indices = [i for i, (n, l) in enumerate(first_conv_layers) if l.kernel_size != 1 and l.kernel_size != (1, 1)][:4]
            
    elif layer_type == "spatial":
        selected_indices = [i for i, (n, layer) in enumerate(first_conv_layers) if layer.kernel_size != 1 and layer.kernel_size != (1, 1)]
    elif layer_type == "all":
        selected_indices = list(range(len(first_conv_layers)))
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")
        
    num_selected = len(selected_indices)
    selected_layers_names = [first_conv_layers[i][0] for i in selected_indices]
    clean_selected_names = [get_clean_name(name, backbone_name) for name in selected_layers_names]
    print(f"Selected {num_selected} layers: {selected_layers_names}")
    print(f"Selected {num_selected} layers (Paper-friendly names): {clean_selected_names}")
    
    # 3. 收集每個模型的所有層資料與統計資訊
    model_data_list = []
    all_active_logs = []
    
    csv_rows = []
    
    for encoder, model_title, conv_layers in model_conv_layers:
        # 計算此模型全局權重稀疏度 (Global Weight Sparsity)
        total_params = 0
        zero_params = 0
        for name, layer in conv_layers:
            w = layer.weight.detach().cpu()
            total_params += w.numel()
            zero_params += (w.abs() < threshold).sum().item()
        
        global_w_sparsity = (zero_params / total_params) * 100.0 if total_params > 0 else 0.0
        print(f"\n>>> Analyzing {model_title} | Global Conv Sparsity: {global_w_sparsity:.2f}%")
        
        layer_binary_status = []
        layer_magnitude_log = []
        
        for idx in selected_indices:
            if idx >= len(conv_layers):
                continue
            name, layer = conv_layers[idx]
            weight = layer.weight.detach().cpu()
            flat_norms = weight.abs().sum(dim=(2, 3)).numpy().flatten()
            
            # 計算二值剪枝
            binary_status = (flat_norms < threshold).astype(float)
            layer_binary_status.append(binary_status)
            
            # 統計與印出
            pruned_ratio = np.mean(binary_status) * 100.0
            w_sparsity = (weight.abs() < threshold).float().mean().item() * 100.0
            print(f"  Layer {name} | Kernels: {len(flat_norms)} | Weight Sparsity: {w_sparsity:.2f}% | Kernel Pruned: {pruned_ratio:.2f}%")
            
            # 收集 CSV / Markdown 欄位
            clean_name = get_clean_name(name, backbone_name)
            csv_rows.append({
                "Model": model_title,
                "Layer_Index": idx,
                "Layer_Name": clean_name,
                "Original_Name": name,
                "Total_Kernels": len(flat_norms),
                "Weight_Sparsity_Pct": f"{w_sparsity:.2f}",
                "Kernel_Sparsity_Pct": f"{pruned_ratio:.2f}"
            })
            
        model_data_list.append({
            'title': model_title,
            'binary_status': layer_binary_status,
            'global_sparsity': global_w_sparsity
        })
        
    # 匯出 CSV 和 Markdown 表格
    import csv
    csv_fields = ["Model", "Layer_Index", "Layer_Name", "Original_Name", "Total_Kernels", "Weight_Sparsity_Pct", "Kernel_Sparsity_Pct"]
    csv_path = os.path.join(output_dir, f"kernel_pruning_sparsity_{layer_type}.csv")
    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n🎉 Sparsity report successfully exported to:")
        print(f"  - CSV Table:      {csv_path}")
    except Exception as e:
        print(f"Warning: Failed to write CSV file: {e}")

    md_path = os.path.join(output_dir, f"kernel_pruning_sparsity_{layer_type}.md")
    try:
        with open(md_path, mode="w", encoding="utf-8") as f:
            f.write(f"# Kernel Pruning and Sparsity Report ({layer_type.capitalize()} Layers)\n\n")
            f.write("| Model | Layer Index | Layer Name | Original Name | Total Kernels | Weight Sparsity (%) | Kernel Sparsity (%) |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for row in csv_rows:
                f.write(f"| {row['Model']} | {row['Layer_Index']} | {row['Layer_Name']} | {row['Original_Name']} | {row['Total_Kernels']} | {row['Weight_Sparsity_Pct']}% | {row['Kernel_Sparsity_Pct']}% |\n")
        print(f"  - Markdown Table: {md_path}")
    except Exception as e:
        print(f"Warning: Failed to write Markdown file: {e}")

            
    # 4. 準備繪圖
    from matplotlib.colors import ListedColormap
    cmap_binary = ListedColormap(["#0f172a", "#ffffff"])
    
    fig_width = max(8.0, num_selected * 0.5)
    fig_height = 3.5
    
    for m_idx, data in enumerate(model_data_list):
        model_title = data['title']
        layer_binary_status = data['binary_status']
        
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=300)
        fig.patch.set_facecolor('white')
        
        # 繪製 Binary Map
        for idx in range(num_selected):
            col_data = layer_binary_status[idx].reshape(-1, 1)
            im_bin = ax.imshow(col_data, cmap=cmap_binary, vmin=0, vmax=1, aspect='auto', interpolation='nearest',
                               extent=[idx - 0.5, idx + 0.5, 1, 0])
                                   
        # 軸刻度與標籤美化
        ax.set_facecolor('white')
        ax.set_xticks(np.arange(num_selected))
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.set_xlim(-0.5, num_selected - 0.5)
        ax.set_ylim(1, 0)
            
        # 設定 X 軸標籤
        ax.set_xlabel("Layers (Input $\\rightarrow$ Output)", fontsize=10)
            
        # 加上底部的 Legend
        legend_pruned = mpatches.Patch(facecolor="#ffffff", edgecolor="#cbd5e1", label='Pruned (Zero)')
        legend_active = mpatches.Patch(facecolor="#0f172a", label='Active')
        
        fig.legend(handles=[legend_pruned, legend_active], loc='lower center', ncol=2, fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.02))
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        
        save_name = f"kernel_profile_{layer_type}_{model_title.replace(' ', '_').lower()}"
        save_path_png = os.path.join(output_dir, f"{save_name}.png")
        save_path_pdf = os.path.join(output_dir, f"{save_name}.pdf")
        
        plt.savefig(save_path_png, bbox_inches='tight', dpi=300, facecolor='white')
        plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\n🎉 Scientific figure successfully saved to:")
        print(f"  - PNG (300 DPI): {save_path_png}")
        print(f"  - PDF (Vector):   {save_path_pdf}\n")


def analyze_topology_overlap(model1_info, model2_info, backbone_name, output_dir, threshold=1e-7, layer_type="spatial"):
    """
    計算兩個模型在每個對應卷積層 (Layer-wise) 的遮罩重合度 (Topology Overlap Ratio)。
    $$\text{Overlap}_l = \frac{\| \mathbf{M}_{1, l} \odot \mathbf{M}_{2, l} \|_0}{\| \mathbf{M}_{1, l} \|_0}$$
    """
    encoder1, name1 = model1_info
    encoder2, name2 = model2_info
    
    # 提取卷積層
    conv_layers1 = []
    visited_modules1 = set()
    for name, module in encoder1.named_modules():
        target_layer = module
        if hasattr(module, 'layer'):
            target_layer = module.layer
        if isinstance(target_layer, nn.Conv2d):
            if target_layer not in visited_modules1:
                visited_modules1.add(target_layer)
                conv_layers1.append((name, target_layer))
                
    conv_layers2 = []
    visited_modules2 = set()
    for name, module in encoder2.named_modules():
        target_layer = module
        if hasattr(module, 'layer'):
            target_layer = module.layer
        if isinstance(target_layer, nn.Conv2d):
            if target_layer not in visited_modules2:
                visited_modules2.add(target_layer)
                conv_layers2.append((name, target_layer))
                
    if len(conv_layers1) != len(conv_layers2):
        print(f"⚠️ Warning: Model conv layer count mismatch ({len(conv_layers1)} vs {len(conv_layers2)}). Cannot compute overlap.")
        return

    # 根據 layer_type 過濾網路層
    selected_indices = []
    if layer_type == "representative":
        # 1. 挑選 Stem
        stem_idx = None
        for i, (name, layer) in enumerate(conv_layers1):
            if layer.kernel_size != 1 and layer.kernel_size != (1, 1):
                if name in ["0", "0.0", "conv1"] or name.endswith(".conv1") or (name.endswith(".0") and not ("." in name[:-2])):
                    stem_idx = i
                    break
        if stem_idx is not None:
            selected_indices.append(stem_idx)
            
        # 2. 挑選不同 Stage 的代表性 3x3 卷積層
        candidates = [
            "4.0.conv1", "5.0.conv1", "6.0.conv1", "7.0.conv1",
            "2.0.branch2.3", "3.0.branch2.3", "4.0.branch2.3",
            "stage2.0.branch2.3", "stage3.0.branch2.3", "stage4.0.branch2.3"
        ]
        for i, (name, layer) in enumerate(conv_layers1):
            if any(c in name for c in candidates):
                if i not in selected_indices:
                    selected_indices.append(i)
                    
        if not selected_indices:
            selected_indices = [i for i, (n, l) in enumerate(conv_layers1) if l.kernel_size != 1 and l.kernel_size != (1, 1)][:4]
            
    elif layer_type == "spatial":
        selected_indices = [i for i, (n, layer) in enumerate(conv_layers1) if layer.kernel_size != 1 and layer.kernel_size != (1, 1)]
    elif layer_type == "all":
        selected_indices = list(range(len(conv_layers1)))
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")

    overlap_data = []
    print("\n" + "="*60)
    print(f"📊 Layer-wise Topology Overlap Analysis ({name1} vs {name2})")
    print("="*60)
    print(f"{'Layer Name':<20} | {'Clean Name':<15} | {'Overlap Ratio':<15} | {'IoU (Jaccard)':<15}")
    print("-" * 75)

    for idx in selected_indices:
        name_h, layer_h = conv_layers1[idx]
        name_r, layer_r = conv_layers2[idx]
        
        w_h = layer_h.weight.detach().cpu()
        w_r = layer_r.weight.detach().cpu()
        
        # 二值遮罩 (1 代表 active，0 代表 pruned)
        mask_h = (w_h.abs() >= threshold).float()
        mask_r = (w_r.abs() >= threshold).float()
        
        intersection = (mask_h * mask_r).sum().item()
        union = (mask_h + mask_r > 0).float().sum().item()
        
        denom_h = mask_h.sum().item()
        
        overlap_ratio = (intersection / denom_h) if denom_h > 0 else 0.0
        iou = (intersection / union) if union > 0 else 0.0
        
        clean_name = get_clean_name(name_h, backbone_name)
        
        print(f"{name_h:<20} | {clean_name:<15} | {overlap_ratio*100:6.2f}%         | {iou*100:6.2f}%")
        
        overlap_data.append({
            "Original_Name": name_h,
            "Clean_Name": clean_name,
            "Ours_Active": int(denom_h),
            "RigL_Active": int(mask_r.sum().item()),
            "Overlap_Count": int(intersection),
            "Overlap_Ratio": overlap_ratio,
            "IoU": iou
        })

    # 保存報告 CSV/Markdown
    csv_path = os.path.join(output_dir, "topology_overlap_report.csv")
    md_path = os.path.join(output_dir, "topology_overlap_report.md")
    
    import csv
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Original_Name", "Clean_Name", "Ours_Active", "RigL_Active", "Overlap_Count", "Overlap_Ratio", "IoU"])
        writer.writeheader()
        writer.writerows(overlap_data)
        
    with open(md_path, mode="w", encoding="utf-8") as f:
        f.write(f"# Topology Overlap Analysis ({name1} vs {name2})\n\n")
        f.write("| Original Name | Clean Name | Ours Active | RigL Active | Overlap Count | Overlap Ratio (%) | Jaccard IoU (%) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in overlap_data:
            f.write(f"| {row['Original_Name']} | {row['Clean_Name']} | {row['Ours_Active']} | {row['RigL_Active']} | {row['Overlap_Count']} | {row['Overlap_Ratio']*100:.2f}% | {row['IoU']*100:.2f}% |\n")

    print(f"\n✅ Reports saved to:")
    print(f"  - CSV: {csv_path}")
    print(f"  - MD:  {md_path}")

    # 繪製 Bar Chart (只保留對比，不加標題，LaTeX 友善)
    plt.figure(figsize=(10, 5))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    plt.rcParams['font.size'] = 11
    
    clean_names = [row["Clean_Name"] for row in overlap_data]
    overlap_ratios = [row["Overlap_Ratio"] * 100 for row in overlap_data]
    ious = [row["IoU"] * 100 for row in overlap_data]
    
    x = np.arange(len(clean_names))
    width = 0.35
    
    plt.bar(x - width/2, overlap_ratios, width, label="Overlap Ratio (Intersection / Ours)", color="#1f77b4")
    plt.bar(x + width/2, ious, width, label="Jaccard IoU (Intersection / Union)", color="#aec7e8", hatch="//")
    
    plt.xticks(x, clean_names, rotation=45, ha="right")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.legend(frameon=True)
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    fig_png = os.path.join(output_dir, "topology_overlap_comparison.png")
    fig_pdf = os.path.join(output_dir, "topology_overlap_comparison.pdf")
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Bar chart saved to:")
    print(f"  - PNG: {fig_png}")
    print(f"  - PDF: {fig_pdf}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Kernel-level Pruning Map using 2D Heatmaps")
    parser.add_argument("--hebbian_path", type=str, default="", help="Path to Hebbian model checkpoint (last.pt)")
    parser.add_argument("--rigl_path", type=str, default="", help="Path to RigL model checkpoint (last.pt)")
    parser.add_argument("--sparsity", type=float, default=0.99, help="Target global sparsity (default: 0.99)")
    parser.add_argument("--threshold", type=float, default=1e-7, help="Threshold below which a kernel is considered pruned (default: 1e-7)")
    parser.add_argument("--out_dir", type=str, default="runs/visualizations", help="Output directory for heatmaps")
    parser.add_argument("--layer_type", type=str, choices=["representative", "spatial", "all"], default="spatial",
                        help="Which layers to visualize: 'representative' (4 stage-representative layers), "
                             "'spatial' (all 3x3/spatial convolutional layers, default), "
                             "or 'all' (all Conv2d layers including 1x1 pointwise convs)")
    
    # Hebbian 設定相關參數
    parser.add_argument("--use_erk", action="store_true", default=True, help="Use ERK sparsity distribution")
    parser.add_argument("--no_erk", action="store_false", dest="use_erk", help="Disable ERK sparsity distribution")
    parser.add_argument("--protect_highway", action="store_true", default=True, help="Protect 1x1/highway layers")
    parser.add_argument("--no_protect_highway", action="store_false", dest="protect_highway", help="Do not protect 1x1/highway layers")
    
    args = parser.parse_args()
    
    # 每次執行前清理舊的檔案 (僅限 kernel pruning 相關的檔案，防止刪除 SVD 分析等其他實驗產物)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"🧹 Cleaning old kernel pruning visualization files in {out_dir}...")
    for f_name in os.listdir(out_dir):
        if (f_name.startswith("kernel_profile_") or 
            f_name.startswith("kernel_pruning_sparsity_") or 
            f_name.startswith("topology_overlap_")):
            try:
                os.remove(os.path.join(out_dir, f_name))
            except Exception as e:
                print(f"Warning: Failed to remove {f_name}: {e}")
    
    models_to_compare = []
    backbone_name = None
    
    # 1. 處理 Hebbian (Ours)
    hebbian_path = args.hebbian_path or os.environ.get("HEBBIAN_PATH", "")
    if hebbian_path and os.path.exists(hebbian_path):
        print("\n" + "="*50)
        print("🔍 Loading Hebbian (Ours) Model Structure...")
        print("="*50)
        try:
            encoder, _, b_name = load_sparse_model(
                "hebbian", 
                hebbian_path, 
                target_sparsity=args.sparsity,
                use_erk=args.use_erk,
                protect_highway=args.protect_highway
            )
            models_to_compare.append((encoder, "Ours"))
            backbone_name = b_name
        except Exception as e:
            traceback.print_exc()
            print(f"❌ Failed to load Hebbian model: {e}")
            
    # 2. 處理 RigL
    rigl_path = args.rigl_path or os.environ.get("RIGL_PATH", "")
    if rigl_path and os.path.exists(rigl_path):
        print("\n" + "="*50)
        print("🔍 Loading RigL Model Structure...")
        print("="*50)
        try:
            encoder, _, b_name = load_sparse_model(
                "rigl", 
                rigl_path, 
                target_sparsity=args.sparsity,
                use_erk=args.use_erk,
                protect_highway=args.protect_highway
            )
            models_to_compare.append((encoder, "RigL Baseline"))
            if backbone_name is None:
                backbone_name = b_name
        except Exception as e:
            traceback.print_exc()
            print(f"❌ Failed to load RigL model: {e}")
            
    # 3. 進行繪圖與重合度分析
    if len(models_to_compare) > 0:
        # 原本的 Heatmap 對比繪圖
        analyze_and_plot_comparison(
            models_to_compare,
            backbone_name,
            args.out_dir,
            threshold=args.threshold,
            layer_type=args.layer_type
        )
        
        # 進行網路拓撲層間重合度分析
        if len(models_to_compare) == 2:
            analyze_topology_overlap(
                models_to_compare[0],
                models_to_compare[1],
                backbone_name,
                args.out_dir,
                threshold=args.threshold,
                layer_type=args.layer_type
            )
    else:
        print("\n⚠️ No model path provided. Please specify model path via arguments or environment variables.")
        print("Usage Example:")
        print("  python visualize_kernel_pruning.py --hebbian_path runs/Hebbian_SSL_20260508-094600/last.pt --sparsity 0.96 --threshold 1e-6")
        print("  python visualize_kernel_pruning.py --rigl_path runs/shuffleNet_v05_SimSiam__4/last.pt --sparsity 0.99 --threshold 1e-6")
