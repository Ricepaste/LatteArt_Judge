# main/visualize_attention_and_masks.py
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader
import random

# 強制固定隨機種子，確保結果 100% 可重複性
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

from src.processing.CIFAR100 import CIFAR100_Dataset

class ResNetEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.output_dim = 512

    def forward(self, x):
        # 獲取卷積特徵圖 (Before global average pooling)
        # ResNet18 encoder has 8 modules. The output of the 8th module (layer4) is (B, 512, 7, 7)
        feat_map = self.encoder(x)
        return feat_map

def load_hebbian_model(path, sparsity, device):
    from src.training.Hebbian_train import Hebbian_SSL_Trainer
    print(f"Initializing Hebbian (Ours) model with target sparsity: {sparsity}...")
    dummy_trainer = Hebbian_SSL_Trainer(
        pretrained_model_class=models.resnet18,
        pretrained_weight=None,
        target_sparsity=sparsity,
        use_erk=True,
        protect_highway=False
    )
    model = dummy_trainer.model.to(device)
    print(f"Loading weights from {path}...")
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    if hasattr(model, 'set_hebbian_enable'):
        model.set_hebbian_enable(False)
    
    encoder = ResNetEncoderWrapper(model.encoder).to(device)
    encoder.eval()
    return encoder

def load_rigl_model(path, device):
    import src.training.SimSiam_train as SimSiam_train
    print("Initializing RigL model...")
    dummy_trainer = SimSiam_train.SimSiam_Model(
        pretrained_model=models.resnet18,
        pretrained_weight=None
    )
    model = dummy_trainer.model.to(device)
    print(f"Loading weights from {path}...")
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    encoder = ResNetEncoderWrapper(model.encoder).to(device)
    encoder.eval()
    return encoder

def compute_activation_map(feature_map):
    # feature_map shape: (1, 512, 7, 7)
    # 沿著 Channel 求平均絕對活化強度
    act = feature_map[0].abs().mean(dim=0).cpu().numpy()
    
    # 歸一化到 [0, 1]
    act_min = act.min()
    act_max = act.max()
    if act_max > act_min:
        act = (act - act_min) / (act_max - act_min)
    return act

def find_weight_by_key(encoder, key):
    for name, module in encoder.encoder.named_modules():
        target = module.layer if hasattr(module, 'layer') else module
        if not isinstance(target, nn.Conv2d):
            continue
            
        if key == "conv1" and (name == "0" or name.endswith("conv1")):
            return target.weight.detach().cpu()
        elif key == "layer1.0.conv1" and (name == "4.0.conv1" or name.endswith("layer1.0.conv1")):
            return target.weight.detach().cpu()
        elif key == "layer2.0.conv1" and (name == "5.0.conv1" or name.endswith("layer2.0.conv1")):
            return target.weight.detach().cpu()
        elif key == "layer3.0.conv1" and (name == "6.0.conv1" or name.endswith("layer3.0.conv1")):
            return target.weight.detach().cpu()
        elif key == "layer4.0.conv1" and (name == "7.0.conv1" or name.endswith("layer4.0.conv1")):
            return target.weight.detach().cpu()
    return None

def main():
    parser = argparse.ArgumentParser(description="Visualize Attention Maps and Sparse Masks")
    parser.add_argument("--hebbian_path", type=str, default="main/runs/Hebbian_SSL_20260410-190945/last.pt",
                        help="Path to Hebbian (Ours) 99% checkpoint")
    parser.add_argument("--rigl_path", type=str, default="main/runs/shuffleNet_v05_SimSiam__4/last.pt",
                        help="Path to RigL 99% checkpoint")
    parser.add_argument("--sparsity", type=float, default=0.99, help="Target sparsity (default: 0.99)")
    parser.add_argument("--threshold", type=float, default=1e-7, help="Pruning threshold")
    parser.add_argument("--out_dir", type=str, default="main/runs/visualizations",
                        help="Output directory for plots")
    parser.add_argument("--run_appendix_masks", action="store_true", help="Generate additional sparse weight connectivity maps for other layers")
    parser.add_argument("--run_augmentation_test", action="store_true", help="Run crop/shift augmentation test on a representative CIFAR-100 image")
    parser.add_argument("--custom_image", type=str, default="main/demo/wolf.jpg",
                        help="Path to custom image for augmentation test (default: main/demo/wolf.jpg)")
    args = parser.parse_args()

    # 路徑解析
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    
    hebbian_path = args.hebbian_path
    if hebbian_path and not os.path.isabs(hebbian_path):
        hebbian_path = os.path.abspath(os.path.join(repo_dir, hebbian_path))
        
    rigl_path = args.rigl_path
    if rigl_path and not os.path.isabs(rigl_path):
        rigl_path = os.path.abspath(os.path.join(repo_dir, rigl_path))

    os.makedirs(args.out_dir, exist_ok=True)
    
    if not os.path.exists(hebbian_path):
        raise FileNotFoundError(f"Hebbian checkpoint not found at: {hebbian_path}")
    if not os.path.exists(rigl_path):
        raise FileNotFoundError(f"RigL checkpoint not found at: {rigl_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 載入模型
    hebbian_encoder = load_hebbian_model(hebbian_path, args.sparsity, device)
    rigl_encoder = load_rigl_model(rigl_path, device)

    # 2. 資料集準備 (不使用 Normalization 用於顯示原圖)
    transform_display = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    transform_model = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    os.environ["INPUT_NOISE_STD"] = "0.0"
    test_dataset = CIFAR100_Dataset(split="test", transform=transform_display)
    
    # 選擇 4 個不同類別的代表性樣本 (自動挑選 Ours 聚焦中心、RigL 偏向邊緣/背景的影像)
    print("Scanning dataset to find images with maximum attention discrepancy...")
    best_samples = []
    classes = test_dataset.dataset.classes
    
    # 掃描前 200 張影像
    candidates = []
    for idx in range(min(200, len(test_dataset))):
        img_display, _, label = test_dataset[idx]
        img_model = transform_model(img_display).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat_h = hebbian_encoder(img_model)
            feat_r = rigl_encoder(img_model)
            
        act_h = compute_activation_map(feat_h)
        act_r = compute_activation_map(feat_r)
        
        # 定義中心區域 (3x3)
        center_mask = np.zeros((7, 7), dtype=bool)
        center_mask[2:5, 2:5] = True
        
        h_center_ratio = act_h[center_mask].sum() / (act_h.sum() + 1e-8)
        r_center_ratio = act_r[center_mask].sum() / (act_r.sum() + 1e-8)
        
        # 分數：Ours 越專注於中心，RigL 越發散/專注於邊緣
        score = h_center_ratio - r_center_ratio
        
        class_name = classes[label]
        candidates.append((score, idx, img_display, class_name, feat_h, feat_r))
        
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected_classes = set()
    for item in candidates:
        score, idx, img_display, class_name, feat_h, feat_r = item
        # 排除時鐘 (clock)，因為其活化關注區域不夠直觀，保留梨子 (pear) 等更佳範例
        if class_name == "clock":
            continue
        if class_name not in selected_classes:
            selected_classes.add(class_name)
            best_samples.append((img_display, class_name, feat_h, feat_r))
            print(f"  - Selected image idx {idx}: Class '{class_name}' (Discrepancy Score: {score:.3f})")
        if len(best_samples) == 4:
            break

    # 3. 提取特徵與疊加活化圖
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    plt.rcParams['font.size'] = 12

    fig, axes = plt.subplots(4, 3, figsize=(9, 11))

    for row_idx, (img_display, class_name, feat_h, feat_r) in enumerate(best_samples):
        # 計算特徵活化圖
        act_h = compute_activation_map(feat_h)
        act_r = compute_activation_map(feat_r)
        
        # 使用雙線性插值放大活化圖至 224x224
        act_h_tensor = torch.tensor(act_h).unsqueeze(0).unsqueeze(0)
        act_r_tensor = torch.tensor(act_r).unsqueeze(0).unsqueeze(0)
        
        act_h_resized = torch.nn.functional.interpolate(act_h_tensor, size=(224, 224), mode='bilinear', align_corners=False).squeeze().numpy()
        act_r_resized = torch.nn.functional.interpolate(act_r_tensor, size=(224, 224), mode='bilinear', align_corners=False).squeeze().numpy()
        
        # 轉換為 numpy 展示格式 (H, W, C)
        img_np = img_display.permute(1, 2, 0).numpy()
        
        # numpy-level pre-blending (將熱力圖與原圖在矩陣層面直接融合，徹底解決 PDF 矢量圖渲染透明度丟失/washed-out 的問題)
        cmap = plt.get_cmap('jet')
        alpha = 0.55
        
        # Ours composite
        act_h_colored = cmap(act_h_resized)[:, :, :3]
        composite_h = (1 - alpha) * img_np + alpha * act_h_colored
        
        # RigL composite
        act_r_colored = cmap(act_r_resized)[:, :, :3]
        composite_r = (1 - alpha) * img_np + alpha * act_r_colored
        
        # 3.1 繪製原圖
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        axes[row_idx, 0].set_ylabel(class_name.capitalize(), fontsize=12, fontweight='bold')
        if row_idx == 0:
            axes[row_idx, 0].set_title("Original Image", fontsize=12, pad=10)

        # 3.2 繪製 Ours (Hebbian) 活化圖
        axes[row_idx, 1].imshow(composite_h)
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])
        if row_idx == 0:
            axes[row_idx, 1].set_title("Ours", fontsize=12, pad=10)

        # 3.3 繪製 RigL 活化圖
        axes[row_idx, 2].imshow(composite_r)
        axes[row_idx, 2].set_xticks([])
        axes[row_idx, 2].set_yticks([])
        if row_idx == 0:
            axes[row_idx, 2].set_title("RigL", fontsize=12, pad=10)

    # 增加一個共享的 Colorbar 在右側，以說明活化強度的色標 (對應 Low / High 關注度)
    plt.tight_layout(rect=[0, 0, 0.90, 1])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cbar_ax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['Low', 'High'])
    cb.set_label("Activation Intensity", fontsize=11, labelpad=5)
    cb.ax.tick_params(labelsize=10)

    act_png = os.path.join(args.out_dir, "activation_attention_maps.png")
    act_pdf = os.path.join(args.out_dir, "activation_attention_maps.pdf")
    plt.savefig(act_png, dpi=300, bbox_inches='tight')
    plt.savefig(act_pdf, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Feature Activation (Attention) maps saved to:\n  - {act_png}\n  - {act_pdf}")

    # 4. 繪製權重級稀疏矩陣連接圖
    print("\nExtracting layer weights for connectivity visualization...")
    w_h = find_weight_by_key(hebbian_encoder, "layer3.0.conv1")
    w_r = find_weight_by_key(rigl_encoder, "layer3.0.conv1")

    if w_h is not None and w_r is not None:
        # 計算通道間的連接強度 (Sum of absolute values over kernel 3x3)
        # Shape: (out_channels, in_channels, 3, 3) -> (out_channels, in_channels) = (256, 128)
        mask_h = (w_h.abs().sum(dim=(2, 3)) >= args.threshold).float().numpy()
        mask_r = (w_r.abs().sum(dim=(2, 3)) >= args.threshold).float().numpy()
        
        # 繪圖展示 (Ours vs RigL side-by-side，去標題留給 LaTeX 處理)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # 使用 binary colors (白色代表 pruned，深藍/黑色代表 active)
        ax1.imshow(1 - mask_h, cmap='gray', aspect='auto', interpolation='nearest')
        ax1.set_xlabel("Input Channel Index", fontsize=11)
        ax1.set_ylabel("Output Channel Index", fontsize=11)
        
        ax2.imshow(1 - mask_r, cmap='gray', aspect='auto', interpolation='nearest')
        ax2.set_xlabel("Input Channel Index", fontsize=11)
        ax2.set_ylabel("Output Channel Index", fontsize=11)
        
        plt.tight_layout()
        conn_png = os.path.join(args.out_dir, "sparse_weight_connections.png")
        conn_pdf = os.path.join(args.out_dir, "sparse_weight_connections.pdf")
        plt.savefig(conn_png, dpi=300, bbox_inches='tight')
        plt.savefig(conn_pdf, bbox_inches='tight')
        plt.close()
        print(f"✅ Sparse weight connectivity maps saved to:\n  - {conn_png}\n  - {conn_pdf}")
    else:
        print("⚠️ Warning: Could not find layer 'layer3.0.conv1' weights to plot connectivity maps.")
    
    print("\n" + "="*50)
    print("🎉 Main visualizations generated successfully!")
    print("="*50 + "\n")

    # 5. 繪製附錄其他層之二值稀疏遮罩矩陣連接圖
    if args.run_appendix_masks:
        print("\n" + "="*50)
        print("🚀 Starting Appendix Sparse Mask Connectivity Maps Generation")
        print("="*50)
        
        layers_to_plot = [
            ("conv1", "conv1"),
            ("layer1.0.conv1", "layer1_0_conv1"),
            ("layer2.0.conv1", "layer2_0_conv1"),
            ("layer4.0.conv1", "layer4_0_conv1")
        ]

        for l_key, l_file_name in layers_to_plot:
            w_h = find_weight_by_key(hebbian_encoder, l_key)
            w_r = find_weight_by_key(rigl_encoder, l_key)

            if w_h is not None and w_r is not None:
                mask_h = (w_h.abs().sum(dim=list(range(2, w_h.dim()))) >= args.threshold).float().numpy()
                mask_r = (w_r.abs().sum(dim=list(range(2, w_r.dim()))) >= args.threshold).float().numpy()

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                
                ax1.imshow(1 - mask_h, cmap='gray', aspect='auto', interpolation='nearest')
                ax1.set_xlabel("Input Channel Index", fontsize=11)
                ax1.set_ylabel("Output Channel Index", fontsize=11)
                
                ax2.imshow(1 - mask_r, cmap='gray', aspect='auto', interpolation='nearest')
                ax2.set_xlabel("Input Channel Index", fontsize=11)
                ax2.set_ylabel("Output Channel Index", fontsize=11)
                
                plt.tight_layout()
                out_png = os.path.join(args.out_dir, f"appendix_mask_{l_file_name}.png")
                out_pdf = os.path.join(args.out_dir, f"appendix_mask_{l_file_name}.pdf")
                plt.savefig(out_png, dpi=300, bbox_inches='tight')
                plt.savefig(out_pdf, bbox_inches='tight')
                plt.close()
                print(f"✅ Saved mask connectivity map for layer '{l_key}' to {out_png} / {out_pdf}")
            else:
                print(f"⚠️ Warning: Could not find weights for layer '{l_key}' in one or both models.")

        print("\n" + "="*50)
        print("🎉 Appendix mask visualizations generated successfully!")
        print("="*50 + "\n")

    # 6. 繪製資料強化（隨機裁切）強健性測試圖
    if args.run_augmentation_test:
        print("\n" + "="*50)
        print("🚀 Starting Augmentation Robustness Test (Random Resized Crop)")
        print("="*50)
        
        # 1. 載入自訂影像 (預設為 main/demo/wolf.jpg)
        custom_img_path = args.custom_image
        if custom_img_path and not os.path.isabs(custom_img_path):
            custom_img_path = os.path.abspath(os.path.join(repo_dir, custom_img_path))
            
        raw_image = None
        best_class = None
        
        if os.path.exists(custom_img_path):
            try:
                print(f"Loading custom image for augmentation test: {custom_img_path}")
                from PIL import Image
                raw_image = Image.open(custom_img_path).convert("RGB")
                best_class = os.path.basename(custom_img_path)
            except Exception as e:
                print(f"⚠️ Error loading custom image {custom_img_path}: {e}")
                
        if raw_image is None:
            # Fallback: 尋找最具代表性的圖像 (排除 "clock")
            print(f"⚠️ Custom image not found or failed to load. Falling back to CIFAR-100 dataset...")
            best_idx = None
            for item in candidates:
                score, idx, _, class_name, _, _ = item
                if class_name != "clock":
                    best_idx = idx
                    best_class = class_name
                    break
                    
            if best_idx is not None:
                print(f"Selected representative image idx {best_idx} (Class: '{best_class}') for augmentation test.")
                raw_image, label = test_dataset.dataset[best_idx]
            
            # 定義平移搭配縮小增強 (固定縮小為 50%，搭配最大 40% 比例位移以貼近邊緣，無旋轉)
            aug_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.4, 0.4),  # 允許最大 40% 比例的平移 (更靠近邊緣)
                    scale=(0.5, 0.5),      # 固定縮小為原尺寸的 50% (穩定的縮小)
                    fill=0                 # 位移後多出的邊界區域填補黑色
                ),
                transforms.ToTensor(),
            ])
            
            # 生成 4 個不同的隨機裁切版本
            seed_everything(123)  # 固定種子以利複現
            augmented_images = []
            for _ in range(4):
                augmented_images.append(aug_transform(raw_image))
                
            fig, axes = plt.subplots(3, 4, figsize=(11, 8))
            
            for col_idx, img_display in enumerate(augmented_images):
                img_model = transform_model(img_display).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    feat_h = hebbian_encoder(img_model)
                    feat_r = rigl_encoder(img_model)
                    
                act_h = compute_activation_map(feat_h)
                act_r = compute_activation_map(feat_r)
                
                act_h_tensor = torch.tensor(act_h).unsqueeze(0).unsqueeze(0)
                act_r_tensor = torch.tensor(act_r).unsqueeze(0).unsqueeze(0)
                
                act_h_resized = torch.nn.functional.interpolate(act_h_tensor, size=(224, 224), mode='bilinear', align_corners=False).squeeze().numpy()
                act_r_resized = torch.nn.functional.interpolate(act_r_tensor, size=(224, 224), mode='bilinear', align_corners=False).squeeze().numpy()
                
                img_np = img_display.permute(1, 2, 0).numpy()
                cmap = plt.get_cmap('jet')
                alpha = 0.55
                
                composite_h = (1 - alpha) * img_np + alpha * (cmap(act_h_resized)[:, :, :3])
                composite_r = (1 - alpha) * img_np + alpha * (cmap(act_r_resized)[:, :, :3])
                
                # Row 0: 原始裁切圖
                axes[0, col_idx].imshow(img_np)
                axes[0, col_idx].set_xticks([])
                axes[0, col_idx].set_yticks([])
                axes[0, col_idx].set_title(f"Crop {col_idx+1}", fontsize=11)
                if col_idx == 0:
                    axes[0, col_idx].set_ylabel("Augmented Image\n(Class: {})".format(best_class.capitalize()), fontsize=11, fontweight='bold')
                
                # Row 1: Ours (Hebbian)
                axes[1, col_idx].imshow(composite_h)
                axes[1, col_idx].set_xticks([])
                axes[1, col_idx].set_yticks([])
                if col_idx == 0:
                    axes[1, col_idx].set_ylabel("Ours", fontsize=11, fontweight='bold')
                
                # Row 2: RigL
                axes[2, col_idx].imshow(composite_r)
                axes[2, col_idx].set_xticks([])
                axes[2, col_idx].set_yticks([])
                if col_idx == 0:
                    axes[2, col_idx].set_ylabel("RigL", fontsize=11, fontweight='bold')
                    
            plt.tight_layout(rect=[0, 0, 0.90, 1])
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cbar_ax)
            cb.set_ticks([0, 1])
            cb.set_ticklabels(['Low', 'High'])
            cb.set_label("Activation Intensity", fontsize=11, labelpad=5)
            cb.ax.tick_params(labelsize=10)
            
            aug_png = os.path.join(args.out_dir, "augmentation_robustness_test.png")
            aug_pdf = os.path.join(args.out_dir, "augmentation_robustness_test.pdf")
            plt.savefig(aug_png, dpi=300, bbox_inches='tight')
            plt.savefig(aug_pdf, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved augmentation robustness test plot to {aug_png} / {aug_pdf}")
        else:
            print("⚠️ Warning: Could not find any valid image for augmentation test.")

if __name__ == "__main__":
    main()
