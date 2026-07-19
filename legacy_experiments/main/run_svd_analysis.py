# main/run_svd_analysis.py
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
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

# 3. 抽取 Encoder 供特徵提取使用 (對齊 evaluate_model.py)
class ResNetEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.output_dim = 512

    def forward(self, x):
        x = self.encoder(x)
        if x.dim() == 4:
            x = x.mean([2, 3])
        x = x.view(x.size(0), -1)
        return x

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
    # 關閉 hebbian 生長與更新
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

def extract_features(model, dataloader, device):
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # CIFAR100_Dataset returns (image1, image2, label)
            images, _, _ = batch
            images = images.to(device)
            feat = model(images) # Shape: (B, 512)
            features.append(feat.cpu())
    return torch.cat(features, dim=0)

def main():
    parser = argparse.ArgumentParser(description="Singular Value Decomposition (SVD) Spectrum Analysis")
    parser.add_argument("--hebbian_path", type=str, default="main/runs/Hebbian_SSL_20260410-190945/last.pt",
                        help="Path to Hebbian (Ours) pre-trained checkpoint")
    parser.add_argument("--rigl_path", type=str, default="main/runs/shuffleNet_v05_SimSiam__4/last.pt",
                        help="Path to RigL pre-trained checkpoint")
    parser.add_argument("--sparsity", type=float, default=0.99, help="Target sparsity for Hebbian model (default: 0.99)")
    parser.add_argument("--out_dir", type=str, default="main/runs/visualizations",
                        help="Output directory for plots and results")
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
    
    # 檢查路徑
    if not os.path.exists(hebbian_path):
        raise FileNotFoundError(f"Hebbian checkpoint not found at: {hebbian_path}")
    if not os.path.exists(rigl_path):
        raise FileNotFoundError(f"RigL checkpoint not found at: {rigl_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 載入模型
    hebbian_encoder = load_hebbian_model(hebbian_path, args.sparsity, device)
    rigl_encoder = load_rigl_model(rigl_path, device)

    # 2. 資料集準備 (與 evaluate_model.py 的測試集完全一致)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    print("Loading CIFAR-100 test dataset...")
    # 設置環境變數以防止噪聲干擾
    os.environ["INPUT_NOISE_STD"] = "0.0"
    test_dataset = CIFAR100_Dataset(split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    print(f"Total test samples: {len(test_dataset)}")

    # 3. 提取特徵矩陣
    print("\n>>> Extracting features for Ours (Hebbian)...")
    hebbian_feats_raw = extract_features(hebbian_encoder, test_loader, device)
    
    print("\n>>> Extracting features for RigL Baseline...")
    rigl_feats_raw = extract_features(rigl_encoder, test_loader, device)

    print(f"\nHebbian raw features shape: {hebbian_feats_raw.shape}")
    print(f"RigL raw features shape: {rigl_feats_raw.shape}")

    # 同時分析原始特徵與 L2 歸一化特徵
    hebbian_feats_l2 = torch.nn.functional.normalize(hebbian_feats_raw, dim=1)
    rigl_feats_l2 = torch.nn.functional.normalize(rigl_feats_raw, dim=1)

    features_dict = {
        "Raw Features": (hebbian_feats_raw, rigl_feats_raw),
        "L2-Normalized Features": (hebbian_feats_l2, rigl_feats_l2)
    }

    # 4. 進行 SVD 分解與能量集中度計算
    svd_results = {}
    print("\n" + "="*50)
    print("📊 Computing SVD and Energy Distribution Statistics")
    print("="*50)

    for mode_name, (h_feat, r_feat) in features_dict.items():
        # Hebbian SVD
        _, S_h, _ = torch.linalg.svd(h_feat, full_matrices=False)
        # RigL SVD
        _, S_r, _ = torch.linalg.svd(r_feat, full_matrices=False)

        # 轉為 numpy array
        S_h_np = S_h.numpy()
        S_r_np = S_r.numpy()

        # 歸一化
        S_h_norm = S_h_np / S_h_np.sum()
        S_r_norm = S_r_np / S_r_np.sum()

        # 累積能量
        cum_h = np.cumsum(S_h_norm)
        cum_r = np.cumsum(S_r_norm)

        # 計算解釋能量百分比所需的維度
        def dims_for_energy(cum_arr, threshold):
            return np.argmax(cum_arr >= threshold) + 1

        dims_h_90 = dims_for_energy(cum_h, 0.90)
        dims_r_90 = dims_for_energy(cum_r, 0.90)
        dims_h_95 = dims_for_energy(cum_h, 0.95)
        dims_r_95 = dims_for_energy(cum_r, 0.95)

        # 計算前 10 維/特徵占的總能量
        energy_top10_h = cum_h[9] * 100
        energy_top10_r = cum_r[9] * 100

        print(f"\n[{mode_name}]")
        print(f"  > Ours (Hebbian):")
        print(f"    - Dimensions to explain 90% energy: {dims_h_90} / 512 ({(dims_h_90/512)*100:.1f}%)")
        print(f"    - Dimensions to explain 95% energy: {dims_h_95} / 512 ({(dims_h_95/512)*100:.1f}%)")
        print(f"    - Energy in top-10 singular values: {energy_top10_h:.2f}%")
        print(f"  > RigL Baseline:")
        print(f"    - Dimensions to explain 90% energy: {dims_r_90} / 512 ({(dims_r_90/512)*100:.1f}%)")
        print(f"    - Dimensions to explain 95% energy: {dims_r_95} / 512 ({(dims_r_95/512)*100:.1f}%)")
        print(f"    - Energy in top-10 singular values: {energy_top10_r:.2f}%")
        
        # 判斷崩塌傾向
        collapse_ratio = dims_h_90 / dims_r_90
        print(f"  > Ratio of active dimensions (Ours / RigL): {collapse_ratio:.2f}x")
        if dims_r_90 < dims_h_90 * 0.5:
            print("  ⚠️ Warning: RigL Baseline shows severe Dimensional Collapse compared to Hebbian (Ours)!")
        else:
            print("  💡 Both models retain distributed representations, but Ours is more uniform.")

        svd_results[mode_name] = {
            "S_h_norm": S_h_norm,
            "S_r_norm": S_r_norm,
            "cum_h": cum_h,
            "cum_r": cum_r
        }

    # 5. 繪製學術對比圖表 (僅保留 Log 尺度且不加標題)
    print("\n>>> Generating publication-ready log-scale plots (without titles)...")
    # 設定字體與風格
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'

    # 5.1 繪製 Raw Features Log Scale
    raw_res = svd_results["Raw Features"]
    plt.figure(figsize=(6, 4.5))
    plt.plot(raw_res["S_h_norm"], label="Ours (GF-DST / Hebbian)", color="#1f77b4", linewidth=2.0)
    plt.plot(raw_res["S_r_norm"], label="RigL Baseline", color="#d62728", linewidth=2.0, linestyle="--")
    plt.yscale("log")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Singular Value")
    plt.legend(frameon=True)
    plt.tight_layout()
    
    raw_png = os.path.join(args.out_dir, "singular_value_spectrum_raw_log.png")
    raw_pdf = os.path.join(args.out_dir, "singular_value_spectrum_raw_log.pdf")
    plt.savefig(raw_png, dpi=300, bbox_inches='tight')
    plt.savefig(raw_pdf, bbox_inches='tight')
    plt.close()

    # 5.2 繪製 L2-Normalized Features Log Scale
    l2_res = svd_results["L2-Normalized Features"]
    plt.figure(figsize=(6, 4.5))
    plt.plot(l2_res["S_h_norm"], label="Ours (GF-DST / Hebbian)", color="#1f77b4", linewidth=2.0)
    plt.plot(l2_res["S_r_norm"], label="RigL Baseline", color="#d62728", linewidth=2.0, linestyle="--")
    plt.yscale("log")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Singular Value")
    plt.legend(frameon=True)
    plt.tight_layout()
    
    l2_png = os.path.join(args.out_dir, "singular_value_spectrum_l2_log.png")
    l2_pdf = os.path.join(args.out_dir, "singular_value_spectrum_l2_log.pdf")
    plt.savefig(l2_png, dpi=300, bbox_inches='tight')
    plt.savefig(l2_pdf, bbox_inches='tight')
    
    # 同步儲存至預設檔名，便於外部系統存取
    default_png = os.path.join(args.out_dir, "singular_value_spectrum.png")
    default_pdf = os.path.join(args.out_dir, "singular_value_spectrum.pdf")
    plt.savefig(default_png, dpi=300, bbox_inches='tight')
    plt.savefig(default_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Plots successfully saved to {args.out_dir}:")
    print(f"  - Raw features log plot: {raw_png} / {raw_pdf}")
    print(f"  - L2-normalized features log plot: {l2_png} / {l2_pdf}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
