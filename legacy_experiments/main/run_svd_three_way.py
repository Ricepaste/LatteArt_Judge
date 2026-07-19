# main/run_svd_three_way.py
import os
import argparse
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

# Fix random seed
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/app")

try:
    from src.processing.CIFAR100 import CIFAR100_Dataset
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

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

def load_hebbian_model(path, sparsity, device, label="Model"):
    from src.training.Hebbian_train import Hebbian_SSL_Trainer
    print(f"Initializing Hebbian ({label}) model with target sparsity: {sparsity}...")
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

def extract_features(model, dataloader, device):
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images, _, _ = batch
            images = images.to(device)
            feat = model(images)
            features.append(feat.cpu())
    return torch.cat(features, dim=0)

def main():
    parser = argparse.ArgumentParser(description="Singular Value Decomposition (SVD) Three-Way Comparison")
    parser.add_argument("--ours_path", type=str, default="main/runs/Hebbian_SSL_20260410-190945/last.pt",
                        help="Path to Hebbian (Ours) pre-trained checkpoint")
    parser.add_argument("--pos_hebb_path", type=str, required=True,
                        help="Path to Positive Hebbian (Ablation) pre-trained checkpoint")
    parser.add_argument("--rigl_path", type=str, default="main/runs/shuffleNet_v05_SimSiam__4/last.pt",
                        help="Path to RigL pre-trained checkpoint")
    parser.add_argument("--sparsity", type=float, default=0.99, help="Target sparsity (default: 0.99)")
    parser.add_argument("--out_dir", type=str, default="main/runs/visualizations",
                        help="Output directory for plots")
    args = parser.parse_args()

    # Resolve absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    
    def get_abs_path(p):
        if p and not os.path.isabs(p):
            return os.path.abspath(os.path.join(repo_dir, p))
        return p

    ours_path = get_abs_path(args.ours_path)
    pos_hebb_path = get_abs_path(args.pos_hebb_path)
    rigl_path = get_abs_path(args.rigl_path)
    os.makedirs(get_abs_path(args.out_dir), exist_ok=True)

    print(f"Ours Path: {ours_path}")
    print(f"Positive Hebbian Path: {pos_hebb_path}")
    print(f"RigL Path: {rigl_path}")

    # Check existence
    for name, path in [("Ours", ours_path), ("Positive Hebbian", pos_hebb_path), ("RigL", rigl_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} checkpoint not found at: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    ours_encoder = load_hebbian_model(ours_path, args.sparsity, device, label="Ours - Anti-Hebbian")
    pos_hebb_encoder = load_hebbian_model(pos_hebb_path, args.sparsity, device, label="Ablation - Positive Hebbian")
    rigl_encoder = load_rigl_model(rigl_path, device)

    # CIFAR-100 test set dataloader
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    print("Loading CIFAR-100 test dataset...")
    os.environ["INPUT_NOISE_STD"] = "0.0"
    test_dataset = CIFAR100_Dataset(split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # Extract features
    print("\n>>> Extracting features for Ours (Anti-Hebbian)...")
    ours_feats = extract_features(ours_encoder, test_loader, device)
    
    print("\n>>> Extracting features for Positive Hebbian (Ablation)...")
    pos_hebb_feats = extract_features(pos_hebb_encoder, test_loader, device)
    
    print("\n>>> Extracting features for RigL Baseline...")
    rigl_feats = extract_features(rigl_encoder, test_loader, device)

    # Normalize (standard for SVD spectrum comparison)
    ours_norm = torch.nn.functional.normalize(ours_feats, dim=1)
    pos_hebb_norm = torch.nn.functional.normalize(pos_hebb_feats, dim=1)
    rigl_norm = torch.nn.functional.normalize(rigl_feats, dim=1)

    # Compute SVD
    _, S_ours, _ = torch.linalg.svd(ours_norm, full_matrices=False)
    _, S_pos, _ = torch.linalg.svd(pos_hebb_norm, full_matrices=False)
    _, S_rigl, _ = torch.linalg.svd(rigl_norm, full_matrices=False)

    S_ours = S_ours.numpy() / S_ours.numpy().sum()
    S_pos = S_pos.numpy() / S_pos.numpy().sum()
    S_rigl = S_rigl.numpy() / S_rigl.numpy().sum()

    cum_ours = np.cumsum(S_ours)
    cum_pos = np.cumsum(S_pos)
    cum_rigl = np.cumsum(S_rigl)

    def dims_for_energy(cum_arr, threshold):
        return np.argmax(cum_arr >= threshold) + 1

    print("\n" + "=" * 60)
    print("📊 Three-Way SVD Spectrum Comparison Statistics")
    print("=" * 60)
    for label, s_arr, cum_arr in [("Ours (Anti-Hebbian)", S_ours, cum_ours), 
                                  ("Positive Hebbian (Ablation)", S_pos, cum_pos),
                                  ("RigL Baseline", S_rigl, cum_rigl)]:
        d90 = dims_for_energy(cum_arr, 0.90)
        d95 = dims_for_energy(cum_arr, 0.95)
        top10_energy = cum_arr[9] * 100
        print(f"[{label}]")
        print(f"  - Dimensions to explain 90% energy: {d90} / 512 ({(d90/512)*100:.1f}%)")
        print(f"  - Dimensions to explain 95% energy: {d95} / 512 ({(d95/512)*100:.1f}%)")
        print(f"  - Energy in top-10 singular values: {top10_energy:.2f}%")
        print("-" * 40)

    # Plot
    print("\n>>> Plotting SVD comparison...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'

    plt.figure(figsize=(6, 4.5))
    plt.plot(S_ours, label="Ours (Anti-Hebbian)", color="#1f77b4", linewidth=2.0)
    plt.plot(S_rigl, label="RigL Baseline", color="#d62728", linewidth=2.0, linestyle="--")
    plt.plot(S_pos, label="Positive Hebbian (Ablation)", color="#ff7f0e", linewidth=2.0, linestyle="-.")
    
    plt.yscale("log")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Singular Value")
    plt.legend(frameon=True)
    plt.tight_layout()

    out_png = os.path.join(get_abs_path(args.out_dir), "singular_value_spectrum_three_way.png")
    out_pdf = os.path.join(get_abs_path(args.out_dir), "singular_value_spectrum_three_way.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Plots successfully saved to: {out_png} and {out_pdf}")

if __name__ == "__main__":
    main()
