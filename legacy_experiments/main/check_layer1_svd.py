import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from torchvision import models, transforms
from torch.utils.data import DataLoader
from src.training.Hebbian_train import Hebbian_SSL_Trainer

# Setup paths
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MAIN_DIR)

def resolve_path(relative_path):
    search_dirs = [
        MAIN_DIR,
        REPO_DIR,
        "/app/main",
        "/app"
    ]
    for d in search_dirs:
        full_path = os.path.join(d, relative_path)
        if os.path.exists(full_path):
            return full_path
    return None

class ResNetLayer1Wrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.output_dim = 64

    def forward(self, x):
        # self.encoder contains:
        # 0: conv1, 1: bn1, 2: relu, 3: maxpool, 4: layer1
        # Slicing up to 5 runs through layer1, returning shape (B, 64, H, W)
        return self.encoder[:5](x)

def get_layer1_encoder(checkpoint_path, use_no_bn=False, target_sparsity=0.99):
    if use_no_bn:
        def resnet18_no_bn(weights=None):
            model = models.resnet18(weights=weights)
            def replace_bn(m):
                for name, child in m.named_children():
                    if isinstance(child, nn.BatchNorm2d):
                        setattr(m, name, nn.Identity())
                    else:
                        replace_bn(child)
            replace_bn(model)
            return model
        model_class = resnet18_no_bn
    else:
        model_class = models.resnet18

    trainer = Hebbian_SSL_Trainer(
        pretrained_model_class=model_class,
        pretrained_weight=None,
        target_sparsity=target_sparsity,
        use_erk=True,
        protect_highway=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = trainer.model.to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    if hasattr(model, 'set_hebbian_enable'):
        model.set_hebbian_enable(False)
    
    wrapper = ResNetLayer1Wrapper(model.encoder).to(device)
    wrapper.eval()
    return wrapper

def extract_layer1_features(encoder, loader, num_batches=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_features = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            images, _, _ = batch
            images = images.to(device)
            feat = encoder(images)  # (B, 64, H, W)
            B, C, H, W = feat.shape
            # Reshape to treat each spatial position as a sample
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)
            all_features.append(feat_flat.cpu())
    return torch.cat(all_features, dim=0)

def analyze_channel_correlation(feats):
    N, C = feats.shape
    # Center features
    feats_centered = feats - feats.mean(dim=0, keepdim=True)
    # Compute std dev
    feats_std = torch.sqrt((feats_centered**2).mean(dim=0, keepdim=True) + 1e-8)
    # Normalize features
    feats_norm = feats_centered / feats_std
    
    # Correlation matrix
    corr = torch.matmul(feats_norm.t(), feats_norm) / N
    
    abs_corr = torch.abs(corr)
    diagonal_mask = torch.eye(C, device=abs_corr.device).bool()
    off_diag_abs_corr = abs_corr[~diagonal_mask]
    
    mean_abs_corr = off_diag_abs_corr.mean().item()
    max_abs_corr = off_diag_abs_corr.max().item()
    
    # SVD of correlation matrix
    _, S, _ = torch.linalg.svd(corr)
    S = S / S.sum()
    svd_entropy = -torch.sum(S * torch.log(S + 1e-10)).item()
    
    cum_energy = torch.cumsum(S, dim=0)
    dim_90 = (cum_energy >= 0.90).nonzero()[0].item() + 1
    dim_95 = (cum_energy >= 0.95).nonzero()[0].item() + 1
    
    return {
        "mean_abs_corr": mean_abs_corr,
        "max_abs_corr": max_abs_corr,
        "svd_entropy": svd_entropy,
        "dim_90": dim_90,
        "dim_95": dim_95
    }

def print_result_table(name, stats):
    print(f"\n[{name}]")
    print(f"  - Mean Absolute Off-Diagonal Channel Correlation: {stats['mean_abs_corr']:.4f}")
    print(f"  - Max Absolute Channel Correlation:              {stats['max_abs_corr']:.4f}")
    print(f"  - SVD Entropy of Channel Correlation Matrix:     {stats['svd_entropy']:.4f}")
    print(f"  - Channels needed to explain 90% energy:         {stats['dim_90']} / 64 ({(stats['dim_90']/64)*100:.2f}%)")
    print(f"  - Channels needed to explain 95% energy:         {stats['dim_95']} / 64 ({(stats['dim_95']/64)*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Layer 1 SVD and Representation Diversity")
    parser.add_argument("--ours", type=str, default=None, help="Path to Ours checkpoint")
    parser.add_argument("--pos", type=str, default=None, help="Path to Positive Hebbian checkpoint")
    parser.add_argument("--ours_nobn", type=str, default=None, help="Path to Ours No-BN checkpoint")
    parser.add_argument("--pos_nobn", type=str, default=None, help="Path to Positive Hebbian No-BN checkpoint")
    args = parser.parse_args()

    print("="*80)
    print("🔬 ANALYZING LAYER 1 REPRESENTATION DIVERSITY & SVD SPECTRUM")
    print("="*80)

    # Search for default paths if not provided
    ours_path = args.ours or resolve_path("runs/Hebbian_SSL_20260410-190945/last.pt")
    pos_path = args.pos or resolve_path("runs/Hebbian_SSL_20260525-152245/last.pt")
    ours_nobn_path = args.ours_nobn or resolve_path("runs/NoBN_Backbone_Ours_Lateral_Inhibition/last.pt")
    pos_nobn_path = args.pos_nobn or resolve_path("runs/NoBN_Backbone_Positive_Hebbian_Ablation/last.pt")

    # Log found paths
    print("Resolved checkpoint paths:")
    print(f"  - Standard Ours:      {ours_path or '❌ Not Found'}")
    print(f"  - Standard PosHebb:   {pos_path or '❌ Not Found'}")
    print(f"  - NoBN Ours:          {ours_nobn_path or '❌ Not Found'}")
    print(f"  - NoBN PosHebb:       {pos_nobn_path or '❌ Not Found'}")
    print("-"*80)

    from src.processing.CIFAR100 import CIFAR100_Dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    os.environ["INPUT_NOISE_STD"] = "0.0"
    test_dataset = CIFAR100_Dataset(split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # 1. Evaluate Standard Models
    if ours_path and pos_path:
        print("\n⏳ Loading standard models and extracting features...")
        encoder_ours = get_layer1_encoder(ours_path, use_no_bn=False)
        encoder_pos = get_layer1_encoder(pos_path, use_no_bn=False)
        
        feats_ours = extract_layer1_features(encoder_ours, test_loader)
        feats_pos = extract_layer1_features(encoder_pos, test_loader)
        
        stats_ours = analyze_channel_correlation(feats_ours)
        stats_pos = analyze_channel_correlation(feats_pos)
        
        print("\n" + "="*40 + " STANDARD MODELS (WITH BN) " + "="*40)
        print_result_table("Ours - Anti-Hebbian Lateral Inhibition", stats_ours)
        print_result_table("Positive Hebbian Ablation", stats_pos)
        
        gap_90 = stats_pos['dim_90'] - stats_ours['dim_90']
        gap_entropy = stats_ours['svd_entropy'] - stats_pos['svd_entropy']
        print("-" * 80)
        print(f"📢 STANDARD MODEL GAP: SVD Entropy Gap = {gap_entropy:+.4f} | 90% Energy Dim Gap = {gap_90:+} channels")
        print("="*80)

    # 2. Evaluate NoBN Models
    if ours_nobn_path and pos_nobn_path:
        print("\n⏳ Loading No-BN models and extracting features...")
        encoder_ours_nobn = get_layer1_encoder(ours_nobn_path, use_no_bn=True)
        encoder_pos_nobn = get_layer1_encoder(pos_nobn_path, use_no_bn=True)
        
        feats_ours_nobn = extract_layer1_features(encoder_ours_nobn, test_loader)
        feats_pos_nobn = extract_layer1_features(encoder_pos_nobn, test_loader)
        
        stats_ours_nobn = analyze_channel_correlation(feats_ours_nobn)
        stats_pos_nobn = analyze_channel_correlation(feats_pos_nobn)
        
        print("\n" + "="*40 + " NO-BN MODELS " + "="*40)
        print_result_table("Ours No-BN - Anti-Hebbian Lateral Inhibition", stats_ours_nobn)
        print_result_table("Positive Hebbian Ablation No-BN", stats_pos_nobn)
        
        gap_90_nobn = stats_pos_nobn['dim_90'] - stats_ours_nobn['dim_90']
        gap_entropy_nobn = stats_ours_nobn['svd_entropy'] - stats_pos_nobn['svd_entropy']
        print("-" * 80)
        print(f"📢 NO-BN MODEL GAP: SVD Entropy Gap = {gap_entropy_nobn:+.4f} | 90% Energy Dim Gap = {gap_90_nobn:+} channels")
        print("="*80)

    if not (ours_path and pos_path) and not (ours_nobn_path and pos_nobn_path):
        print("\n❌ Could not find corresponding pairs of checkpoints to compare.")
        print("Please specify paths manually using command arguments, for example:")
        print("  python main/check_layer1_svd.py --ours <ours_last_pt> --pos <pos_last_pt>")
    print()

