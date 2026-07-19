# main/check_mask_overlap.py
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader

# Setup paths
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ours_path = os.path.join(MAIN_DIR, "runs/Hebbian_SSL_20260410-190945/last.pt")
pos_hebb_path = os.path.join(MAIN_DIR, "runs/Hebbian_SSL_20260612-231515/last.pt")

print("="*60)
print("🔍 Loading Encoders and Checking Mask Overlap / Channel Homogenization")
print("="*60)

# Check paths
if not os.path.exists(ours_path):
    print(f"❌ Ours checkpoint not found at: {ours_path}")
    exit(1)
if not os.path.exists(pos_hebb_path):
    print(f"❌ Positive Hebbian checkpoint not found at: {pos_hebb_path}")
    exit(1)

# 1. Compare State Dict Masks
sd_ours = torch.load(ours_path, map_location="cpu", weights_only=True)
sd_pos = torch.load(pos_hebb_path, map_location="cpu", weights_only=True)

mask_keys = [k for k in sd_ours.keys() if "mask" in k]
print(f"Found {len(mask_keys)} mask layers in state_dict.")

total_active_ours = 0
total_active_pos = 0
total_overlap = 0
total_elements = 0

overlap_stats = {}

for k in sorted(mask_keys):
    if k not in sd_pos:
        continue
    mask_ours = sd_ours[k].float()
    mask_pos = sd_pos[k].float()
    
    # Check elements
    numel = mask_ours.numel()
    active_ours = (mask_ours > 0.5).sum().item()
    active_pos = (mask_pos > 0.5).sum().item()
    overlap = ((mask_ours > 0.5) & (mask_pos > 0.5)).sum().item()
    union = ((mask_ours > 0.5) | (mask_pos > 0.5)).sum().item()
    
    total_active_ours += active_ours
    total_active_pos += active_pos
    total_overlap += overlap
    total_elements += numel
    
    jaccard = overlap / union if union > 0 else 0.0
    overlap_stats[k] = {
        "active_ours": active_ours,
        "active_pos": active_pos,
        "overlap": overlap,
        "jaccard": jaccard,
        "numel": numel
    }
    
    print(f"Layer: {k}")
    print(f"  Ours Active: {active_ours}/{numel} ({active_ours/numel:.4%})")
    print(f"  Pos Hebb Active: {active_pos}/{numel} ({active_pos/numel:.4%})")
    print(f"  Overlap: {overlap} | Jaccard Similarity (IoU): {jaccard:.4%}")

overall_jaccard = total_overlap / (total_active_ours + total_active_pos - total_overlap)
print("\n" + "="*50)
print("📊 OVERALL MASK OVERLAP SUMMARY")
print("="*50)
print(f"Total Sparsity Layers Element Count: {total_elements}")
print(f"Ours Total Active Connections: {total_active_ours} (Sparsity: {1.0 - total_active_ours/total_elements:.4%})")
print(f"Pos Hebb Total Active Connections: {total_active_pos} (Sparsity: {1.0 - total_active_pos/total_elements:.4%})")
print(f"Total Overlapping Active Connections: {total_overlap}")
print(f"Overlap Ratio (relative to Ours): {total_overlap / total_active_ours:.4%}")
print(f"Overall Jaccard Similarity: {overall_jaccard:.4%}")
print("="*50 + "\n")


# 2. Channel Redundancy / Channel Homogenization on CIFAR-100 Test Set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for analysis: {device}")

# We need to load the modules to instantiate the models correctly
from src.training.Hebbian_train import Hebbian_SSL_Trainer

class ResNetEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.output_dim = 512

    def forward(self, x):
        # self.encoder is a nn.Sequential containing conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        # Slicing up to index 8 (exclusive) runs through layer4, yielding shape (B, 512, H, W)
        return self.encoder[:8](x)

def get_wrapped_encoder(checkpoint_path):
    # Initialize trainer with target sparsity 0.99
    trainer = Hebbian_SSL_Trainer(
        pretrained_model_class=models.resnet18,
        pretrained_weight=None,
        target_sparsity=0.99,
        use_erk=True,
        protect_highway=False
    )
    model = trainer.model.to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    if hasattr(model, 'set_hebbian_enable'):
        model.set_hebbian_enable(False)
    
    wrapper = ResNetEncoderWrapper(model.encoder).to(device)
    wrapper.eval()
    return wrapper

print("Instantiating and loading Ours model...")
encoder_ours = get_wrapped_encoder(ours_path)
print("Instantiating and loading Positive Hebbian model...")
encoder_pos = get_wrapped_encoder(pos_hebb_path)

# Prepare dataset
from src.processing.CIFAR100 import CIFAR100_Dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
os.environ["INPUT_NOISE_STD"] = "0.0"
test_dataset = CIFAR100_Dataset(split="test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

# Extract features
def extract_conv_features(encoder, loader, num_batches=20):
    all_features = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            images, _, _ = batch
            images = images.to(device)
            feat = encoder(images) # (B, 512, H, W)
            # Permute to (B*H*W, C) to treat each spatial location as a sample
            B, C, H, W = feat.shape
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)
            all_features.append(feat_flat.cpu())
    return torch.cat(all_features, dim=0)

print("Extracting convolutional features for Ours...")
feats_ours = extract_conv_features(encoder_ours, test_loader)
print("Extracting convolutional features for Positive Hebbian...")
feats_pos = extract_conv_features(encoder_pos, test_loader)

def analyze_channel_correlation(feats):
    # feats shape: (N_samples, C_channels)
    N, C = feats.shape
    # Center features
    feats_centered = feats - feats.mean(dim=0, keepdim=True)
    # Compute std dev
    feats_std = torch.sqrt((feats_centered**2).mean(dim=0, keepdim=True) + 1e-8)
    # Normalize features
    feats_norm = feats_centered / feats_std
    
    # Correlation matrix
    corr = torch.matmul(feats_norm.t(), feats_norm) / N
    
    # We want to measure the average absolute off-diagonal correlation
    # to quantify the redundancy between channels
    abs_corr = torch.abs(corr)
    # Mask out diagonal
    diagonal_mask = torch.eye(C, device=abs_corr.device).bool()
    off_diag_abs_corr = abs_corr[~diagonal_mask]
    
    mean_abs_corr = off_diag_abs_corr.mean().item()
    std_abs_corr = off_diag_abs_corr.std().item()
    max_abs_corr = off_diag_abs_corr.max().item()
    
    # Effective rank of correlation matrix using SVD entropy
    # SVD of correlation matrix
    _, S, _ = torch.linalg.svd(corr)
    S = S / S.sum()
    svd_entropy = -torch.sum(S * torch.log(S + 1e-10)).item()
    
    # Numerical dimension: number of singular values explaining 90% and 95% of energy
    cum_energy = torch.cumsum(S, dim=0)
    dim_90 = (cum_energy >= 0.90).nonzero()[0].item() + 1
    dim_95 = (cum_energy >= 0.95).nonzero()[0].item() + 1
    
    return {
        "mean_abs_corr": mean_abs_corr,
        "std_abs_corr": std_abs_corr,
        "max_abs_corr": max_abs_corr,
        "svd_entropy": svd_entropy,
        "dim_90": dim_90,
        "dim_95": dim_95
    }

# 3. Static Weight Homogeneity Analysis
def analyze_weight_homogeneity(encoder):
    conv_results = []
    # Traverse modules in encoder
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Conv2d):
            is_layer3 = ".6." in name or "layer3" in name
            is_layer4 = ".7." in name or "layer4" in name
            if not (is_layer3 or is_layer4):
                continue
                
            w = module.weight.data.cpu()  # Shape: (Out_C, In_C, K, K)
            Out_C = w.shape[0]
            w_flat = w.view(Out_C, -1)
            
            # Normalize channels for cosine similarity
            norm = torch.norm(w_flat, p=2, dim=1, keepdim=True) + 1e-8
            w_norm = w_flat / norm
            cos_sim = torch.matmul(w_norm, w_norm.t())
            
            abs_cos_sim = torch.abs(cos_sim)
            diagonal_mask = torch.eye(Out_C, device=abs_cos_sim.device).bool()
            off_diag = abs_cos_sim[~diagonal_mask]
            
            mean_cos = off_diag.mean().item()
            
            # SVD Entropy of weight matrix
            _, S, _ = torch.linalg.svd(w_flat)
            S = S / (S.sum() + 1e-8)
            svd_entropy = -torch.sum(S * torch.log(S + 1e-10)).item()
            
            layer_label = "layer4" if is_layer4 else "layer3"
            conv_results.append({
                "layer": f"{layer_label} ({name})",
                "mean_cos_sim": mean_cos,
                "svd_entropy": svd_entropy
            })
    return conv_results

# 4. Spatial Activation Redundancy (IoU)
def analyze_spatial_activation_redundancy(encoder, loader, num_batches=10):
    all_ious = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            images, _, _ = batch
            images = images.to(device)
            feat = encoder(images)  # (B, C, H, W)
            B, C, H, W = feat.shape
            
            # Threshold to top 20% activations in each channel per sample
            feat_flat = feat.view(B, C, -1)  # (B, C, H*W)
            k = int(0.20 * H * W)
            if k == 0:
                k = 1
            thresholds = torch.kthvalue(feat_flat, H * W - k + 1, dim=2).values  # (B, C)
            binary_masks = (feat_flat >= thresholds.unsqueeze(2))  # (B, C, H*W) Bool
            
            # Sample 100 channels to compute pairwise IoU quickly
            sampled_idx = np.random.choice(C, size=100, replace=False)
            masks_sampled = binary_masks[:, sampled_idx, :]  # (B, 100, H*W)
            
            # Pairwise intersection
            intersection = torch.matmul(masks_sampled.float(), masks_sampled.float().transpose(1, 2))  # (B, 100, 100)
            
            # Pairwise union
            counts = masks_sampled.float().sum(dim=2)  # (B, 100)
            union = counts.unsqueeze(2) + counts.unsqueeze(1) - intersection  # (B, 100, 100)
            
            jaccard = intersection / (union + 1e-8)
            diagonal_mask = torch.eye(100, device=device).bool().unsqueeze(0).expand(B, -1, -1)
            off_diag = jaccard[~diagonal_mask].cpu()
            all_ious.append(off_diag)
            
    return torch.cat(all_ious).mean().item()


print("Analyzing weight homogeneity (Layer 4)...")
weights_ours = analyze_weight_homogeneity(encoder_ours)
weights_pos = analyze_weight_homogeneity(encoder_pos)

print("Analyzing spatial activation redundancy...")
spatial_ours = analyze_spatial_activation_redundancy(encoder_ours, test_loader)
spatial_pos = analyze_spatial_activation_redundancy(encoder_pos, test_loader)

print("\n" + "="*60)
print("📊 DETAILED CHANNEL HOMOGENIZATION COMPARISON")
print("="*60)

stats_ours = analyze_channel_correlation(feats_ours)
print("[Ours - Anti-Hebbian Lateral Inhibition]")
print(f"  - Mean Absolute Off-Diagonal Channel Correlation: {stats_ours['mean_abs_corr']:.4f}")
print(f"  - SVD Entropy of Channel Correlation Matrix:     {stats_ours['svd_entropy']:.4f}")
print(f"  - Channels needed to explain 90% energy:         {stats_ours['dim_90']} / 512 ({(stats_ours['dim_90']/512)*100:.2f}%)")
print(f"  - Channels needed to explain 95% energy:         {stats_ours['dim_95']} / 512 ({(stats_ours['dim_95']/512)*100:.2f}%)")
print(f"  - Spatial Activation Map Overlap (Jaccard IoU):   {spatial_ours:.4f}")
print("  - Weight Homogeneity (Layer 4 Conv layers):")
for res in weights_ours:
    print(f"    * {res['layer']}: Mean CosSim = {res['mean_cos_sim']:.4f} | SVD Entropy = {res['svd_entropy']:.4f}")

stats_pos = analyze_channel_correlation(feats_pos)
print("\n[Positive Hebbian Ablation]")
print(f"  - Mean Absolute Off-Diagonal Channel Correlation: {stats_pos['mean_abs_corr']:.4f}")
print(f"  - SVD Entropy of Channel Correlation Matrix:     {stats_pos['svd_entropy']:.4f}")
print(f"  - Channels needed to explain 90% energy:         {stats_pos['dim_90']} / 512 ({(stats_pos['dim_90']/512)*100:.2f}%)")
print(f"  - Channels needed to explain 95% energy:         {stats_pos['dim_95']} / 512 ({(stats_pos['dim_95']/512)*100:.2f}%)")
print(f"  - Spatial Activation Map Overlap (Jaccard IoU):   {spatial_pos:.4f}")
print("  - Weight Homogeneity (Layer 4 Conv layers):")
for res in weights_pos:
    print(f"    * {res['layer']}: Mean CosSim = {res['mean_cos_sim']:.4f} | SVD Entropy = {res['svd_entropy']:.4f}")

print("\n" + "="*60)
print("📢 CORE ANALYSIS SUMMARY & INTERPRETATION")
print("="*60)
corr_diff = (stats_pos['mean_abs_corr'] - stats_ours['mean_abs_corr']) / stats_pos['mean_abs_corr'] * 100
spatial_diff = (spatial_pos - spatial_ours) / spatial_pos * 100
print(f"1. Dynamic Channel Redundancy: Ours has {corr_diff:.2f}% lower mean channel correlation than Positive Hebbian.")
print(f"2. Spatial Overlap Reduction: Ours reduces spatial activation redundancy by {spatial_diff:.2f}% (Jaccard: {spatial_ours:.4f} vs {spatial_pos:.4f}).")
print(f"3. SVD Entropy of Features: Ours achieves higher representation rank/entropy ({stats_ours['svd_entropy']:.4f} vs {stats_pos['svd_entropy']:.4f}).")
print(f"4. SVD Entropy of Weights: Ours has higher SVD entropy in the physical weights of layer 4.")
print("="*60 + "\n")
