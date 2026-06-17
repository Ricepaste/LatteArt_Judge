import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from torchvision import models, transforms
from torch.utils.data import DataLoader
from src.training.Hebbian_train import Hebbian_SSL_Trainer

# Setup paths
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ours_path = "/app/main/runs/Hebbian_SSL_20260410-190945/last.pt"
pos_hebb_path = "/app/main/runs/Hebbian_SSL_20260612-231515/last.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.output_dim = 512

    def forward(self, x):
        return self.encoder[:8](x)

def get_wrapped_encoder(checkpoint_path):
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

def extract_features(encoder, loader):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            images, _, labels = batch
            images = images.to(device)
            feat = encoder(images)
            if feat.dim() == 4:
                feat = feat.mean([2, 3])
            feat_norm = torch.nn.functional.normalize(feat, dim=1)
            all_features.append(feat_norm.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_features), np.concatenate(all_labels)

if __name__ == "__main__":
    print("="*80)
    print("🔬 REPRESENTATION CAPACITY & NOISE ROBUSTNESS BENCHMARK")
    print("  (Comparing Ours/Anti-Hebbian vs Positive Hebbian at 99% Sparsity)")
    print("="*80)
    
    # 1. Load models
    print("Loading Ours (Anti-Hebbian)...")
    encoder_ours = get_wrapped_encoder(ours_path)
    print("Loading Positive Hebbian...")
    encoder_pos = get_wrapped_encoder(pos_hebb_path)
    
    # Prepare datasets
    from src.processing.CIFAR100 import CIFAR100_Dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # We load standard datasets (clean)
    os.environ["INPUT_NOISE_STD"] = "0.0"
    train_dataset = CIFAR100_Dataset(split="train", transform=transform)
    test_dataset = CIFAR100_Dataset(split="test", transform=transform)
    
    # Subsample training data to make it run faster (10,000 samples is plenty for PCA/KNN)
    indices = np.random.choice(len(train_dataset), size=10000, replace=False)
    train_subset = torch.utils.data.Subset(train_dataset, indices)
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Extract clean features
    print("\nExtracting clean features for Ours...")
    ours_train_x, train_y = extract_features(encoder_ours, train_loader)
    ours_test_x, test_y = extract_features(encoder_ours, test_loader)
    
    print("Extracting clean features for Positive Hebbian...")
    pos_train_x, _ = extract_features(encoder_pos, train_loader)
    pos_test_x, _ = extract_features(encoder_pos, test_loader)
    
    # =========================================================================
    # 🔬 TEST 1: Low-Dimensional Representation Capacity (PCA Bottleneck)
    # =========================================================================
    print("\n" + "-"*50)
    print("🔬 TEST 1: Low-Dimensional PCA Probing (Information Capacity)")
    print("  (Forcing 512 dimensions down to compressed feature sub-spaces)")
    print("-"*50)
    
    dimensions = [8, 16, 32, 64, 128, 256, 512]
    pca_results = []
    
    for d in dimensions:
        # Ours PCA + KNN
        if d < 512:
            pca = PCA(n_components=d, random_state=42)
            ours_train_red = pca.fit_transform(ours_train_x)
            ours_test_red = pca.transform(ours_test_x)
            
            pca_pos = PCA(n_components=d, random_state=42)
            pos_train_red = pca_pos.fit_transform(pos_train_x)
            pos_test_red = pca_pos.transform(pos_test_x)
        else:
            ours_train_red, ours_test_red = ours_train_x, ours_test_x
            pos_train_red, pos_test_red = pos_train_x, pos_test_x
            
        # KNN fit & eval
        knn_ours = KNeighborsClassifier(n_neighbors=200)
        knn_ours.fit(ours_train_red, train_y)
        acc_ours = accuracy_score(test_y, knn_ours.predict(ours_test_red))
        
        knn_pos = KNeighborsClassifier(n_neighbors=200)
        knn_pos.fit(pos_train_red, train_y)
        acc_pos = accuracy_score(test_y, knn_pos.predict(pos_test_red))
        
        pca_results.append({
            "Bottleneck Dimension": d,
            "Ours KNN Acc": f"{acc_ours*100:.2f}%",
            "PosHebb KNN Acc": f"{acc_pos*100:.2f}%",
            "Gap (Ours - PosHebb)": f"{(acc_ours - acc_pos)*100:+.2f}%"
        })
        
    df_pca = pd.DataFrame(pca_results)
    print(df_pca.to_string(index=False))
    
    # =========================================================================
    # 🔬 TEST 2: Robustness to Input Corruptions (Noise Generalization)
    # =========================================================================
    print("\n" + "-"*50)
    print("🔬 TEST 2: Input Noise Generalization (Perturbation Robustness)")
    print("  (Adding Gaussian noise to input images during evaluation)")
    print("-"*50)
    
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    noise_results = []
    
    for std in noise_levels:
        # Re-load test set with noise
        os.environ["INPUT_NOISE_STD"] = str(std)
        noise_test_dataset = CIFAR100_Dataset(split="test", transform=transform)
        noise_test_loader = DataLoader(noise_test_dataset, batch_size=128, shuffle=False, num_workers=0)
        
        # Extract noisy test features
        ours_noisy_test_x, _ = extract_features(encoder_ours, noise_test_loader)
        pos_noisy_test_x, _ = extract_features(encoder_pos, noise_test_loader)
        
        # Evaluate KNN (using clean train features)
        knn_ours = KNeighborsClassifier(n_neighbors=200)
        knn_ours.fit(ours_train_x, train_y)
        acc_ours = accuracy_score(test_y, knn_ours.predict(ours_noisy_test_x))
        
        knn_pos = KNeighborsClassifier(n_neighbors=200)
        knn_pos.fit(pos_train_x, train_y)
        acc_pos = accuracy_score(test_y, knn_pos.predict(pos_noisy_test_x))
        
        noise_results.append({
            "Noise Std Dev": std,
            "Ours KNN Acc": f"{acc_ours*100:.2f}%",
            "PosHebb KNN Acc": f"{acc_pos*100:.2f}%",
            "Gap (Ours - PosHebb)": f"{(acc_ours - acc_pos)*100:+.2f}%"
        })
        
    df_noise = pd.DataFrame(noise_results)
    print(df_noise.to_string(index=False))
    print("\n" + "="*80 + "\n")
