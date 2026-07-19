import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import DataLoader
from src.training.Hebbian_train import Hebbian_SSL_Trainer

# Setup paths
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ours_path = "/app/main/runs/Hebbian_SSL_20260410-190945/last.pt"
pos_hebb_path = "/app/main/runs/Hebbian_SSL_20260612-231515/last.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetEncoderWrapperBeforeBN(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.output_dim = 512

    def forward(self, x):
        # Slices ResNet up to before the final BN of the second block in layer4
        x = self.encoder[:7](x)
        layer4 = self.encoder[7]
        x = layer4[0](x)
        block1 = layer4[1]
        out = block1.conv1(x)
        out = block1.bn1(out)
        out = block1.relu(out)
        out = block1.conv2(out) # Before final BN (bn2)
        return out

def get_wrapped_encoder_before_bn(checkpoint_path):
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
    
    wrapper = ResNetEncoderWrapperBeforeBN(model.encoder).to(device)
    wrapper.eval()
    return wrapper

def extract_features_to_tensor(encoder, loader):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            images, _, labels = batch
            images = images.to(device)
            feat = encoder(images) # (B, 512, H, W)
            if feat.dim() == 4:
                feat = feat.mean([2, 3]) # Global Avg Pool
            # Note: We do NOT normalize here to see raw scales, or we can normalize. 
            # In standard evaluate_model.py, it normalizes features before linear probe.
            # We will follow evaluate_model.py and normalize.
            feat_norm = torch.nn.functional.normalize(feat, dim=1)
            all_features.append(feat_norm.cpu())
            all_labels.append(labels)
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        # We do NOT use BatchNorm1d inside the classifier to avoid人為修正!
        # Standard evaluate_model.py uses BN inside classifier: self.bn = nn.BatchNorm1d(..., affine=False)
        # To see raw feature power, we evaluate WITHOUT BN inside classifier.
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def train_and_eval_linear_probe(train_x, train_y, test_x, test_y, epochs=50):
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    classifier = LinearClassifier(512, 100).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        classifier.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = classifier(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    # Eval
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = classifier(bx)
            _, pred = out.max(1)
            total += by.size(0)
            correct += pred.eq(by).sum().item()
            
    return correct / total

if __name__ == "__main__":
    print("="*80)
    print("🔬 EVALUATING REPRESENTATIONS WITHOUT BN IN LINEAR PROBING")
    print("  (Training linear classifier directly on raw Pre-BN features)")
    print("="*80)
    
    print("Loading models...")
    encoder_ours = get_wrapped_encoder_before_bn(ours_path)
    encoder_pos = get_wrapped_encoder_before_bn(pos_hebb_path)
    
    # Prepare datasets
    from src.processing.CIFAR100 import CIFAR100_Dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    os.environ["INPUT_NOISE_STD"] = "0.0"
    train_dataset = CIFAR100_Dataset(split="train", transform=transform)
    test_dataset = CIFAR100_Dataset(split="test", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    print("\nExtracting Pre-BN features for Ours...")
    ours_train_x, ours_train_y = extract_features_to_tensor(encoder_ours, train_loader)
    ours_test_x, ours_test_y = extract_features_to_tensor(encoder_ours, test_loader)
    
    print("Extracting Pre-BN features for Positive Hebbian...")
    pos_train_x, pos_train_y = extract_features_to_tensor(encoder_pos, train_loader)
    pos_test_x, pos_test_y = extract_features_to_tensor(encoder_pos, test_loader)
    
    print("\nTraining Linear Probe on Pre-BN features (50 epochs)...")
    ours_acc = train_and_eval_linear_probe(ours_train_x, ours_train_y, ours_test_x, ours_test_y, epochs=50)
    pos_acc = train_and_eval_linear_probe(pos_train_x, pos_train_y, pos_test_x, pos_test_y, epochs=50)
    
    print("\n" + "="*60)
    print("📊 LINEAR PROBING RESULTS ON RAW PRE-BN FEATURES")
    print("="*60)
    print(f"Ours (Anti-Hebbian) Linear Acc:  {ours_acc*100:.2f}%")
    print(f"Positive Hebbian Linear Acc:     {pos_acc*100:.2f}%")
    print(f"Gap (Ours - PosHebb):            {(ours_acc - pos_acc)*100:+.2f}%")
    print("="*60 + "\n")
