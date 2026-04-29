import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import random
import numpy as np
from tqdm import tqdm

from src.processing.CIFAR100 import CIFAR100_Dataset
from src.processing.CIFAR10 import CIFAR10_Dataset

# 環境變數設定
ENCODER_PATH = os.environ.get("ENCODER_PATH", "")
METHOD = os.environ.get("METHOD", "hebbian").lower()
DATASET_NAME = os.environ.get("TARGET_DATASET", "cifar10").lower()
LINEAR_EPOCHS = int(os.environ.get("NUM_EPOCHS", "100"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not ENCODER_PATH or not os.path.exists(ENCODER_PATH):
    raise ValueError(f"Invalid ENCODER_PATH: {ENCODER_PATH}")

print("="*60)
print(f"🌍 Transfer Learning Evaluation (Cross-Dataset Generalization)")
print(f"Method: {METHOD.upper()}")
print(f"Source Model Path: {ENCODER_PATH}")
print(f"Target Evaluation Dataset: {DATASET_NAME.upper()}")
print("="*60)

# 動態選擇模組與初始化
if METHOD == "hebbian":
    from src.training.Hebbian_train import Hebbian_SSL_Trainer
    dummy_trainer = Hebbian_SSL_Trainer(
        pretrained_model_class=models.resnet18,
        target_sparsity=0.99, # Sparsity parameter is needed for initialization
    )
    simsiam_model = dummy_trainer.model.to(device)
else:
    import src.training.SimSiam_train as SimSiam_train
    dummy_trainer = SimSiam_train.SimSiam_Model(
        pretrained_model=models.resnet18,
    )
    simsiam_model = dummy_trainer.model.to(device)

print(f"Loading weights from {ENCODER_PATH}...")
state_dict = torch.load(ENCODER_PATH, map_location=device, weights_only=True)
simsiam_model.load_state_dict(state_dict, strict=False)
print("Weights loaded successfully!")

# 如果是 Hebbian，確保評估時不再觸發生長
if hasattr(simsiam_model, 'set_hebbian_enable'):
    simsiam_model.set_hebbian_enable(False)

# --- 2. 稀疏度確認 ---
total_params = 0
zero_params = 0
with torch.no_grad():
    for name, module in simsiam_model.named_modules():
        if 'encoder' in name and (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
            weight = module.weight
            total_params += weight.numel()
            zero_params += (weight.abs() < 1e-7).sum().item()

print("-" * 50)
print(f"Encoder Global Sparsity: {zero_params / total_params * 100:.2f}%")
print("-" * 50)

# 3. 抽取 Encoder 供評估使用
class ResNetEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.output_dim = 512

    def forward(self, x):
        # 通過 Sparse 模型特徵層
        x = self.encoder(x)
        # Global Average Pooling (B, 512, 7, 7) -> (B, 512)
        if x.dim() == 4:
            x = x.mean([2, 3])
        x = x.view(x.size(0), -1)
        return x

encoder = ResNetEncoderWrapper(simsiam_model.encoder).to(device)
encoder.eval()

# 4. 資料集準備 (Linear Probing 專用的乾淨資料，不要 Noise)
os.environ["INPUT_NOISE_STD"] = "0.0"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

if DATASET_NAME == "cifar100":
    train_dataset = CIFAR100_Dataset(split="train", transform=transform)
    test_dataset = CIFAR100_Dataset(split="test", transform=transform)
    num_classes = 100
else:
    train_dataset = CIFAR10_Dataset(split="train", transform=transform)
    test_dataset = CIFAR10_Dataset(split="test", transform=transform)
    num_classes = 10

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

# ==================== KNN Evaluation ====================
print("\n--- Starting KNN Evaluation ---")
def get_features(loader):
    features = []
    labels = []
    with torch.no_grad():
        for images1, _, target in tqdm(loader, desc="Extracting features"):
            images1 = images1.to(device)
            feat = encoder(images1)
            features.append(feat.cpu().numpy())
            labels.append(target.numpy())
    return np.vstack(features), np.concatenate(labels)

train_features, train_labels = get_features(train_loader)
test_features, test_labels = get_features(test_loader)

knn = KNeighborsClassifier(n_neighbors=200)
knn.fit(train_features, train_labels)
knn_preds = knn.predict(test_features)
knn_acc = accuracy_score(test_labels, knn_preds)
print(f"KNN Protocol Accuracy (k=200): {knn_acc:.4f}")

# ==================== Linear Probing ====================
print("\n--- Starting Linear Probing ---")
classifier = nn.Linear(encoder.output_dim, num_classes).to(device)
optimizer = torch.optim.SGD(classifier.parameters(), lr=30.0, momentum=0.9, weight_decay=0)
criterion = nn.CrossEntropyLoss()

epochs = LINEAR_EPOCHS
for epoch in range(epochs):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images1, _, labels in train_loader:
        images1, labels = images1.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.no_grad():
            features = encoder(images1)
            
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if (epoch+1) % 10 == 0 or epoch == epochs - 1:
        train_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

# Test phase
classifier.eval()
correct = 0
total = 0
with torch.no_grad():
    for images1, _, labels in test_loader:
        images1, labels = images1.to(device), labels.to(device)
        features = encoder(images1)
        outputs = classifier(features)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = correct / total
print(f"\nFinal Linear Probing Accuracy: {test_acc:.4f}")

# 寫入結果檔案
result_file = f"transfer_results_{METHOD}_to_{DATASET_NAME}.txt"
with open(result_file, "a") as f:
    f.write(f"Source Model: {ENCODER_PATH}\n")
    f.write(f"Target Dataset: {DATASET_NAME}\n")
    f.write(f"KNN Accuracy: {knn_acc:.4f}\n")
    f.write(f"Linear Probing: {test_acc:.4f}\n")
    f.write("-" * 30 + "\n")
