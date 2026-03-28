import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.models as models
import numpy as np
from tqdm import tqdm

from src.training.Hebbian_train import Hebbian_SSL_Trainer

# -------------------------------------------------------------
# 替換為您的預訓練權重檔案路徑 (請填入最新的 V6 跑出來的 run 資料夾中的 best.pt)
# 例如: ENCODER_PATH = "./runs/Hebbian_SSL_20260318-120000/best.pt"
ENCODER_PATH = "./runs/Hebbian_SSL_20260326-174032/last.pt" 

# Ablation 設定（請確保與您訓練時的設定完全一致，才能正確載入權重）
TARGET_SPARSITY = 0.95
USE_ERK = True          
PROTECT_HIGHWAY = True  
# -------------------------------------------------------------

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載 CIFAR-10 數據集
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((224, 224), scale=(0.2, 1)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.2),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((224, 224), scale=(0.9, 1)),
        transforms.ToTensor(),
    ]
)

print("Loading CIFAR-10 dataset...")
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

# -------------------------------------------------------------
# Linear Probing 設定
# -------------------------------------------------------------
# 標準 Linear Probing 必須使用 100% 的訓練數據 (設為 1.0)
# 若要進行半監督評估 (Semi-supervised)，可設為 0.1 (10%) 或 0.01 (1%)
EVAL_FRACTION = 1.0
LINEAR_EPOCHS = 100  # 標準線性評估通常需要訓練較長 Epoch (如 90~100) 以確保分類器完全收斂

torch.manual_seed(0)
np.random.seed(0)

# 抽樣部分：每個 class 抽樣 EVAL_FRACTION 的數據來訓練 Linear Probe 
num_train = len(train_dataset)
indices = list(range(num_train))
labels = train_dataset.targets  # 獲取所有標籤

train_idx = []
for label in range(10):  # CIFAR-10 有 10 個 class
    label_indices = [i for i, x in enumerate(labels) if x == label]
    train_idx.extend(
        np.random.choice(
            label_indices, size=int(EVAL_FRACTION * len(label_indices)), replace=False
        )
    )

train_sampler = SubsetRandomSampler(train_idx)

train_loader = DataLoader(train_dataset, batch_size=256, sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_size=150, shuffle=False)

# -------------------------------------------------------------
# 模型構建與權重載入
# -------------------------------------------------------------
print("Building Hebbian Sparse ResNet-18 Encoder...")
# 透過 Trainer 的 init 來幫我們組裝含有 HebbianSparseLayer 的架構
dummy_trainer = Hebbian_SSL_Trainer(
    pretrained_model_class=models.resnet18,
    target_sparsity=TARGET_SPARSITY,
    use_erk=USE_ERK,
    protect_highway=PROTECT_HIGHWAY
)

simsiam_model = dummy_trainer.model

# 嘗試載入由 SimSiam 訓練存下的權重
try:
    print(f"Loading weights from {ENCODER_PATH}...")
    simsiam_model.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    print("Weights loaded successfully!")
except Exception as e:
    print(f"Failed to load weights: {e}")
    print("Please make sure you have filled in the correct ENCODER_PATH.")

# 將 Encoder 抽出來並包裝
class ResNetEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.output_dim = 512 # ResNet-18 layer4 輸出的 channel 數

    def forward(self, x):
        # 通過 Sparse ResNet-18 到 layer4
        x = self.encoder(x)
        # Global Average Pooling (B, 512, 7, 7) -> (B, 512)
        x = x.mean([2, 3])
        x = x.view(x.size(0), -1)
        return x

encoder = ResNetEncoderWrapper(simsiam_model.encoder).to(device)
encoder.eval()  # 設定為評估模式
# 確保 Hebbian 計算被關閉
simsiam_model.set_hebbian_enable(False) 

# 添加線性分類器 (ResNet-18 輸出為 512 維)
class LinearClassifier(nn.Module):
    def __init__(self, encoder_output_dim, num_classes=10):
        super(LinearClassifier, self).__init__()
        # 在 Linear 前面擋一層 BN 以對齊不同特徵的 Scale
        self.bn = nn.BatchNorm1d(encoder_output_dim, affine=False)
        self.linear = nn.Linear(encoder_output_dim, num_classes)

    def forward(self, x):
        return self.linear(self.bn(x))

classifier = LinearClassifier(512, 10).to(device)

# 訓練線性分類器
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = LINEAR_EPOCHS  # 使用我們上方設定好的 Epoch 數量

LOG = []

print(f"\nStarting Linear Probing Evaluation for {epochs} Epochs...")
for epoch in tqdm(range(epochs), unit="epoch"):
    # 訓練階段
    classifier.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="train", leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():  # 凍結 encoder 的權重 (Linear Probing 只能優化分類器)
            features = encoder(data)
            features = torch.nn.functional.normalize(features, dim=1) # 強制 L2 正規化
        output = classifier(features)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 測試階段
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="test", leave=False):
            data, target = data.to(device), target.to(device)
            features = encoder(data)
            features = torch.nn.functional.normalize(features, dim=1) # 強制 L2 正規化
            output = classifier(features)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"\nEpoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    LOG.append((epoch, test_loss, accuracy))

print("\n--- Final Results ---")
for epoch, test_loss, accuracy in LOG:
    print(f"Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
