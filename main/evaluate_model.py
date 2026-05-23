import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import numpy as np
from tqdm import tqdm
import random

# 強制固定隨機種子，確保評估結果具備 100% 可重複性 (Reproducibility)
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

from src.processing.CIFAR100 import CIFAR100_Dataset
from src.processing.CIFAR10 import CIFAR10_Dataset

# 環境變數設定
ENCODER_PATH = os.environ.get("ENCODER_PATH", "")
METHOD = os.environ.get("METHOD", "hebbian").lower()
DATASET_NAME = os.environ.get("TARGET_DATASET", "cifar10").lower()
LINEAR_EPOCHS = int(os.environ.get("NUM_EPOCHS", "100"))
EVAL_FRACTION = float(os.environ.get("EVAL_FRACTION", "1.0"))
USE_ERK = os.environ.get("USE_ERK", "True") == "True"
PROTECT_HIGHWAY = os.environ.get("PROTECT_HIGHWAY", "False") == "True"
TARGET_SPARSITY = float(os.environ.get("TARGET_SPARSITY", "0.99"))
LABEL_NOISE_RATE = float(os.environ.get("LABEL_NOISE_RATE", "0.0"))
SHUFFLE_MASK = os.environ.get("SHUFFLE_MASK", "False") == "True"
RANDOM_PRUNE_DENSE = os.environ.get("RANDOM_PRUNE_DENSE", "False") == "True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not ENCODER_PATH or (ENCODER_PATH.lower() not in ["imagenet", "official"] and not os.path.exists(ENCODER_PATH)):
    raise ValueError(f"Invalid ENCODER_PATH: {ENCODER_PATH}")

print("="*60)
print(f"🌍 Transfer Learning Evaluation (Cross-Dataset Generalization)")
print(f"Method: {METHOD.upper()}")
print(f"Source Model Path: {ENCODER_PATH}")
print(f"Target Evaluation Dataset: {DATASET_NAME.upper()}")
print(f"Evaluation Data Fraction: {EVAL_FRACTION * 100}%")
if LABEL_NOISE_RATE > 0.0:
    print(f"Label Noise Flip Rate: {LABEL_NOISE_RATE * 100:.1f}%")
if SHUFFLE_MASK:
    print("🎲 Mode: SHUFFLE_MASK (Randomly shuffling model masks)")
if RANDOM_PRUNE_DENSE:
    print(f"🎲 Mode: RANDOM_PRUNE_DENSE (Randomly pruning dense model to {TARGET_SPARSITY * 100}%)")
print("="*60)

# 動態選擇模組與初始化
backbone_weights = None
if ENCODER_PATH.lower() in ["imagenet", "official"]:
    backbone_weights = models.ResNet18_Weights.IMAGENET1K_V1

if METHOD == "hebbian":
    from src.training.Hebbian_train import Hebbian_SSL_Trainer
    dummy_trainer = Hebbian_SSL_Trainer(
        pretrained_model_class=models.resnet18,
        pretrained_weight=backbone_weights,
        target_sparsity=TARGET_SPARSITY, # 動態對齊目標稀疏度
        use_erk=USE_ERK,
        protect_highway=PROTECT_HIGHWAY
    )
    simsiam_model = dummy_trainer.model.to(device)
elif METHOD in ["random", "set"]:
    from src.training.SET_train import SET_SSL_Trainer
    dummy_trainer = SET_SSL_Trainer(
        pretrained_model_class=models.resnet18,
        pretrained_weight=backbone_weights,
        target_sparsity=TARGET_SPARSITY,
        dataset_name=DATASET_NAME
    )
    simsiam_model = dummy_trainer.model.to(device)
else:
    import src.training.SimSiam_train as SimSiam_train
    dummy_trainer = SimSiam_train.SimSiam_Model(
        pretrained_model=models.resnet18,
        pretrained_weight=backbone_weights
    )
    simsiam_model = dummy_trainer.model.to(device)

if ENCODER_PATH.lower() in ["imagenet", "official"]:
    print("🌍 Using official ImageNet pre-trained weights directly.")
else:
    print(f"Loading weights from {ENCODER_PATH}...")
    state_dict = torch.load(ENCODER_PATH, map_location=device, weights_only=True)
    simsiam_model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully!")

# Shuffled Mask (打亂 Hebbian/RigL 的遮罩)
if SHUFFLE_MASK:
    print("🎲 Randomly shuffling model masks...")
    count = 0
    for name, module in simsiam_model.named_modules():
        if hasattr(module, 'mask') and module.mask is not None:
            mask_flat = module.mask.view(-1)
            perm = torch.randperm(mask_flat.numel(), device=mask_flat.device)
            shuffled_mask = mask_flat[perm].view_as(module.mask)
            module.mask.copy_(shuffled_mask)
            
            # 重新將打亂後的遮罩套用至權重
            if hasattr(module, 'layer') and hasattr(module.layer, 'weight'):
                with torch.no_grad():
                    module.layer.weight.copy_(module.layer.weight * module.mask)
            count += 1
    print(f"🎲 Shuffled masks for {count} layers.")

# Random Prune Dense (對密集模型隨機強行剪枝)
if RANDOM_PRUNE_DENSE:
    print(f"🎲 Randomly pruning dense model weights to target sparsity {TARGET_SPARSITY * 100:.2f}%...")
    count = 0
    for name, module in simsiam_model.named_modules():
        # 對齊 Hebbian 的卷積層保護邏輯，只隨機剪枝 Spatial 卷積層 (kernel_size > 1)
        if isinstance(module, nn.Conv2d) and module.kernel_size != 1 and module.kernel_size != (1, 1):
            with torch.no_grad():
                w = module.weight
                numel = w.numel()
                k = int((1 - TARGET_SPARSITY) * numel) # 保留的 active 連接數
                
                mask = torch.zeros(numel, dtype=torch.float32, device=w.device)
                if k > 0:
                    indices = torch.randperm(numel, device=w.device)[:k]
                    mask[indices] = 1.0
                mask = mask.view_as(w)
                w.copy_(w * mask)
                count += 1
    print(f"🎲 Randomly pruned {count} Conv2d layers.")

# 如果是 Hebbian，確保評估時不再觸發生長
if hasattr(simsiam_model, 'set_hebbian_enable'):
    simsiam_model.set_hebbian_enable(False)

total_params = 0
zero_params = 0

# 對齊 Hebbian_train.py 的計算邏輯：排除第一層與 BN 層
with torch.no_grad():
    for name, param in simsiam_model.named_parameters():
        # 排除包含 'bn' 的層, 排除 downsample.1 (BN), 排除非權重的 bias 或其他 1D 參數
        if 'encoder' in name and 'weight' in name and 'bn' not in name and 'downsample.1' not in name and param.dim() > 1:
            param_numel = param.numel()
            param_zeros = (param.data.abs() < 1e-7).sum().item()
            total_params += param_numel
            zero_params += param_zeros

print("-" * 50)
print(f"Encoder Real Global Sparsity: {zero_params / total_params * 100:.2f}% (Consistent with training)")
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

from torch.utils.data import SubsetRandomSampler
import torchvision.datasets as datasets

if DATASET_NAME == "cifar100":
    train_dataset = CIFAR100_Dataset(split="train", transform=transform)
    test_dataset = CIFAR100_Dataset(split="test", transform=transform)
    num_classes = 100
elif DATASET_NAME == "cifar10":
    train_dataset = CIFAR10_Dataset(split="train", transform=transform)
    test_dataset = CIFAR10_Dataset(split="test", transform=transform)
    num_classes = 10
elif DATASET_NAME == "svhn":
    # SVHN split is 'train' and 'test'
    train_dataset = datasets.SVHN(root="./data", split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN(root="./data", split='test', download=True, transform=transform)
    num_classes = 10
elif DATASET_NAME == "dtd":
    # DTD (Describable Textures Dataset)
    train_dataset = datasets.DTD(root="./data", split='train', download=True, transform=transform)
    test_dataset = datasets.DTD(root="./data", split='test', download=True, transform=transform)
    num_classes = 47
elif DATASET_NAME == "pcam":
    # PCAM (PatchCamelyon Medical Dataset)
    train_dataset = datasets.PCAM(root="./data", split='train', download=True, transform=transform)
    test_dataset = datasets.PCAM(root="./data", split='test', download=True, transform=transform)
    num_classes = 2
elif DATASET_NAME == "eurosat":
    full_dataset = datasets.EuroSAT(root="./data", download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    num_classes = 10
else:
    raise ValueError(f"Unknown dataset: {DATASET_NAME}")

# Label Noise Dataset Wrapper
class LabelNoiseDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_rate, num_classes, seed=42):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        
        num_items = len(dataset)
        orig_labels = []
        
        # 1. 抽取原始標籤，用來做 noisy label 的生成與 sampling 支援
        if hasattr(dataset, 'targets'):
            orig_labels = [int(x) for x in dataset.targets]
        elif hasattr(dataset, 'labels'):
            orig_labels = [int(x) for x in dataset.labels]
        else:
            # Fallback 逐個元素讀取標籤
            sample_item = dataset[0]
            label_pos = 1 if len(sample_item) == 2 else 2
            
            if isinstance(dataset, torch.utils.data.Subset):
                for i in range(num_items):
                    if hasattr(dataset.dataset, 'targets'):
                        orig_labels.append(int(dataset.dataset.targets[dataset.indices[i]]))
                    else:
                        orig_labels.append(int(dataset[i][label_pos]))
            else:
                for i in range(num_items):
                    orig_labels.append(int(dataset[i][label_pos]))
                    
        # 2. 引入標籤噪聲 (Symmetric Label Noise)
        rng = np.random.default_rng(seed)
        noisy_labels = np.array(orig_labels, dtype=int)
        
        if noise_rate > 0.0:
            num_corrupt = int(noise_rate * num_items)
            corrupt_indices = rng.choice(num_items, size=num_corrupt, replace=False)
            
            for idx in corrupt_indices:
                orig_l = orig_labels[idx]
                possible_classes = [c for c in range(num_classes) if c != orig_l]
                if possible_classes:
                    noisy_labels[idx] = rng.choice(possible_classes)
                else:
                    noisy_labels[idx] = orig_l
                    
        self.targets = list(noisy_labels)
        self.labels = noisy_labels
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        noisy_lbl = self.targets[idx]
        if len(item) == 3:
            return item[0], item[1], noisy_lbl
        else:
            return item[0], noisy_lbl

# 套用標籤噪聲包裝器
if LABEL_NOISE_RATE > 0.0:
    print(f"Applying Label Noise Wrapper to train_dataset (Flip Rate: {LABEL_NOISE_RATE * 100:.1f}%)")
    train_dataset = LabelNoiseDatasetWrapper(train_dataset, LABEL_NOISE_RATE, num_classes, seed=42)


# Few-shot sampling logic
num_train = len(train_dataset)
if hasattr(train_dataset, 'targets'):
    labels = train_dataset.targets
elif hasattr(train_dataset, 'labels'): # SVHN uses .labels
    labels = train_dataset.labels
else:
    # Fallback for EuroSAT/Split datasets: extract labels manually
    print(f"Extracting labels for {DATASET_NAME}...")
    labels = []
    # If it's a Subset (from random_split), we need to handle it
    # 自動偵測標籤位置 (有些資料集回傳 2 個元素，有些 3 個)
    sample_item = train_dataset[0]
    label_pos = 1 if len(sample_item) == 2 else 2
    
    if isinstance(train_dataset, torch.utils.data.Subset):
        for i in range(len(train_dataset)):
            # 優先嘗試直接抓 .targets，若無則根據偵測到的位置抓取
            if hasattr(train_dataset.dataset, 'targets'):
                labels.append(train_dataset.dataset.targets[train_dataset.indices[i]])
            else:
                labels.append(train_dataset[i][label_pos])
    else:
        for i in range(num_train):
            labels.append(train_dataset[i][label_pos])

train_idx = []
for label in range(num_classes):
    # 使用 int(x) 確保相容 Tensor, Numpy 或純整數標籤
    label_indices = [i for i, x in enumerate(labels) if int(x) == label]
    if len(label_indices) > 0:
        sample_size = max(1, int(EVAL_FRACTION * len(label_indices)))
        train_idx.extend(
            np.random.choice(label_indices, size=sample_size, replace=False)
        )

train_sampler = SubsetRandomSampler(train_idx)
train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

# ==================== KNN Evaluation ====================
print("\n--- Starting KNN Evaluation ---")
def get_features(loader):
    features = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            if len(batch) == 3:
                images1, _, target = batch
            else:
                images1, target = batch
                
            images1 = images1.to(device)
            feat = encoder(images1)
            # L2 Normalization (Standard for SSL KNN)
            feat = torch.nn.functional.normalize(feat, dim=1)
            features.append(feat.cpu().numpy())
            labels.append(target.numpy())
    return np.vstack(features), np.concatenate(labels)

train_features, train_labels = get_features(train_loader)
test_features, test_labels = get_features(test_loader)

n_neighbors = min(200, len(train_features))
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(train_features, train_labels)
knn_preds = knn.predict(test_features)
knn_acc = accuracy_score(test_labels, knn_preds)
print(f"KNN Protocol Accuracy (k=200): {knn_acc:.4f}")

# ==================== Linear Probing ====================
print("\n--- Starting Linear Probing ---")
class LinearClassifier(nn.Module):
    def __init__(self, encoder_output_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.bn = nn.BatchNorm1d(encoder_output_dim, affine=False)
        self.linear = nn.Linear(encoder_output_dim, num_classes)

    def forward(self, x):
        return self.linear(self.bn(x))

classifier = LinearClassifier(encoder.output_dim, num_classes).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = LINEAR_EPOCHS
for epoch in range(epochs):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in train_loader:
        if len(batch) == 3:
            images1, _, labels = batch
        else:
            images1, labels = batch
        images1, labels = images1.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.no_grad():
            features = encoder(images1)
            features = torch.nn.functional.normalize(features, dim=1)
            
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
    for batch in test_loader:
        if len(batch) == 3:
            images1, _, labels = batch
        else:
            images1, labels = batch
        images1, labels = images1.to(device), labels.to(device)
        features = encoder(images1)
        features = torch.nn.functional.normalize(features, dim=1)
        outputs = classifier(features)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = correct / total
print(f"\nFinal Linear Probing Accuracy: {test_acc:.4f}")

# 寫入結果檔案 (CSV 格式方便收集，可直接丟 Excel)
summary_file = os.environ.get("SUMMARY_FILE_OVERRIDE", "transfer_summary_master.csv")
file_exists = os.path.isfile(summary_file)

with open(summary_file, "a") as f:
    if not file_exists:
        f.write("Method,Source_Model,Target_Dataset,Data_Fraction,Label_Noise_Rate,KNN_Acc,Linear_Acc\n")
    
    # 簡化模型名稱 (只保留資料夾名稱)
    model_name = os.path.basename(os.path.dirname(ENCODER_PATH))
    
    saved_method = METHOD
    if SHUFFLE_MASK:
        saved_method = f"{METHOD}_shuffled"
    elif RANDOM_PRUNE_DENSE:
        saved_method = f"{METHOD}_random_pruned"
        
    f.write(f"{saved_method},{model_name},{DATASET_NAME},{EVAL_FRACTION},{LABEL_NOISE_RATE:.4f},{knn_acc:.4f},{test_acc:.4f}\n")

print("\n" + "="*60)
print(f"📊 FINAL RESULTS for {METHOD.upper()} on {DATASET_NAME.upper()} ({EVAL_FRACTION*100}% labels)")
print(f"  > KNN Accuracy: {knn_acc*100:.2f}%")
print(f"  > Linear Probing: {test_acc*100:.2f}%")
print(f"Results appended to {summary_file}")
print("="*60 + "\n")
