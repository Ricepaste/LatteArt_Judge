import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from src.training.SimSiam_train import SimSiam_Model

# -------------------------------------------------------------
# 替換為您的預訓練權重檔案路徑
ENCODER_PATH = "./runs/shuffleNet_v05_SimSiam__33/last.pt" 

# -------------------------------------------------------------
# Evaluation 設定
# -------------------------------------------------------------
EVAL_FRACTION = 1.0
EPOCHS = 100

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

torch.manual_seed(0)
np.random.seed(0)

# 抽樣部分
num_train = len(train_dataset)
labels = train_dataset.targets

train_idx = []
for label in range(10):
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
print("Building RigL ResNet-18 Encoder...")
dummy_trainer = SimSiam_Model(
    pretrained_model=models.resnet18,
    base_lr=0.03
)

simsiam_model = dummy_trainer.model

try:
    print(f"Loading weights from {ENCODER_PATH}...")
    state_dict = torch.load(ENCODER_PATH, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    simsiam_model.load_state_dict(new_state_dict)
    print("Weights loaded successfully!")
except Exception as e:
    print(f"Failed to load weights: {e}")

# 將 Encoder 抽出來並包裝
class ResNetEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean([2, 3])
        x = x.view(x.size(0), -1)
        return x

encoder = ResNetEncoderWrapper(simsiam_model.encoder).to(device)
encoder.eval()

# -------------------------------------------------------------
# Cosine Linear Head & 2-layer MLP Probing
# -------------------------------------------------------------
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.sigma is not None:
            nn.init.constant_(self.sigma, 1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

class MLPProbing(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLPProbing, self).init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------
# 訓練與評估
# -------------------------------------------------------------
def run_eval(classifier, name, epochs):
    print(f"\nStarting {name} Evaluation for {epochs} Epochs...")
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        classifier.train()
        for data, target in tqdm(train_loader, desc=f"{name} train e{epoch}", leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                features = encoder(data)
                features = F.normalize(features, dim=1)
            output = classifier(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        classifier.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                features = encoder(data)
                features = F.normalize(features, dim=1)
                output = classifier(features)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        if accuracy > best_acc:
            best_acc = accuracy
        print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.2f}% (Best: {best_acc:.2f}%)")
    
    print(f"\n--- Final Results for {name} ---")
    print(f"Best Accuracy: {best_acc:.2f}%")

# 1. Cosine Linear Head
cosine_classifier = CosineLinear(512, 10).to(device)
run_eval(cosine_classifier, "Cosine Linear Head", EPOCHS)

# 2. 2-layer MLP Probing
mlp_classifier = MLPProbing(512, 512, 10).to(device)
run_eval(mlp_classifier, "2-layer MLP Probing", EPOCHS)
