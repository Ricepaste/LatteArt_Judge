import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights
import torchvision.models as models

# 替換為您的預訓練權重檔案路徑
ENCODER_PATH = "./runs/efficientnet_b0_BYOL_2/best.pt"

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載 CIFAR-10 數據集
train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


# 加載預訓練的 encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.efficientnet = models.efficientnet_b0(
        #     weights=EfficientNet_B0_Weights.DEFAULT
        # )
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.encoder = nn.Sequential(
            self.efficientnet.features, self.efficientnet.avgpool
        )
        self.output_dim = self.efficientnet.classifier[1].in_features

    def forward(self, x):
        x = self.encoder(x)  # 只提取特徵，不經過最後的 classifier
        x = x.view(x.size(0), -1)
        return x


encoder = Encoder()
encoder.encoder.load_state_dict(torch.load(ENCODER_PATH))
encoder = encoder.to(device)
encoder.eval()  # 設定為評估模式


# 添加線性分類器
class LinearClassifier(nn.Module):
    def __init__(self, encoder_output_dim, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(encoder_output_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


classifier = LinearClassifier(encoder.output_dim).to(
    device
)  # 假設您的 encoder 有 output_dim 屬性

# 訓練線性分類器
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 10  # 可以根據需要調整 epoch 數量

for epoch in range(epochs):
    # 訓練階段
    classifier.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():  # 凍結 encoder 的權重
            features = encoder(data)
        output = classifier(features)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 測試階段
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            features = encoder(data)
            output = classifier(features)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)  # type: ignore
    accuracy = 100.0 * correct / len(test_loader.dataset)  # type: ignore
    print(
        f"Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
    )
