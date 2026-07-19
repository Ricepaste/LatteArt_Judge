import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights
import torchvision.models as models

# 替換為您的預訓練權重檔案路徑
ENCODER_PATH = "./runs/efficientnet_b0_BYOL_2/last.pt"

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載 CIFAR-10 數據集
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((224, 224), scale=(0.6, 1)),
        transforms.ColorJitter(
            contrast=(0.5, 0.8), saturation=(1.2, 1.5)  # type: ignore
        ),  # type: ignore
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=10),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((224, 224), scale=(0.9, 1)),
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)


# 加載預訓練的 encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.efficientnet = models.efficientnet_b0(
        #     weights=EfficientNet_B0_Weights.DEFAULT
        # )
        self.shuffleNet = models.shufflenet_v2_x0_5(weights=None)
        self.encoder = nn.Sequential(
            self.shuffleNet.conv1,
            self.shuffleNet.maxpool,
            self.shuffleNet.stage2,
            self.shuffleNet.stage3,
            self.shuffleNet.stage4,
            self.shuffleNet.conv5,
        )
        self.output_dim = 1024

    def forward(self, x):
        x = self.encoder(x).mean([2, 3])  # 只提取特徵，不經過最後的 classifier
        x = x.view(x.size(0), -1)
        return x


encoder = Encoder()
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

# FIXME 修正encoder並沒有被訓練到的問題
# 訓練線性分類器
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 20  # 可以根據需要調整 epoch 數量

LOG = []

for epoch in range(epochs):
    # 訓練階段
    classifier.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if epoch < 10:
            features = encoder(data)
        else:
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
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(test_loader)}")
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
    LOG.append((epoch, test_loss, accuracy))

for epoch, test_loss, accuracy in LOG:
    print(
        f"Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
    )
