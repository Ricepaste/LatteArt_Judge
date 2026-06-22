import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import os

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class CIFAR10_Dataset(Dataset):
    def __init__(self, split="train", transform=None):
        # 自動偵測資料集存放路徑以適應不同執行環境，避免無網路環境下載失敗
        data_root = "./data"
        if not os.path.exists(os.path.join(data_root, "cifar-10-batches-py")) and os.path.exists("main/data/cifar-10-batches-py"):
            data_root = "main/data"

        if split == "train":
            self.dataset = torchvision.datasets.CIFAR10(
                root=data_root, train=True, download=True
            )
        elif split == "val" or split == "test":
            self.dataset = torchvision.datasets.CIFAR10(
                root=data_root, train=False, download=True
            )
        else:
            raise ValueError("Invalid split: {}".format(split))

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]  # CIFAR10 返回 (image, label)

        noise_std = float(os.environ.get("INPUT_NOISE_STD", "0.0"))

        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)  # 生成兩個經過不同增強後的圖像
            
            if noise_std > 0:
                noise_transform = AddGaussianNoise(0., noise_std)
                image1 = noise_transform(image1)
                image2 = noise_transform(image2)

        return image1, image2, label
