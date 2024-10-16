from cProfile import label
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CIFAR10_Dataset(Dataset):
    def __init__(self, split="train", transform=None):
        if split == "train":
            self.dataset = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True
            )
        elif split == "val" or split == "test":
            self.dataset = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True
            )
        else:
            raise ValueError("Invalid split: {}".format(split))

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]  # CIFAR10 返回 (image, label)

        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)  # 生成兩個經過不同增強後的圖像

        return image1, image2, label
