import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class CIFAR100_Dataset(Dataset):
    def __init__(self, split="train", transform=None):
        if split == "train":
            self.dataset = torchvision.datasets.CIFAR100(
                root="./data", train=True, download=True
            )
        elif split == "val" or split == "test":
            self.dataset = torchvision.datasets.CIFAR100(
                root="./data", train=False, download=True
            )
        else:
            raise ValueError("Invalid split: {}".format(split))

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)  # SimSiam generates two augmented views

        return image1, image2, label
