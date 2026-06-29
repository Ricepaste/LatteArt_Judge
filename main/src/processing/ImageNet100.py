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

class ImageNet100_Dataset(Dataset):
    def __init__(self, split="train", transform=None):
        # 優先從環境變數讀取本地高速資料集路徑，避開網路檔案系統瓶頸
        env_local_dir = os.environ.get("LOCAL_DATASET_DIR")
        data_root = None
        
        if env_local_dir and os.path.exists(os.path.join(env_local_dir, "train")):
            data_root = env_local_dir
        else:
            candidates = [
                "./data/imagenet-100",
                "main/data/imagenet-100",
                "./data/imagenet-100/imagenet-100",
                "main/data/imagenet-100/imagenet-100",
                "./imagenet-100",
                "main/imagenet-100",
                "./imagenet-100/imagenet-100"
            ]
            for c in candidates:
                # 必須包含 train/ 子目錄才算有效路徑，避免選到空的或套娃資料夾
                if os.path.exists(os.path.join(c, "train")):
                    data_root = c
                    break
                
        if data_root is None:
            # 預設回退路徑以供錯誤提示
            data_root = "./data/imagenet-100"
            
        split_dir = os.path.join(data_root, "train" if split == "train" else "val")
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"ImageNet-100 directory not found at {split_dir}. "
                f"Checked candidates: {candidates}"
            )
            
        self.dataset = torchvision.datasets.ImageFolder(root=split_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        noise_std = float(os.environ.get("INPUT_NOISE_STD", "0.0"))

        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)
            
            if noise_std > 0:
                noise_transform = AddGaussianNoise(0., noise_std)
                image1 = noise_transform(image1)
                image2 = noise_transform(image2)

        return image1, image2, label
