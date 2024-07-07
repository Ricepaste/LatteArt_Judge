from torch.utils.data.dataset import Dataset
import os
from PIL import Image


class UL_LatteDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith(("jpg", "jpeg", "png"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns two images for online network and target network.
        both images are the same, but different transformations are applied.
        """
        img_path = self.image_paths[idx]
        image1 = Image.open(img_path).convert("RGB")
        image2 = Image.open(img_path).convert("RGB")
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2
