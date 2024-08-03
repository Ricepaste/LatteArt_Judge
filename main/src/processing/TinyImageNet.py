import os
from PIL import Image
from torch.utils.data import Dataset
from dotenv import load_dotenv


class TinyImageNetBYOLDataset(Dataset):
    def __init__(self, split="train", transform=None):
        load_dotenv()  # 加载 .env 文件，請把資料集的位址放在 DATASET_DIR 環境變數
        dataset_root = os.getenv("DATASET_DIR")  # 獲得 DATASET_DIR 環境變數的位址

        assert dataset_root is not None, "Dataset root not specified"
        self.root = dataset_root
        self.split = split
        self.transform = transform
        self.image_paths = self.get_image_paths()

    def get_image_paths(self):
        """
        取得所有圖像的路徑，並儲存在 image_paths 陣列中
        """
        image_paths = []
        split_dir = os.path.join(self.root, self.split)

        if self.split == "train":
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name, "images")
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    image_paths.append(image_path)
        elif self.split == "val":
            for image_name in os.listdir(os.path.join(split_dir, "images")):
                image_path = os.path.join(split_dir, "images", image_name)
                image_paths.append(image_path)

        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)  # 生成兩個經過不同增強後的圖像

        return image1, image2
