from pickle import FLOAT, INT
from tkinter.ttk import Progressbar
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from src.processing.Unlabeled_LatteDataset import UL_LatteDataset
from src.processing.CIFAR10 import CIFAR10_Dataset

# Import SimSiam modules
import src.module.SimSiam_Module as SimSiam_Module


class SimSiam_Model:
    def __init__(
        self,
        pretrained_model=models.shufflenet_v2_x0_5,
        pretrained_weight=None,
        load_weight: str = "",
    ) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = pretrained_weight
        self.pretrained_model = pretrained_model(weights=self.weights)
        if self.weights is not None:
            self.preprocess = self.weights.transforms()
        else:
            self.preprocess = None

        # TODO: 這裡的 transform 需要再確認
        self.data_transforms = {
            "train": transforms.Compose(
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
            ),
            "val": transforms.Compose(
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
            ),
        }

        # Initialize SimSiam model
        self.model = SimSiam_Module.SimSiam(self.pretrained_model).to(self.device)

        if load_weight != "":
            self.model.load_state_dict(torch.load(load_weight))

        # Use SGD optimizer for SimSiam
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
        )
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        # Use SimSiam loss
        self.criterion = SimSiam_Module.SimSiamLoss()

        print("Loaded pretrained model:", self.pretrained_model)
        print("Use device:", self.device)
        print("Original Preprocess:", self.preprocess)

    def dataset_initialize(self, DATASET_DIR=".\\LabelTool", BATCH_SIZE=64, WORKERS=0):

        # 使用 ImageFolder 可方便轉換為 dataset
        self.data_dir = DATASET_DIR
        self.image_datasets = {
            x: CIFAR10_Dataset(x, self.data_transforms[x]) for x in ["train", "val"]
        }

        self.dataloaders = {
            x: DataLoader(
                self.image_datasets[x],
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=WORKERS,
            )
            for x in ["train", "val"]
        }
        self.test_one_dataloaders = DataLoader(
            CIFAR10_Dataset("val", self.data_transforms["val"]),
            batch_size=1,
            shuffle=True,
            num_workers=WORKERS,
        )

        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ["train", "val"]}
        print(self.dataset_sizes)

    def train(
        self,
        num_epochs=25,
        batch_size=64,
        workers=0,
        dataset_dir=".\\LabelTool",
    ):
        # 初始化資料集
        self.dataset_initialize(
            DATASET_DIR=dataset_dir, BATCH_SIZE=batch_size, WORKERS=workers
        )

        # TODO: 存擋系統可以單獨拉出來寫成一個函數
        files = os.listdir(".\\runs")
        i = 0
        name = "efficientnet_b0_SimSiam"
        while name in files:
            i += 1
            name = "efficientnet_b0_SimSiam_{}".format(i)
        writer = SummaryWriter("runs\\{}".format(name))

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0

                Progressbar = 0

                # 逐批訓練或驗證
                for i, (img0, img1) in enumerate(self.dataloaders[phase]):

                    img0, img1 = (
                        img0.to(self.device),
                        img1.to(self.device),
                    )

                    self.optimizer.zero_grad()

                    # 訓練時需要梯度下降
                    with torch.set_grad_enabled(phase == "train"):
                        p1, p2, z1, z2 = self.model(img0, img1)

                        loss = self.criterion(p1, p2, z1, z2)

                        # 訓練時需要 backward + optimize
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    # 統計損失
                    running_loss += loss.item() * img0.size(0)

                    Progressbar += img0.size(0)
                    print(
                        f"{running_loss} loss, {loss.item()} item, \
                            size :{img0.size(0)} / {self.dataset_sizes[phase]}, {Progressbar / self.dataset_sizes[phase] * 100:2f}%"
                    )

                if phase == "train":
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]

                if phase == "train":
                    writer.add_scalar("training/loss", epoch_loss, epoch)
                elif phase == "val":
                    writer.add_scalar("validation/loss", epoch_loss, epoch)

                print("{} Loss: {:.4f}".format(phase, epoch_loss))

                # 如果是評估階段，且準確率創新高即存入 best_model_wts
                # if phase == 'val' and\
                #         (epoch_acc >= best_acc or epoch_loss <= best_loss):
                if phase == "val" and (epoch_loss <= best_loss):
                    best_loss = epoch_loss
                    # 存最佳權重
                    files = os.listdir(".\\runs")
                    i = 0
                    name = "efficientnet_b0_SimSiam"
                    old_name = name
                    while name in files:
                        old_name = name
                        i += 1
                        name = "efficientnet_b0_SimSiam_{}".format(i)
                    torch.save(
                        self.model.state_dict(),
                        ".\\runs\\{}\\best.pt".format(old_name),
                    )

                if phase == "train":
                    # 存最後權重
                    files = os.listdir(".\\runs")
                    i = 0
                    old_name = None
                    name = "efficientnet_b0_SimSiam"
                    while name in files:
                        old_name = name
                        i += 1
                        name = "efficientnet_b0_SimSiam_{}".format(i)
                    torch.save(
                        self.model.state_dict(),
                        ".\\runs\\{}\\last.pt".format(old_name),
                    )

            print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                (time_elapsed // 60), (time_elapsed % 60)
            )
        )

        writer.flush()  # type: ignore
        writer.close()  # type: ignore

        return self.model.encoder
