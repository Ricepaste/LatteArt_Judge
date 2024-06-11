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

from LatteDataset import TonyLatteDataset
import Siamese_Model


class LatteArtJudge_Model:
    def __init__(
        self,
        pretrained_model=models.efficientnet_b0,
        pretrained_weight=EfficientNet_B0_Weights.DEFAULT,
        model=Siamese_Model.SNN,
        load_weight: str = "",
        freeze=True,
    ) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = pretrained_weight
        self.pretrained_model = pretrained_model(weights=self.weights)
        self.preprocess = self.weights.transforms()

        self.data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(800, scale=(0.8, 1)),  # 資料增補 224
                    transforms.Resize(900),
                    transforms.ColorJitter(
                        contrast=(0.5, 0.8), saturation=(1.2, 1.5)  # type: ignore
                    ),  # type: ignore
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomAffine(degrees=10),
                ]
            ),
            "val": transforms.Compose(
                [
                    # transforms.Resize((255, 255)),
                    transforms.Resize(900),
                    transforms.CenterCrop(800),
                    transforms.Resize(900),
                    # transforms.ColorJitter(contrast=(0.5, 0.8), saturation=(1.2, 1.5)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.Grayscale(num_output_channels=3),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.RandomVerticalFlip(p=0.5),
                    # transforms.RandomRotation(degrees=10),
                    # transforms.RandomAffine(degrees=10),
                ]
            ),
        }

        # freeze the pretrained model
        if freeze:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        self.model = model(self.pretrained_model).to(self.device)

        if load_weight != "":
            self.model.load_state_dict(torch.load(load_weight))

        # 原先使用的損失函數
        # self.criterion = torch.nn.MarginRankingLoss(margin=0.5, reduction="mean")
        self.criterion = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        print("Loaded pretrained model:", self.pretrained_model)
        print("Use device:", self.device)
        print("Original Preprocess:", self.preprocess)
        print(
            "Pretrained model top layer:",
            self.pretrained_model._modules["classifier"],
            sep="\n",
        )

    def dataset_initialize(self, BATCH_SIZE, WORKERS):

        # 使用 ImageFolder 可方便轉換為 dataset
        self.data_dir = ".\\LabelTool"
        # image_datasets = {
        #     x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        #     for x in ["train", "val"]
        # }
        self.image_datasets = {
            x: TonyLatteDataset(
                os.path.join(self.data_dir, x), self.data_transforms[x], x
            )
            for x in ["train", "val"]
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
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ["train", "val"]}
        print(self.dataset_sizes)

        for i, x in self.dataloaders["train"]:
            print(i.shape)
            print(x)
            print(x.data)

    def train(
        self,
        num_epochs=25,
        TRAIN_EFN=False,
        BATCH_SIZE=8,
        WORKERS=0,
    ):
        # 初始化資料集
        self.dataset_initialize(BATCH_SIZE, WORKERS)

        files = os.listdir(".\\runs")
        i = 0
        name = "efficientnet_b0"
        while name in files:
            i += 1
            name = "efficientnet_b0_{}".format(i)
        writer = SummaryWriter("runs\\{}".format(name))

        since = time.time()
        writer.add_graph(self.model, torch.zeros(1, 3, 800, 800).to(self.device))

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        # val_index = 0
        # best_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            distribution = []
            distribution_val = []
            total = 0

            # epoch 10 之後開始訓練原模型
            if epoch == 10 and TRAIN_EFN:
                for param in self.model.parameters():
                    param.requires_grad = True

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                if epoch != 0 and epoch % 10 == 0 and phase == "train":
                    self.dataset: TonyLatteDataset = self.dataloaders[phase].dataset  # type: ignore
                    # self.dataset.restratify()
                    dataset_sizes = {
                        phase: len(self.dataloaders[phase].dataset)  # type: ignore
                        for phase in ["train", "val"]
                    }
                    print(dataset_sizes["train"])

                # 逐批訓練或驗證
                for i, ((img0, img1), label) in enumerate(self.dataloaders[phase]):

                    img0, img1, label = (
                        img0.to(self.device),
                        img1.to(self.device),
                        label.to(self.device),
                    )

                    self.optimizer.zero_grad()

                    # 訓練時需要梯度下降
                    with torch.set_grad_enabled(phase == "train"):
                        output = self.model(img0, img1)
                        loss = self.criterion(output, label)

                        # 訓練時需要 backward + optimize
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    # 統計損失
                    running_loss += loss.item() * img0.size(0)

                    # 統計正確率
                    proba = output > 0.5
                    total += label.size(0)
                    running_corrects += (proba == label).sum().item()
                    print(
                        f"Epoch: {epoch}, Accuracy: {100 * running_corrects / total:.2f}"
                    )

                    # 輸出label的分布範圍，確認是否在猜測期望值
                    if phase == "train":
                        distribution.append(
                            output.detach().cpu().numpy().astype(float).tolist()
                        )
                    elif phase == "val":
                        distribution_val.append(
                            output.detach().cpu().numpy().astype(float).tolist()
                        )

                if phase == "train":
                    distribution = [y for x in distribution for y in x]
                    writer.add_histogram(  # type: ignore
                        "training/output_distribution", np.array(distribution), epoch
                    )
                elif phase == "val":
                    distribution_val = [y for x in distribution_val for y in x]
                    writer.add_histogram(  # type: ignore
                        "validation/output_distribution",
                        np.array(distribution_val),
                        epoch,
                    )

                if phase == "train":
                    self.scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = float(running_corrects) / dataset_sizes[phase]

                if phase == "train":
                    writer.add_scalar(
                        "training/loss", epoch_loss, epoch  # type: ignore
                    )
                    writer.add_scalar(
                        "training/accuracy", epoch_acc, epoch  # type: ignore
                    )
                elif phase == "val":
                    writer.add_scalar(
                        "validation/loss", epoch_loss, epoch  # type: ignore
                    )
                    writer.add_scalar(
                        "validation/accuracy", epoch_acc, epoch  # type: ignore
                    )

                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

                # 如果是評估階段，且準確率創新高即存入 best_model_wts
                # if phase == 'val' and\
                #         (epoch_acc >= best_acc or epoch_loss <= best_loss):
                if phase == "val" and (epoch_acc >= best_acc):
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                (time_elapsed // 60), (time_elapsed % 60)
            )
        )
        print(f"Best val Acc: {best_acc:4f}")

        # 存最後權重
        files = os.listdir(".\\runs")
        i = 0
        old_name = None
        name = "efficientnet_b0"
        while name in files:
            old_name = name
            i += 1
            name = "efficientnet_b0_{}".format(i)
        torch.save(self.model.state_dict(), ".\\runs\\{}\\last.pt".format(old_name))

        # 載入最佳模型
        self.model.load_state_dict(best_model_wts)

        # 存最佳權重
        files = os.listdir(".\\runs")
        i = 0
        name = "efficientnet_b0"
        while name in files:
            old_name = name
            i += 1
            name = "efficientnet_b0_{}".format(i)
        torch.save(self.model.state_dict(), ".\\runs\\{}\\best.pt".format(old_name))

        writer.flush()  # type: ignore
        writer.close()  # type: ignore

        return self.model
