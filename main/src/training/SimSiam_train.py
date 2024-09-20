from pickle import FLOAT, INT
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import torch
import os
from grad_cache.grad_cache import GradCache
from tqdm import tqdm

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
                    transforms.ColorJitter(contrast=(0.5, 0.8), saturation=(1.2, 1.5)),
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
                    transforms.ColorJitter(contrast=(0.5, 0.8), saturation=(1.2, 1.5)),
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
        grad_cache_chunk_size=0,
        batch_size=64,
        workers=0,
        dataset_dir=".\\LabelTool",
    ):
        # 初始化資料集、梯度快取
        self.dataset_initialize(
            DATASET_DIR=dataset_dir, BATCH_SIZE=batch_size, WORKERS=workers
        )

        # Initialize grad cache: include loss function and model
        if grad_cache_chunk_size > 0:
            self.criterion = SimSiam_Module.SimSiamLoss_unsymmetric()
            online_model = SimSiam_Module.SimSiam_online(
                self.pretrained_model,
            ).to(self.device)
            target_model = SimSiam_Module.SimSiam_target(
                self.pretrained_model,
                online=online_model,
            ).to(self.device)

            self.model = [
                online_model,
                target_model,
            ]
            gc = GradCache(
                models=self.model,
                chunk_sizes=grad_cache_chunk_size,
                loss_fn=self.criterion,
            )

        writer = self.save_model(models=self.model, type="tensorboard_init")
        assert isinstance(
            writer, SummaryWriter
        ), "TensorBoard writer initialization failed"

        since = time.time()
        best_loss = float("inf")

        for epoch in tqdm(range(num_epochs), unit="epochs", dynamic_ncols=True):

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.apply_to_models(self.model, lambda model: model.train())
                else:
                    self.apply_to_models(self.model, lambda model: model.eval())

                running_loss = 0.0

                # 逐批訓練或驗證
                for i, (img0, img1) in enumerate(
                    tqdm(
                        self.dataloaders[phase],
                        unit="batchs",
                        leave=False,
                        dynamic_ncols=True,
                    )
                ):

                    img0, img1 = (
                        img0.to(self.device),
                        img1.to(self.device),
                    )

                    self.optimizer.zero_grad()

                    # 訓練時需要梯度下降
                    with torch.set_grad_enabled(phase == "train"):

                        if not (isinstance(self.model, list)):
                            p1, p2, z1, z2 = self.model(img0, img1)
                            loss = self.criterion(p1, p2, z1, z2)
                        elif phase == "val":
                            p1 = self.model[0](img0)
                            z1 = self.model[1](img1)
                            loss = self.criterion(
                                p1.requires_grad_(), z1.requires_grad_()
                            )
                        else:
                            loss = gc.cache_step(img0, img1)

                        # 訓練時需要 backward + optimize
                        if phase == "train":
                            if not (isinstance(self.model, list)):
                                loss.backward()
                            self.optimizer.step()

                    # 統計損失
                    running_loss += loss.item() * img0.size(0)

                if phase == "train":
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]

                if phase == "train":
                    writer.add_scalar("training/loss", epoch_loss, epoch)
                elif phase == "val":
                    writer.add_scalar("validation/loss", epoch_loss, epoch)

                print("\n{} Loss: {:.4f}".format(phase, epoch_loss))

                # 如果是評估階段，且準確率創新高即存入 best_model_wts
                if phase == "val" and (epoch_loss <= best_loss):
                    best_loss = epoch_loss
                    self.save_model(self.model, type="best")

                # 存最後權重
                if phase == "train":
                    self.save_model(self.model, type="last")

            print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                (time_elapsed // 60), (time_elapsed % 60)
            )
        )

        writer.flush()  # type: ignore
        writer.close()  # type: ignore

    def apply_to_models(self, models, func):
        """
        对模型列表或单个模型应用某个函数。

        Args:
            models: 模型列表或单个模型。
            func: 要应用的函数。
        """
        if isinstance(models, list):
            for model in models:
                func(model)
        else:
            func(models)

    def save_model(
        self,
        models,
        filename_prefix="shuffleNet_v05_SimSiam_",
        directory="./runs",
        type="best",
    ):
        """
        保存最佳模型权重到指定目录。
        或者初始化tensorboard的writer並保存到指定目錄

        Args:
        models: 模型列表或单个模型。
        filename_prefix: 文件名
        directory: 保存目录。
        type: "best" 或 "last" 或 "tensorboard_init"
        """
        assert type in [
            "last",
            "best",
            "tensorboard_init",
        ], "type 参数只能是 'best'、'last' 或 'tensorboard_init'"
        files = os.listdir(directory)
        i = 0
        exist_filename = "Unknown"
        next_filename = filename_prefix
        while next_filename in files:
            exist_filename = next_filename
            i += 1
            next_filename = f"{filename_prefix}_{i}"

        if type == "tensorboard_init":
            return SummaryWriter(f"{directory}/{next_filename}")

        filepath = os.path.join(directory, exist_filename, f"{type}.pt")

        if isinstance(models, list):
            torch.save(models[0].state_dict(), filepath)
        else:
            torch.save(models.state_dict(), filepath)
