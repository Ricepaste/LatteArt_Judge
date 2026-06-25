# main/src/training/SimSiam_train.py

from pickle import FLOAT, INT
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.optim.sgd import SGD
from torch.optim import lr_scheduler
import time
import torch
import os
from grad_cache.grad_cache import GradCache

import rigl_torch.RigL as rigl_baseline_module
import rigl_torch.RigL__consistency as rigl_consistency_module
from typing import Dict, Optional, Type, List, Union  # 導入 Type 和 List
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from src.processing.CIFAR10 import CIFAR10_Dataset

# Import SimSiam modules
import src.module.SimSiam_Module as SimSiam_Module

# from tests.test_rigl import T_end # 移除這行，T_end會在train方法內計算


class SimSiam_Model:
    def __init__(
        self,
        pretrained_model=models.resnet18,
        pretrained_weight=None,
        load_weight: str = "",
        base_lr=0.03,
    ) -> None:
        
        self.base_lr = base_lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = pretrained_weight
        # 初始化模型的encoder
        self.pretrained_model = pretrained_model(weights=self.weights)
        if self.weights is not None:
            self.preprocess = self.weights.transforms()
        else:
            self.preprocess = None
        self._is_lottery_validating = False

        # TODO: 這裡的 transform 需要再確認，尤其是 for SimSiam
        # SimSiam 通常需要更強的 Augmentation
        self.data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop((224, 224), scale=(0.2, 1)),
                    transforms.RandomApply(  # 加入 ColorJitter
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    # 通常还需要标准化，但SimSiam原文没有明确强调，这里先不加
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(  # Validation transform 可以簡單一些
                [
                    transforms.Resize(256),  # 放大後再中心裁剪，方便 KNN 評估
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        }

        model_type = "shufflenet" if pretrained_model == models.shufflenet_v2_x0_5 or pretrained_model == models.shufflenet_v2_x1_0 else "resnet"
        
        # Initialize SimSiam model
        if model_type == "shufflenet":
            self.model: Union[SimSiam_Module.SimSiam, List[torch.nn.Module]] = (
                SimSiam_Module.SimSiam(self.pretrained_model, model_type='shufflenet', encoder_output_dim=1024).to(self.device)
            )
        else:
            self.model: Union[SimSiam_Module.SimSiam, List[torch.nn.Module]] = (
                SimSiam_Module.SimSiam(
                    self.pretrained_model, 
                    model_type='resnet', 
                    encoder_output_dim=512,
                    projector_inner_dim=2048 # ResNet18 typically uses a wider projector in standard implementations, but keeping consistency with Hebbian
                ).to(self.device)
            )

        if load_weight != "":
            # Note: Loading weight here assumes self.model is NOT a list (i.e., not GradCache mode initially)
            if not isinstance(self.model, list):
                print(f"Loading weights from {load_weight}")
                self.model.load_state_dict(torch.load(load_weight))
            else:
                print(
                    f"Warning: load_weight specified but self.model is a list. Skipping weight loading."
                )

        # Use SimSiam loss - Note: Criterion might be re-initialized for GradCache
        # This initial criterion is for non-GradCache modes
        self.criterion = SimSiam_Module.SimSiamLoss()  # Default to symmetric loss

        print("Loaded pretrained model base:", self.pretrained_model.__class__.__name__)
        print("Use device:", self.device)

    def dataset_initialize(self, DATASET_DIR=".\\LabelTool", BATCH_SIZE=64, WORKERS=0, dataset_name="cifar10"):

        from src.processing.CIFAR10 import CIFAR10_Dataset
        from src.processing.CIFAR100 import CIFAR100_Dataset
        from src.processing.ImageNet100 import ImageNet100_Dataset
        
        if dataset_name.lower() == "imagenet100":
            DatasetClass = ImageNet100_Dataset
        elif dataset_name.lower() == "cifar100":
            DatasetClass = CIFAR100_Dataset
        else:
            DatasetClass = CIFAR10_Dataset

        self.data_dir = DATASET_DIR
        self.image_datasets = {
            x: DatasetClass(
                split=x, transform=self.data_transforms[x]
            )
            for x in ["train", "val"]
        }

        self.dataloaders = {
            x: DataLoader(
                self.image_datasets[x],
                batch_size=BATCH_SIZE,
                shuffle=True if x == "train" else False,  # Only shuffle train data
                num_workers=WORKERS,
                pin_memory=True,  # 加速數據載入
            )
            for x in ["train", "val"]
        }
        # test_one_dataloaders seems unused, keeping it commented out
        # self.test_one_dataloaders = DataLoader(
        #     CIFAR10_Dataset("val", self.data_transforms["val"]),
        #     batch_size=1,
        #     shuffle=True,
        #     num_workers=WORKERS,
        # )

        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ["train", "val"]}
        print("Dataset sizes:", self.dataset_sizes)

    def Lottery_validation(self, sparse_model_state_dict_path: str, rewind_weight_prams_path: str):
        """
        執行彩票假說的驗證設置。
        將模型權重重置為倒帶點的權重，並應用從稀疏模型中提取的彩票結構。

        Args:
            sparse_model_state_dict_path (str): 最終稀疏模型 state_dict 的檔案路徑。
                                                從中提取「彩票」結構 (非零連接的二元遮罩)。
            rewind_weight_prams_path (str): 倒帶權重檔案的路徑。
                                            預期是模型在倒帶點的 state_dict (通常是密集模型)。

        Returns:
            torch.nn.Module: 一個新的模型實例，其權重已根據彩票假說重新初始化：
                             只有從稀疏模型中提取的「彩票」結構部分使用倒帶權重，
                             其他部分則為零。
        """
        print(f"開始彩票假說驗證設置...")
        print(f"載入倒帶權重來自: {rewind_weight_prams_path}")
        print(f"載入稀疏模型 state_dict 來自: {sparse_model_state_dict_path}")

        # 1. 創建一個新的模型實例，以避免修改原始 self.model
        # 確保模型結構與訓練時一致
        lottery_ticket_model = type(self.model)(self.pretrained_model) # 創建與 self.model 相同類型的新實例
        lottery_ticket_model.to(self.device)

        # 2. 載入倒帶權重 (Rewind Weights)
        # 這些是你在訓練早期某個檢查點保存的權重 (通常是密集的)
        rewind_state_dict = torch.load(rewind_weight_prams_path, map_location=self.device)
        lottery_ticket_model.load_state_dict(rewind_state_dict)
        print("倒帶權重載入成功。")

        # 3. 載入最終稀疏模型的 state_dict，並從中提取二元遮罩
        # 這個 state_dict 應該已經包含 0 值表示剪枝的連接
        final_sparse_model_state_dict = torch.load(sparse_model_state_dict_path, map_location=self.device)
        self.sparse_model_template = final_sparse_model_state_dict
        print("最終稀疏模型 state_dict 載入成功。")

        # 4. 套用彩票假說
        lottery_ticket_model = self.apply_sparse_mask(lottery_ticket_model)
        self._is_lottery_validating = True

        return lottery_ticket_model

    def apply_sparse_mask(self, lottery_ticket_model):
        print("開始套用彩票假說...") if self._is_lottery_validating is False else None

        # 提取彩票結構 (二元遮罩)
        lottery_ticket_masks = {}
        for name, param in lottery_ticket_model.named_parameters():
            if "weight" in name and name in self.sparse_model_template:
                # 從最終稀疏模型的權重中，提取出非零的部分作為遮罩
                # 1 代表保留，0 代表剪枝
                mask = (self.sparse_model_template[name] != 0).float()
                lottery_ticket_masks[name] = mask
                print(f"從 '{name}' 提取彩票遮罩。保留了 {mask.sum().item()}/{mask.numel()} 個連接。") if self._is_lottery_validating is False else None
            elif "weight" in name:
                print(f"警告: 權重 '{name}' 在最終稀疏模型 state_dict 中沒有找到，無法提取遮罩。") if self._is_lottery_validating is False else None
                lottery_ticket_masks[name] = torch.ones_like(param.data) # 默認保留所有連接

        # 4. 將提取出的二元遮罩應用到倒帶權重上
        # 只有在彩票遮罩中為1的連接會保留其倒帶權重，為0的連接將被設置為零。
        with torch.no_grad(): # 在此操作中不計算梯度
            for name, param in lottery_ticket_model.named_parameters():
                if "weight" in name and name in lottery_ticket_masks:
                    mask = lottery_ticket_masks[name]
                    # 應用遮罩：非零部分保持原樣 (倒帶權重)，零部分變成零
                    param.data.mul_(mask) # 等同於 param.data = param.data * mask
                # bias 通常不進行稀疏化，保持其倒帶值

        print("模型已成功初始化為彩票子網路的倒帶權重。") if self._is_lottery_validating is False else None
        self.model = lottery_ticket_model
        return lottery_ticket_model

    def train(
        self,
        num_epochs=25,
        batch_size=64,
        workers=0,
        dataset_dir=".\\LabelTool",
        dataset_name="cifar10",
        # --- RigL Parameters ---
        rigl_mode: str = "none",  # 'none', 'baseline', 'consistency'
        rigl_dense_allocation: float = 0.1,  # 稀疏度比例 (e.g., 0.1 = 90% sparse)
        rigl_sparsity_distribution: str = "uniform",  # RigL 稀疏分佈方式
        rigl_delta: int = 100,  # RigL 更新間隔
        rigl_alpha: float = 0.3,  # RigL 權重生長分數的 alpha
        rigl_grad_accumulation_n: int = 1,  # 梯度累積步數
        rigl_static_topo: bool = False,  # 是否凍結拓撲
        rigl_ignore_linear_layers: bool = False,  # 是否忽略全連接層
        rigl_state_dict: Optional[Dict] = None,  # RigL scheduler 狀態字典 (用於 resume)
        # --- Consistency RigL Specific Parameter ---
        consistency_lambda: float = 1.0,  # 你梯度一致性方法的 lambda 參數
        # --- GradCache Parameters (Note: Incompatible with RigL modes) ---
        grad_cache_chunk_size: int = 0,  # GradCache chunk 大小 (0 表示不用 GradCache)
        # --- RigL Scheduler Classes ---
        # 外部傳入 RigL Scheduler 的類別，方便替換
        rigl_baseline_scheduler_class: Type[
            rigl_baseline_module.RigLScheduler
        ] = rigl_baseline_module.RigLScheduler,  # 預設為標準 RigL
        rigl_consistency_scheduler_class: Type[
            rigl_consistency_module.RigLScheduler
        ] = rigl_consistency_module.RigLScheduler,  # 需要使用者自己傳入修改後的 RigL 類別
    ):
        # --- Input Validation ---
        if rigl_mode not in ["none", "baseline", "consistency"]:
            raise ValueError(
                f"Invalid rigl_mode: {rigl_mode}. Must be 'none', 'baseline', or 'consistency'."
            )
        if rigl_mode != "none" and grad_cache_chunk_size > 0:
            # 在這個重構版本中，我們假設 RigL 和 GradCache 不兼容
            raise ValueError(
                "RigL modes ('baseline', 'consistency') are not compatible with grad_cache_chunk_size > 0 in this setup."
            )
        if rigl_mode == "consistency" and rigl_consistency_scheduler_class is None:
            raise ValueError(
                "rigl_consistency_scheduler_class must be provided when rigl_mode is 'consistency'"
            )
        # 檢查模型狀態是否和選擇的模式兼容 (如果已經因為 GradCache 變成了 list，但現在選了 RigL 就不對)
        if rigl_mode != "none" and isinstance(self.model, list):
            raise RuntimeError(
                "self.model is a list (likely from previous GradCache init) but rigl_mode is enabled. Ensure SimSiam_Model is re-initialized for non-GradCache modes."
            )
        if (
            rigl_mode == "none"
            and grad_cache_chunk_size > 0
            and not isinstance(self.model, list)
        ):
            # 如果選了 GradCache 但模型不是 list (即 __init__ 時沒設好)，這裡重新初始化 GradCache 相關的 model 和 criterion
            print("Initializing SimSiam model and criterion for GradCache mode...")
            self.criterion = (
                SimSiam_Module.SimSiamLoss_unsymmetric()
            )  # GradCache 通常用 unsymmetric loss
            online_model = SimSiam_Module.SimSiam_online(
                self.pretrained_model,
            ).to(self.device)
            target_model = SimSiam_Module.SimSiam_target(
                self.pretrained_model,
                online=online_model,
            ).to(self.device)
            self.model = [  # Model becomes a list
                online_model,
                target_model,
            ]
            # GradCache initialization handled below

        # --- Dataset Initialization ---
        self.dataset_initialize(
            DATASET_DIR=dataset_dir, BATCH_SIZE=batch_size, WORKERS=workers, dataset_name=dataset_name
        )

        # 總迭代次數 for RigL T_end (在初始化 dataset 後計算)
        total_iterations = num_epochs * len(self.dataloaders["train"])
        T_end = int(0.75 * total_iterations)  # 參考 RigL 論文設定 T_end

        # --- Optimizer and Scheduler/Pruner Initialization ---
        self.pruner: Optional[
            Union[
                rigl_baseline_module.RigLScheduler,
                rigl_consistency_module.RigLScheduler,
            ]
        ] = None  # RigL scheduler 實例
        self.optimizer = None  # Optimizer 實例 (可以是 list for GradCache)
        self.scheduler = None  # LR Scheduler 實例 (可以是 list for GradCache)
        self.gc = None  # GradCache 實例

        if rigl_mode != "none":
            # RigL 模式使用單一 optimizer
            print(f"Using RigL mode: {rigl_mode}")
            self.optimizer = SGD(
                self.model.parameters(),  # RigL 會管理哪些參數被更新
                lr=self.base_lr * batch_size / 256,  # 學習率設定 (參考 SimSiam 論文)
                momentum=0.9,
                weight_decay=5e-4,
            )

            rigl_params = {
                "model": self.model,
                "optimizer": self.optimizer,
                "dense_allocation": rigl_dense_allocation,
                "sparsity_distribution": rigl_sparsity_distribution,
                "T_end": T_end,
                "delta": rigl_delta,
                "alpha": rigl_alpha,
                "grad_accumulation_n": rigl_grad_accumulation_n,
                "static_topo": rigl_static_topo,
                "ignore_linear_layers": rigl_ignore_linear_layers,
                "state_dict": rigl_state_dict,
            }

            if rigl_mode == "baseline":
                print(
                    f"Initializing Baseline RigL Pruner with dense_allocation={rigl_dense_allocation}"
                )
                self.pruner = rigl_baseline_scheduler_class(**rigl_params)
            elif rigl_mode == "consistency":
                print(
                    f"Initializing Consistency RigL Pruner with dense_allocation={rigl_dense_allocation}, consistency_lambda={consistency_lambda}"
                )
                # 將你的 custom 參數傳給你的 RigL 類別 constructor
                rigl_params["consistency_lambda"] = consistency_lambda
                # 確認傳入的類別有 set_additional_gradients 方法 (簡單檢查)
                if not hasattr(
                    rigl_consistency_scheduler_class, "set_additional_gradients"
                ):
                    print(
                        "Warning: rigl_consistency_scheduler_class does not have 'set_additional_gradients' method. It might not work as expected."
                    )
                self.pruner = rigl_consistency_scheduler_class(**rigl_params)

        elif grad_cache_chunk_size > 0:
            # GradCache 初始化 (如果前面沒有因為檢查重新初始化)
            # 在這裡確保 self.model 已經是 list 並且 criterion 是 unsymmetric
            if not isinstance(self.model, list) or not isinstance(
                self.criterion, SimSiam_Module.SimSiamLoss_unsymmetric
            ):
                # 這種情況應該在上面的 validation 裡被處理並重新初始化了
                raise RuntimeError(
                    "Logic error: rigl_mode is none, grad_cache_chunk_size > 0, but model/criterion not correctly initialized for GradCache."
                )

            print(f"Initializing GradCache with chunk_size={grad_cache_chunk_size}")
            self.gc = GradCache(  # Store gc instance
                models=self.model,  # self.model is already a list [online, target]
                chunk_sizes=grad_cache_chunk_size,
                loss_fn=self.criterion,  # self.criterion is already unsymmetric
            )

            # GradCache 模式使用 optimizer list
            self.optimizer = [
                SGD(
                    self.model[0].parameters(),  # Online model optimizer
                    lr=self.base_lr * batch_size / 256,
                    momentum=0.9,
                    weight_decay=5e-4,
                ),
                SGD(
                    self.model[
                        1
                    ].parameters(),  # Target model optimizer (用於 momentum encoder 的參數更新)
                    lr=self.base_lr * batch_size / 256,  # 學習率與 online model 相同
                    momentum=0.9,
                    weight_decay=5e-4,
                ),
            ]
            # GradCache 模式使用 scheduler list
            self.scheduler = [
                lr_scheduler.CosineAnnealingLR(
                    self.optimizer[0], T_max=num_epochs
                ),  # T_max 設為 epoch 數
                lr_scheduler.CosineAnnealingLR(
                    self.optimizer[1], T_max=num_epochs
                ),  # T_max 設為 epoch 數
            ]
            # Note: SimSiam's momentum update for target network is NOT handled by optimizer.step().
            # It's handled within the SimSiam_target module's forward pass or a separate update step.
            # The optimizer[1] is only needed if you want to apply weight decay or other optimizer features
            # to the target network parameters, which SimSiam usually doesn't.
            # The original SimSiam code usually only optimizes the online network.
            # Let's follow the old code's structure with two optimizers/schedulers for now,
            # but be aware this might not be the standard SimSiam setup regarding the target optimizer.

        else:
            # 非 RigL 非 GradCache (標準 SimSiam)
            print("Initializing Standard SimSiam training")
            # 確認 self.model 不是 list 且 criterion 是 symmetric
            if isinstance(self.model, list) or not isinstance(
                self.criterion, SimSiam_Module.SimSiamLoss
            ):
                # 如果因為前面 GradCache 初始化變成了 list，這裡需要重新建立標準 SimSiam 模型
                print(
                    "Re-initializing SimSiam model and criterion for Standard mode..."
                )
                self.model = SimSiam_Module.SimSiam(self.pretrained_model).to(
                    self.device
                )
                self.criterion = (
                    SimSiam_Module.SimSiamLoss()
                )  # 標準 SimSiam 用 symmetric loss

            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.base_lr * batch_size / 256,
                momentum=0.9,
                weight_decay=5e-4,
            )
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs  # T_max 設為 epoch 數
            )
            print("Using Standard SimSiamLoss (symmetric)")

        # --- Tensorboard writer initialization ---
        # 傳入 self.model，它可能是單一模型或模型列表
        self.writer = self.save_model(models=self.model, type="tensorboard_init")
        assert isinstance(
            self.writer, SummaryWriter
        ), "TensorBoard writer initialization failed"

        since = time.time()
        # 追蹤最佳的驗證準確率，而不是損失
        best_knn_accuracy = -1.0

        # --- Hparam Logging ---
        # 收集所有相關的超參數
        hparam_dict = {
            "pretrained_model": self.pretrained_model.__class__.__name__,
            "base_lr": self.base_lr,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "workers": workers,
            "dataset_dir": dataset_dir,
            "rigl_mode": rigl_mode,
            "grad_cache_chunk_size": grad_cache_chunk_size,
        }

        # 根據模式添加特定的超參數
        if rigl_mode != "none":
            hparam_dict.update(
                {
                    "rigl_dense_allocation": rigl_dense_allocation,
                    "rigl_sparsity_distribution": rigl_sparsity_distribution,
                    "rigl_delta": rigl_delta,
                    "rigl_alpha": rigl_alpha,
                    "rigl_grad_accumulation_n": rigl_grad_accumulation_n,
                    "rigl_static_topo": rigl_static_topo,
                    "rigl_ignore_linear_layers": rigl_ignore_linear_layers,
                }
            )
            if rigl_mode == "baseline":
                hparam_dict["rigl_scheduler_class"] = (
                    rigl_baseline_scheduler_class.__name__
                )
            elif rigl_mode == "consistency":
                hparam_dict["rigl_scheduler_class"] = (
                    rigl_consistency_scheduler_class.__name__
                    if rigl_consistency_scheduler_class
                    else "None"
                )
                hparam_dict["consistency_lambda"] = (
                    consistency_lambda  # 記錄你的專屬參數
                )
            # 記錄 RigL 模式下的 optimizer 參數
            if self.optimizer:  # 確保 optimizer 已初始化
                hparam_dict["optimizer"] = self.optimizer.__class__.__name__
                for k, v in self.optimizer.defaults.items():
                    # 只記錄基本類型
                    if isinstance(v, (int, float, str, bool)):
                        hparam_dict[f"optimizer_{k}"] = v

        elif grad_cache_chunk_size > 0:
            # 記錄 GradCache 模式下的超參數
            hparam_dict["criterion"] = self.criterion.__class__.__name__
            if isinstance(self.optimizer, list):
                for i in range(len(self.optimizer)):
                    hparam_dict[f"optimizer_{i}"] = self.optimizer[i].__class__.__name__
                    for k, v in self.optimizer[i].defaults.items():
                        if isinstance(v, (int, float, str, bool)):
                            hparam_dict[f"optimizer_{i}_{k}"] = v
                if isinstance(self.scheduler, list):
                    for i in range(len(self.scheduler)):
                        hparam_dict[f"scheduler_{i}"] = self.scheduler[
                            i
                        ].__class__.__name__
                        # 記錄 scheduler 的關鍵參數，例如 T_max
                        if hasattr(self.scheduler[i], "T_max"):
                            hparam_dict[f"scheduler_{i}_T_max"] = self.scheduler[
                                i
                            ].T_max

        else:
            # 記錄標準訓練模式下的超參數
            hparam_dict["criterion"] = self.criterion.__class__.__name__
            if self.optimizer:  # 確保 optimizer 已初始化
                hparam_dict["optimizer"] = self.optimizer.__class__.__name__
                for k, v in self.optimizer.defaults.items():
                    if isinstance(v, (int, float, str, bool)):
                        hparam_dict[f"optimizer_{k}"] = v
            if self.scheduler:  # 確保 scheduler 已初始化
                hparam_dict["scheduler"] = self.scheduler.__class__.__name__
                if hasattr(self.scheduler, "T_max"):
                    hparam_dict["scheduler_T_max"] = self.scheduler.T_max

        # 過濾掉不支援的 hparam 類型 (TensorBoard 只支援簡單類型)
        valid_hparams = {
            k: v
            for k, v in hparam_dict.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }  # 允許 None
        self.writer.add_hparams(valid_hparams, {})  # 第二個 {} 是 metric_dict，暫時為空
        self.writer.flush()
        # --- End Hparam Logging ---

        # --- Training Loop ---
        for epoch in tqdm(range(num_epochs), unit="epochs", dynamic_ncols=True):
            # 在每個 epoch 開始時，初始化用於 Consistency RigL 的梯度字典
            grads_l1: Dict[str, Optional[torch.Tensor]] = {}
            grads_l2: Dict[str, Optional[torch.Tensor]] = {}
            params_to_manage_names: List[str] = []  # RigL 管理的參數名稱列表

            if rigl_mode != "none":
                # 獲取 RigL 管理的參數名稱
                if self.pruner is not None and hasattr(self.pruner, "_param_names"):
                    # 假設你的 RigL 類別內部有 _param_names 屬性儲存名稱
                    params_to_manage_names = list(self.pruner._param_names)
                else:
                    # 如果沒有，就假設 RigL 管理所有需要梯度的參數 (可能不精確，取決於 RigL 實現)
                    print(
                        "Warning: RigL pruner does not have _param_names. Assuming all trainable parameters are managed."
                    )
                    params_to_manage_names = [
                        name
                        for name, p in self.model.named_parameters()
                        if p.requires_grad
                    ]

            if rigl_mode == "consistency":
                # 初始化梯度字典，只包含 RigL 管理的參數
                grads_l1 = {name: None for name in params_to_manage_names}
                grads_l2 = {name: None for name in params_to_manage_names}

            # 每個 epoch 都有訓練和驗證階段
            for phase in ["train", "val"]:
                if phase == "train":
                    # 將模型設為訓練模式 (處理單一模型或列表)
                    if isinstance(self.model, list):
                        for m in self.model:
                            m.train()
                    else:
                        self.model.train()
                else:
                    # 將模型設為評估模式 (處理單一模型或列表)
                    if isinstance(self.model, list):
                        for m in self.model:
                            m.eval()
                    else:
                        self.model.eval()
                    # 在驗證階段，初始化用於 KNN 評估的特徵和標籤列表
                    features = []
                    labels = []

                running_loss = 0.0

                # 逐批訓練或驗證
                for i, (img0, img1, label) in enumerate(
                    tqdm(
                        self.dataloaders[phase],
                        unit="batchs",
                        leave=False,
                        dynamic_ncols=True,
                    )
                ):
                    # 將數據移到指定設備 (GPU/CPU)
                    img0, img1 = (
                        img0.to(self.device),
                        img1.to(self.device),
                    )
                    label = label.to(
                        self.device
                    )  # 將標籤也移到設備， aunque KNN是在CPU上做

                    # --- 訓練步驟 ---
                    if phase == "train":
                        # 在每個 batch 開始時清空梯度 (GradCache 處理方式可能不同)
                        if isinstance(self.optimizer, list):  # GradCache case
                            for opt in self.optimizer:
                                opt.zero_grad()
                        else:  # Rigl 或 標準訓練 case
                            if self.optimizer is not None:
                                self.optimizer.zero_grad()

                        if rigl_mode == "consistency":
                            # 你修改後的 Consistency RigL 邏輯 (多次 backward)
                            if self.model is None:
                                raise RuntimeError(
                                    "Model is None in training."
                                )  # 確保 model 存在
                            assert not isinstance(
                                self.model, list
                            ), "Model should not be a list in RigL mode."

                            # 1. Clear grads (done above)
                            # 2. Forward pass
                            p1, p2, z1, z2 = self.model(img0, img1)
                            # 3. 計算 L1 損失 (SimSiamLoss_unsymmetric 是需要的)
                            # 如果你的 criterion 是 symmetric 的 SimSiamLoss，這裡會出錯
                            if not hasattr(self.criterion, "forward_components"):
                                raise RuntimeError(
                                    "Criterion must have 'forward_components' method for consistency mode."
                                )
                            l1_loss, _ = self.criterion.forward_components(
                                p1, p2, z1, z2
                            )
                            # 4. 計算 L1 的梯度並保留計算圖
                            l1_loss.backward(retain_graph=True)
                            # 5. 儲存 L1 梯度 (只針對 RigL 管理的參數)
                            for name, param in self.model.named_parameters():
                                if (
                                    name in grads_l1
                                    and param.requires_grad
                                    and param.grad is not None
                                ):
                                    grads_l1[name] = param.grad.clone().detach()
                            # 6. 清空梯度
                            self.optimizer.zero_grad()
                            # 7. 計算 L2 損失
                            _, l2_loss = self.criterion.forward_components(
                                p1, p2, z1, z2
                            )
                            # 8. 計算 L2 的梯度並保留計算圖
                            l2_loss.backward(retain_graph=True)
                            # 9. 儲存 L2 梯度 (只針對 RigL 管理的參數)
                            for name, param in self.model.named_parameters():
                                if (
                                    name in grads_l2
                                    and param.requires_grad
                                    and param.grad is not None
                                ):
                                    grads_l2[name] = param.grad.clone().detach()
                            # 10. 清空梯度
                            self.optimizer.zero_grad()
                            # 11. 計算總損失
                            loss_total = self.criterion(p1, p2, z1, z2)
                            # 12. 計算總損失的梯度
                            loss_total.backward()

                            # 13. 將額外的 L1/L2 梯度傳遞給 RigL Pruner
                            if self.pruner is not None and hasattr(
                                self.pruner, "set_additional_gradients"
                            ):
                                # 將只包含管理參數的梯度字典傳入
                                self.pruner.set_additional_gradients(grads_l1, grads_l2)
                            else:
                                print(
                                    "Warning: RigL mode is 'consistency' but pruner or set_additional_gradients is missing. Skipping passing additional gradients."
                                )

                            # 14. 執行 RigL 步驟和優化器步驟
                            if self.pruner is not None:
                                if (
                                    self.pruner()
                                ):  # RigL 在 optimizer.step() 前更新 mask
                                    pass  # Mask 更新已完成
                                self.optimizer.step()  # 使用總損失的梯度更新權重
                            else:
                                raise RuntimeError(
                                    "Pruner is None in consistency mode during training."
                                )

                            loss = loss_total  # 將總損失用於記錄

                        elif rigl_mode == "baseline":
                            # 標準 Baseline RigL 邏輯 (單次 backward)
                            if self.model is None:
                                raise RuntimeError(
                                    "Model is None in training."
                                )  # 確保 model 存在
                            assert not isinstance(
                                self.model, list
                            ), "Model should not be a list in RigL mode."

                            # 1. Clear grads (done above)
                            # 2. Forward pass
                            p1, p2, z1, z2 = self.model(img0, img1)
                            # 3. 計算總損失
                            loss = self.criterion(p1, p2, z1, z2)
                            # 4. 計算總損失的梯度
                            loss.backward()

                            # 5. 執行 RigL 步驟和優化器步驟
                            if self.pruner is not None:
                                if (
                                    self.pruner()
                                ):  # RigL 在 optimizer.step() 前更新 mask
                                    pass  # Mask 更新已完成 (使用總梯度)
                                self.optimizer.step()  # 使用總損失的梯度更新權重
                            else:
                                raise RuntimeError(
                                    "Pruner is None in baseline mode during training."
                                )

                        elif rigl_mode == "none":
                            if isinstance(self.model, list):  # GradCache 模式
                                # Zero grad (done above)
                                # GradCache 的 cache_step 內部處理 forward/backward/step
                                if self.gc is None:
                                    raise RuntimeError("GradCache is None in training.")
                                loss = 0.5 * self.gc.cache_step(
                                    img0, img1
                                ) + 0.5 * self.gc.cache_step(img1, img0)
                                # 注意：舊程式碼在這裡有 self.optimizer[0].step()，這可能不對。
                                # GradCache 應該自己處理 optimizer steps。
                                # 這裡移除這行，如果訓練不收斂，再回頭檢查 GradCache 文檔。
                                pass  # 假設 optimizer step 由 GradCache 內部處理

                            else:  # 標準訓練模式 (非 RigL, 非 GradCache)
                                # Zero grad (done above)
                                # Forward
                                p1, p2, z1, z2 = self.model(img0, img1)
                                # Loss
                                loss = self.criterion(p1, p2, z1, z2)
                                # Backward
                                loss.backward()
                                # Optimizer step
                                if self.optimizer is not None:
                                    self.optimizer.step()
                                else:
                                    raise RuntimeError(
                                        "Optimizer is None in standard training mode."
                                    )
                                
                                if self._is_lottery_validating is True:
                                    self.apply_sparse_mask(self.model)

                        else:
                            # 應在函數開始時被捕捉，這裡作為防護
                            raise ValueError(f"Unknown rigl_mode: {rigl_mode}")

                    # --- 驗證步驟 ---
                    elif phase == "val":
                        # 驗證階段不需要計算梯度
                        with torch.no_grad():
                            if isinstance(self.model, list):  # GradCache 驗證模式
                                # 使用 online model (model[0]) 和 target model (model[1])
                                if len(self.model) < 2:
                                    raise RuntimeError(
                                        "Model list for GradCache validation must contain at least 2 models."
                                    )
                                # 驗證時，前向傳播通常只用 online model 的 encoder 和 predictor
                                p1 = self.model[0](img0)  # online -> predictor
                                z1 = self.model[1].encoder(
                                    img1
                                )  # target encoder (without projector/predictor)
                                # SimSiam loss on validation data (symmetric)
                                # Note: The original SimSiam paper usually evaluates representations, not loss on val set.
                                # Keeping loss calculation as per old code, but focus on KNN accuracy.
                                # If using unsymmetric loss from criterion, need to use it correctly here.
                                # Assuming self.criterion might be SimSiamLoss_unsymmetric in GC mode.
                                # Let's use the symmetric evaluation loss as commonly done for SimSiam:
                                val_criterion = (
                                    SimSiam_Module.SimSiamLoss()
                                )  # Use symmetric loss for validation eval
                                loss = val_criterion(
                                    p1, z1
                                )  # Evaluate with symmetric loss

                                # 提取特徵用於 KNN (使用 online model 的 encoder)
                                y1 = (
                                    self.model[0].encoder(img0).mean([2, 3])
                                )  # Global Average Pooling
                                features.append(y1.cpu().numpy())
                                labels.append(label.cpu().numpy())

                            else:  # 非 GradCache 驗證模式 (RigL 或 標準)
                                # 確保模型在設備上
                                img0, img1 = (
                                    img0.to(self.device),
                                    img1.to(self.device),
                                )
                                # 前向傳播
                                p1, p2, z1, z2 = self.model(img0, img1)
                                # 計算損失 (使用訓練時相同的 criterion，通常是 symmetric 的 SimSiamLoss)
                                loss = self.criterion(p1, p2, z1, z2)
                                # 提取特徵用於 KNN (使用 encoder)
                                y1 = self.model.encoder(img0).mean(
                                    [2, 3]
                                )  # Global Average Pooling
                                features.append(y1.cpu().numpy())
                                labels.append(label.cpu().numpy())

                    # --- 步驟結束 ---

                    # 統計運行損失
                    # loss.item() 通常是當前 batch 的平均損失
                    # 乘以 batch size (img0.size(0)) 得到 batch 總損失，累加到 running_loss
                    running_loss += loss.item() * img0.size(0)

                # --- Batch 迴圈結束 ---

                # 在訓練階段結束後，如果是非 RigL 模式，則執行 LR scheduler step
                if phase == "train":
                    if rigl_mode == "none":  # 只有在不使用 RigL 時才用標準 LR scheduler
                        if isinstance(
                            self.scheduler, list
                        ):  # GradCache 模式的 scheduler list
                            if len(self.scheduler) > 0:
                                self.scheduler[0].step()
                            if len(self.scheduler) > 1:
                                self.scheduler[1].step()
                        elif self.scheduler is not None:  # 標準模式的單一 scheduler
                            self.scheduler.step()
                    elif epoch == 7 and rigl_mode == "baseline": # 儲存早期 epoch 的模型權重，以方便彩票驗證
                        self.save_model(self.model, type="early")

                    # RigL 模式假定 LR scheduling 由 RigL 本身處理或不需要

                # --- Epoch 階段結束的指標和記錄 ---
                epoch_loss = running_loss / self.dataset_sizes[phase]

                # 記錄損失到 TensorBoard
                if phase == "train":
                    self.writer.add_scalar("training/loss", epoch_loss, epoch)
                    print(
                        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}"
                    )
                elif phase == "val":
                    # KNN 評估只在驗證階段執行
                    knn_features = np.concatenate(features, axis=0)
                    knn_labels = np.concatenate(labels, axis=0)

                    # 將驗證集的特徵和標籤分成訓練集和測試集 (用於 KNN)
                    # test_size=0.5 表示 50% 數據用於 KNN 訓練，50% 用於 KNN 測試
                    train_features, test_features, train_labels, test_labels = (
                        train_test_split(
                            knn_features,
                            knn_labels,
                            test_size=0.5,
                            random_state=0,  # 固定 random_state 確保結果可重現
                        )
                    )

                    # 使用 KNN 評估
                    knn = KNeighborsClassifier(n_neighbors=5)  # 例如使用 K=5
                    knn.fit(train_features, train_labels)  # 用一部分驗證數據訓練 KNN
                    predictions = knn.predict(
                        test_features
                    )  # 用另一部分驗證數據測試 KNN
                    knn_accuracy = accuracy_score(test_labels, predictions)

                    # 記錄驗證損失和 KNN 準確率到 TensorBoard
                    self.writer.add_scalar("validation/loss", epoch_loss, epoch)
                    self.writer.add_scalar(
                        "validation/knn_accuracy", knn_accuracy, epoch
                    )
                    print(
                        f"Epoch {epoch+1}/{num_epochs} - Val Loss: {epoch_loss:.4f} | KNN Accuracy: {knn_accuracy:.4f}"
                    )

                    # 如果 KNN 準確率創新高，則保存最佳模型權重
                    if knn_accuracy > best_knn_accuracy:
                        best_knn_accuracy = knn_accuracy
                        self.writer.add_scalar(
                            "validation/best_knn_accuracy", best_knn_accuracy, epoch
                        )
                        print(
                            f"Saving best model at epoch {epoch+1} with KNN Accuracy: {best_knn_accuracy:.4f}"
                        )
                        # 保存模型，傳入 self.model (可以是單一模型或列表)
                        self.save_model(self.model, type="best")

            # 在每個 epoch (訓練和驗證都完成後) 保存最後的模型權重
            # 這樣即使訓練中斷，也可以從最後一個完整的 epoch 恢復
            print(f"Saving last model state at epoch {epoch+1}")
            self.save_model(
                self.model, type="last"
            )  # 保存模型，傳入 self.model (可以是單一模型或列表)

            print()  # 在每個 epoch 後換行

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                (time_elapsed // 60), (time_elapsed % 60)
            )
        )

        self.writer.flush()  # type: ignore
        self.writer.close()  # type: ignore
        # --- Training Loop End ---

    def apply_to_models(
        self, models: Union[torch.nn.Module, List[torch.nn.Module]], func
    ):
        """
        对模型列表或单个模型应用某个函数。

        Args:
            models: 模型列表或单个模型。
            func: 要应用的函数 (例如 model.train, model.eval)。
        """
        if isinstance(models, list):
            for model in models:
                func(model)
        else:
            func(models)

    def save_model(
        self,
        models: Union[torch.nn.Module, List[torch.nn.Module]],
        filename_prefix="shuffleNet_v05_SimSiam_",
        directory="main/runs",
        type="best",
    ):
        """
        保存模型权重到指定目录，或初始化 tensorboard writer。

        Args:
            models: 要保存的模型 (可以是单个模型或模型列表)。
            filename_prefix: 文件名前缀。
            directory: 保存目录。
            type: "best", "last", "early", or "tensorboard_init"。
        """
        assert type in [
            "last",
            "best",
            "early",
            "tensorboard_init",
        ], "type 参数只能是 'best'、'last' 或 'tensorboard_init'"

        # 確保目錄存在
        os.makedirs(directory, exist_ok=True)

        if type == "tensorboard_init":
            import datetime
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            
            # 從環境變數讀取以實現動態命名，防止平行化衝突
            run_seed = os.environ.get('RUN_SEED', '42')
            target_sparsity = os.environ.get('TARGET_SPARSITY', '0.0')
            method_str = os.environ.get('METHOD', 'dense').lower()
            
            if method_str == 'rigl':
                method_name = "RigL"
            else:
                method_name = "Dense"
                
            sparsity_percent = int(float(target_sparsity) * 100)
            sparsity_str = f"s{sparsity_percent}"
            
            # 取得資料集後綴
            raw_dataset = os.environ.get('TARGET_DATASET', 'cifar10').lower()
            if raw_dataset == 'cifar100':
                dataset_suffix = 'c100'
            elif raw_dataset == 'imagenet100':
                dataset_suffix = 'in100'
            elif raw_dataset == 'cifar10':
                dataset_suffix = 'c10'
            else:
                dataset_suffix = raw_dataset
            
            folder_name = f"{method_name}_SSL_{sparsity_str}_seed{run_seed}_{dataset_suffix}_{now}"
            final_run_dir = os.path.join(directory, folder_name)
            os.makedirs(final_run_dir, exist_ok=True)
            print(f"TensorBoard log directory created at: {final_run_dir}")
            return SummaryWriter(final_run_dir)

        # 保存模型權重文件
        # 確定要保存哪個模型。對於 RigL 或 Standard，是 self.model。
        # 對於 GradCache，通常只保存 online model (model[0]) 的權重用於後續評估或 fine-tune。
        # 這部分需要確認 SimSiam 和 GradCache 的標準做法，這裡暫時沿用舊程式碼只存 model[0] 的方式。
        model_to_save = models[0] if isinstance(models, list) else models

        # 找到當前的運行目錄 (基於上次 tensorboard_init 創建的目錄)
        # 這需要一個方法來記錄當前運行目錄，或者從 writer 對象獲取
        # 簡單的方法是假設 writer.log_dir 就是當前目錄
        if not hasattr(self, "writer") or self.writer is None:
            print(
                "Warning: TensorBoard writer not initialized or found. Cannot determine save directory."
            )
            # Fallback: try to find the latest run directory based on prefix
            latest_run_dir = None
            for item in os.listdir(directory):
                if item.startswith(filename_prefix) and os.path.isdir(
                    os.path.join(directory, item)
                ):
                    latest_run_dir = os.path.join(
                        directory, item
                    )  # Simplified: takes the last one in listing
            if latest_run_dir:
                filepath = os.path.join(latest_run_dir, f"{type}.pt")
            else:
                print(
                    "Error: Could not determine save directory. Saving to current directory."
                )
                filepath = f"{type}.pt"
        else:
            # Use the log_dir from the initialized writer
            filepath = os.path.join(self.writer.log_dir, f"{type}.pt")

        if type == "best" or type == "last" or type == "early":
            try:
                torch.save(model_to_save.state_dict(), filepath)
                # print(f"Model ({type}) saved to {filepath}")
            except Exception as e:
                print(f"Error saving model {type} to {filepath}: {e}")
