# main/src/training/ADS_SSL_train.py

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
import time
import torch
import os
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

from src.processing.CIFAR10 import CIFAR10_Dataset
# 1. 導入我們設計的新模組
from src.module.Semi_Sparse_SimSiam_Module import SparseSimSiam

class ADS_SSL_Model:
    def __init__(
        self,
        pretrained_model_class=models.shufflenet_v2_x0_5,
        pretrained_weight=None,
        load_weight: str = "",
        base_lr=0.03,
        load_w_params: bool = True,
        load_s_params: bool = False,
    ) -> None:
        self.base_lr = base_lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = pretrained_weight
        self.pretrained_model = pretrained_model_class(weights=self.weights)

        # SimSiam 需要的資料增強 (保持不變)
        self.data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop((224, 224), scale=(0.2, 1)),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            ),
        }

        # 2. 初始化我們的 SparseSimSiam 模型
        self.model = SparseSimSiam(self.pretrained_model).to(self.device)

        self.load_s_params = load_s_params
        self.load_w_params = load_w_params
        if load_weight != "":
            print(f"Loading weights from {load_weight}")
            state_dict = torch.load(load_weight)
            # 過濾掉 s_params，只載入模型的原始參數
            model_dict = self.model.state_dict()
            if self.load_s_params and self.load_w_params:
                # 全部載入
                pretrained_dict = {k: v for k, v in state_dict.items()}
            elif self.load_s_params and not self.load_w_params:
                # 只載入 s_params
                pretrained_dict = {k: v for k, v in state_dict.items() if 's_params' in k}
            else:
                # 只載入 w_params
                pretrained_dict = {k: v for k, v in state_dict.items() if 's_params' not in k}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
            print("Filtered model weights loaded.")

        print("Loaded pretrained model base:", self.pretrained_model.__class__.__name__)
        print("Initialized ADS-SSL Model (SparseSimSiam).")
        print("Use device:", self.device)

    def dataset_initialize(self, DATASET_DIR=".\\LabelTool", BATCH_SIZE=64, WORKERS=0):
        # (這部分與原程式碼完全相同)
        self.data_dir = DATASET_DIR
        self.image_datasets = {
            x: CIFAR10_Dataset(split=x, transform=self.data_transforms[x])
            for x in ["train", "val"]
        }
        self.dataloaders = {
            x: DataLoader(
                self.image_datasets[x],
                batch_size=BATCH_SIZE,
                shuffle=True if x == "train" else False,
                num_workers=WORKERS,
                pin_memory=True,
            )
            for x in ["train", "val"]
        }
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ["train", "val"]}
        print("Dataset sizes:", self.dataset_sizes)

    def train(
        self,
        num_epochs=25,
        batch_size=64,
        workers=0,
        dataset_dir=".\\LabelTool",
        # --- ADS-SSL Specific Parameters ---
        lambda_val: float = 1e-5,          # L1 稀疏正則化權重
        mask_update_freq: int = 100,      # 遮罩參數 's' 的更新頻率 (T)
        alpha_initial: float = 1.0,       # Alpha 退火初始值
        alpha_final: float = 50.0,        # Alpha 退火最終值
        lr_weights: Optional[float] = None, # 權重優化器的學習率
        lr_mask: float = 0.01,            # 遮罩優化器的學習率
        momentum: float = 0.5,            # 動量編碼器的動量
    ):
        # --- Dataset Initialization ---
        self.dataset_initialize(
            DATASET_DIR=dataset_dir, BATCH_SIZE=batch_size, WORKERS=workers
        )

        # 如果沒有指定權重學習率，則使用 SimSiam 的標準縮放規則
        if lr_weights is None:
            lr_weights = self.base_lr * batch_size / 256

        # --- Optimizer and Scheduler Initialization ---
        # 3. 創建兩個優化器，分別給權重和遮罩
        w_params = [p for name, p in self.model.named_parameters() if 's_params' not in name]
        self.optimizer_w = SGD(
            w_params,
            lr=lr_weights,
            momentum=0.9,
            weight_decay=5e-4,
        )

        self.optimizer_s = Adam(
            self.model.s_params.parameters(),
            lr=lr_mask
        )

        # 權重的學習率調度器
        self.scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_w, T_max=num_epochs)

        # --- Tensorboard writer initialization ---
        self.writer = self.save_model(self.model, type="tensorboard_init")
        assert isinstance(self.writer, SummaryWriter), "TensorBoard writer initialization failed"

        # --- save model init weight for later lottery rewinding ---
        self.save_model(self.model, type="init")

        since = time.time()
        best_knn_accuracy = -1.0

        # --- Hparam Logging ---
        hparam_dict = {
            "method": "ADS-SSL",
            "pretrained_model": self.pretrained_model.__class__.__name__,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lambda_val": lambda_val,
            "mask_update_freq": mask_update_freq,
            "alpha_initial": alpha_initial,
            "alpha_final": alpha_final,
            "lr_weights": lr_weights,
            "lr_mask": lr_mask,
            "momentum_ema": momentum,
            "optimizer_w": self.optimizer_w.__class__.__name__,
            "optimizer_s": self.optimizer_s.__class__.__name__,
        }
        self.writer.add_hparams(hparam_dict, {})
        self.writer.flush()

        # --- Training Loop ---
        total_steps = num_epochs * len(self.dataloaders["train"])
        
        # [新增] 在初始化時計算一次總參數數量
        total_mask_elements = 0
        for s in self.model.s_params.values():
            total_mask_elements += s.numel()
        self.total_mask_elements = total_mask_elements

        for epoch in tqdm(range(num_epochs), unit="epochs", dynamic_ncols=True):
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()
                    # [修正] val_features 和 val_labels 移到循環外
                    val_features_hardmask = []
                    val_features_softmask = []
                    val_labels = []

                running_loss = 0.0
                running_loss_sim = 0.0
                running_loss_l1 = 0.0

                dataloader_iterator = tqdm(self.dataloaders[phase], unit="batchs", leave=False, dynamic_ncols=True)
                for i, (img0, img1, label) in enumerate(dataloader_iterator):
                    img0, img1, label = img0.to(self.device), img1.to(self.device), label.to(self.device)

                    if phase == "train":
                        global_step = epoch * len(self.dataloaders["train"]) + i
                        progress = min(global_step / total_steps, 1.0)
                        current_alpha = alpha_initial + (alpha_final - alpha_initial) * progress

                        self.optimizer_w.zero_grad()
                        self.optimizer_s.zero_grad()

                        # 模型前向傳播
                        p1, z2 = self.model(img0, img1, alpha=current_alpha, momentum=momentum)
                        
                        # 手動計算損失
                        loss_sim = -(F.normalize(p1, dim=1) * F.normalize(z2, dim=1)).sum(dim=1).mean()
                        
                        # 將 loss_l1 初始化為一個在正確設備上的 0 維張量
                        loss_l1 = torch.tensor(0.0, device=self.device)
                        for s in self.model.s_params.values():
                            m = torch.sigmoid(current_alpha * s)
                            loss_l1 += torch.norm(m, p=1)
                            
                        # [修正] 歸一化 L1 損失
                        normalized_loss_l1 = loss_l1 / self.total_mask_elements
                        total_loss = loss_sim + lambda_val * normalized_loss_l1

                        total_loss.backward()
                        self.optimizer_w.step()
                        self.optimizer_s.step()
                        '''
                        # 將目標網路的權重更新到線上網路上

                        if global_step % mask_update_freq == 0:
                            # self.optimizer_s.step()
                            # self.optimizer_s.zero_grad()
                            self.model._apply_target_w_params_to_online()
                            # # [新增] 更新遮罩 EMA
                            # self.model.update_target_network_mask_ema()
                        '''
                        loss = total_loss
                        running_loss_sim += loss_sim.item() * img0.size(0)
                        running_loss_l1 += normalized_loss_l1.item() * img0.size(0)

                    elif phase == "val":
                        with torch.no_grad():
                            # [修正] 分兩步執行：先計算損失，再提取特徵
                            
                            # 1. 計算驗證損失
                            # 調用訓練時的 forward 方法來獲取 p1 和 z2
                            p1_val, z2_val = self.model(img0, img1, alpha=alpha_final, momentum=momentum)
                            loss = -(F.normalize(p1_val, dim=1) * F.normalize(z2_val, dim=1)).sum(dim=1).mean()

                            # 2. 提取用於 KNN 的特徵
                            # 調用 inference 方法
                            features_batch_hardmask = self.model.inference(img0, already_hard_mask=False, dense=False)
                            features_batch_softmask = self.model.inference(img0, already_hard_mask=False, dense=True)
                            val_features_hardmask.append(features_batch_hardmask.cpu().numpy())
                            val_features_softmask.append(features_batch_softmask.cpu().numpy())
                            val_labels.append(label.cpu().numpy())

                    running_loss += loss.item() * img0.size(0)

                # --- Epoch 階段結束的指標和記錄 ---
                epoch_loss = running_loss / self.dataset_sizes[phase]

                if phase == "train":
                    epoch_loss_sim = running_loss_sim / self.dataset_sizes[phase]
                    epoch_loss_l1 = running_loss_l1 / self.dataset_sizes[phase]

                    self.writer.add_scalar("training/loss_total", epoch_loss, epoch)
                    self.writer.add_scalar("training/loss_sim", epoch_loss_sim, epoch)
                    self.writer.add_scalar("training/loss_l1", epoch_loss_l1, epoch)
                    self.writer.add_scalar("training/learning_rate_w", self.optimizer_w.param_groups[0]['lr'], epoch)

                    # [可選] 在每個 epoch 結束時也記錄一次最終的稀疏度
                    # 這可以提供一個更平滑的 epoch-level 視圖
                    with torch.no_grad():
                        total_elements_epoch = 0
                        non_zero_elements_epoch = 0
                        for s in self.model.s_params.values():
                            m = torch.sigmoid(current_alpha * s) # 使用該 epoch 最後的 alpha
                            hard_mask = (m > 0.5).float()
                            total_elements_epoch += hard_mask.numel()
                            non_zero_elements_epoch += hard_mask.sum().item()
                        
                        if total_elements_epoch > 0:
                            epoch_sparsity = 1.0 - (non_zero_elements_epoch / total_elements_epoch)
                        else:
                            epoch_sparsity = 0.0
                        
                        self.writer.add_scalar("training/epoch_sparsity", epoch_sparsity, epoch)
                        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} | Sparsity: {epoch_sparsity:.4f}")

                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} (Sim: {epoch_loss_sim:.4f}, L1: {epoch_loss_l1:.4f})")
                
                elif phase == "val":
                    knn_features_hardmask = np.concatenate(val_features_hardmask, axis=0)
                    knn_features_softmask = np.concatenate(val_features_softmask, axis=0)
                    knn_features_dict = {"hard": knn_features_hardmask, "soft": knn_features_softmask}
                    knn_labels = np.concatenate(val_labels, axis=0)

                    for mask_type, knn_features in knn_features_dict.items():
                        train_features, test_features, train_labels, test_labels = train_test_split(knn_features, knn_labels, test_size=0.5, random_state=0)
                        knn = KNeighborsClassifier(n_neighbors=5)
                        knn.fit(train_features, train_labels)
                        predictions = knn.predict(test_features)
                        knn_accuracy = accuracy_score(test_labels, predictions)

                        if mask_type == "hard":
                            self.writer.add_scalar("validation/loss", epoch_loss, epoch) 
                            self.writer.add_scalar("validation/knn_accuracy", knn_accuracy, epoch)
                            print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {epoch_loss:.4f} | KNN Accuracy: {knn_accuracy:.4f}")

                            if epoch == 6:
                                print(f"Saving early rewinding model at epoch {epoch+1}")
                                self.save_model(self.model, type="early")

                            if knn_accuracy > best_knn_accuracy:
                                best_knn_accuracy = knn_accuracy
                                self.writer.add_scalar("validation/best_knn_accuracy", best_knn_accuracy, epoch)
                                print(f"Saving best model at epoch {epoch+1} with KNN Accuracy: {best_knn_accuracy:.4f}")
                                self.save_model(self.model, type="best")

                        elif mask_type == "soft":
                            self.writer.add_scalar("validation/soft_knn_accuracy", knn_accuracy, epoch)
                            print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {epoch_loss:.4f} | dense KNN Accuracy: {knn_accuracy:.4f}")

            # 在每個 epoch 結束後更新權重的學習率
            self.scheduler_w.step()

            print(f"Saving last model state at epoch {epoch+1}")
            self.save_model(self.model, type="last")
            print()

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        self.writer.flush()
        self.writer.close()

    def rewind_train(
        self,
        num_epochs=150,
        batch_size=128,
        workers=0,
        dataset_dir=".\\LabelTool",
        # --- ADS-SSL Specific Parameters ---
        load_rewind_weight = "",
        lambda_val: float = 1e-5,          # L1 稀疏正則化權重
        mask_update_freq: int = 100,      # 遮罩參數 's' 的更新頻率 (T)
        alpha_initial: float = 9999.0,       # Alpha 退火初始值
        alpha_final: float = 9999.0,        # Alpha 退火最終值
        lr_weights: Optional[float] = None, # 權重優化器的學習率
        lr_mask: float = 0.01,            # 遮罩優化器的學習率
        momentum: float = 0.0,            # 動量編碼器的動量
        threshold = 0.5,
        using_hard_mask: bool = False,     # 是否使用硬遮罩
    ):
        '''
        將前面訓練好的稀疏結構重新初始化參數，並重新訓練以驗證是否為彩票模型
        '''
        # --- Dataset Initialization ---
        self.dataset_initialize(
            DATASET_DIR=dataset_dir, BATCH_SIZE=batch_size, WORKERS=workers
        )
        self.sparsity_threshold = threshold
        
        self.model.reinitialize_target_s_params()

        # --- load rewinding weight ---
        if load_rewind_weight != "":
            print(f"Loading rewinding weights from {load_rewind_weight}")
            state_dict = torch.load(load_rewind_weight)
            # 過濾掉 s_params，只載入模型的原始參數
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if 's_params' not in k}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)

            print("Filtered model weights loaded.")

        # 檢查使用硬遮罩時，s是否已經是硬遮罩(0/1)的值，只能是0或1，若有其他值則報錯
        if using_hard_mask:
            for s in self.model.s_params.values():
                s_unique_values = torch.unique(s)
                if len(s_unique_values) != 2 or not (s_unique_values[0] == 0.0 and s_unique_values[1] == 1.0):
                    raise ValueError("Using hard mask but s is not 0 or 1")
            if self.sparsity_threshold != 0.5:
                print(f"Warning: Using hard mask but sparsity threshold is not 0.5 (instead {self.sparsity_threshold}), threshold set to 0.5")
                self.sparsity_threshold = 0.5

        # 如果沒有指定權重學習率，則使用 SimSiam 的標準縮放規則
        if lr_weights is None:
            lr_weights = self.base_lr * batch_size / 256

        # --- Optimizer and Scheduler Initialization ---
        # 3. 創建兩個優化器，分別給權重和遮罩
        w_params = [p for name, p in self.model.named_parameters() if 's_params' not in name]
        self.optimizer_w = SGD(
            w_params,
            lr=lr_weights,
            momentum=0.9,
            weight_decay=5e-4,
        )

        # 權重的學習率調度器
        self.scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_w, T_max=num_epochs)

        # --- Tensorboard writer initialization ---
        self.writer = self.save_model(self.model, type="tensorboard_init")
        assert isinstance(self.writer, SummaryWriter), "TensorBoard writer initialization failed"

        since = time.time()
        best_knn_accuracy = -1.0

        # --- Hparam Logging ---
        hparam_dict = {
            "method": "ADS-SSL_rewind",
            "pretrained_model": self.pretrained_model.__class__.__name__,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lambda_val": lambda_val,
            "mask_update_freq": mask_update_freq,
            "alpha_initial": alpha_initial,
            "alpha_final": alpha_final,
            "lr_weights": lr_weights,
            "lr_mask": lr_mask,
            "momentum_ema": momentum,
            "optimizer_w": self.optimizer_w.__class__.__name__
        }
        self.writer.add_hparams(hparam_dict, {})
        self.writer.flush()

        # --- Training Loop ---
        total_steps = num_epochs * len(self.dataloaders["train"])
        
        # [新增] 在初始化時計算一次總參數數量
        total_mask_elements = 0
        for s in self.model.s_params.values():
            total_mask_elements += s.numel()
        self.total_mask_elements = total_mask_elements

        for epoch in tqdm(range(num_epochs), unit="epochs", dynamic_ncols=True):
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()
                    # [修正] val_features 和 val_labels 移到循環外
                    val_features_hardmask = []
                    val_labels = []

                running_loss = 0.0
                running_loss_sim = 0.0

                dataloader_iterator = tqdm(self.dataloaders[phase], unit="batchs", leave=False, dynamic_ncols=True)
                for i, (img0, img1, label) in enumerate(dataloader_iterator):
                    img0, img1, label = img0.to(self.device), img1.to(self.device), label.to(self.device)

                    if phase == "train":
                        global_step = epoch * len(self.dataloaders["train"]) + i
                        progress = min(global_step / total_steps, 1.0)
                        current_alpha = alpha_initial + (alpha_final - alpha_initial) * progress

                        self.optimizer_w.zero_grad()

                        # 模型前向傳播
                        p1, z2 = self.model(img0, img1, alpha=current_alpha, momentum=momentum, using_hard_mask=using_hard_mask)
                        
                        # 手動計算損失
                        loss_sim = -(F.normalize(p1, dim=1) * F.normalize(z2, dim=1)).sum(dim=1).mean()
                        total_loss = loss_sim

                        total_loss.backward()
                        self.optimizer_w.step()
                        
                        loss = total_loss
                        running_loss_sim += loss_sim.item() * img0.size(0)

                    elif phase == "val":
                        with torch.no_grad():
                            # [修正] 分兩步執行：先計算損失，再提取特徵
                            
                            # 1. 計算驗證損失
                            # 調用訓練時的 forward 方法來獲取 p1 和 z2
                            p1_val, z2_val = self.model(img0, img1, alpha=alpha_final, momentum=momentum, using_hard_mask=using_hard_mask)
                            loss = -(F.normalize(p1_val, dim=1) * F.normalize(z2_val, dim=1)).sum(dim=1).mean()

                            # 2. 提取用於 KNN 的特徵
                            # 調用 inference 方法
                            features_batch_hardmask = self.model.inference(img0, already_hard_mask=True)
                            val_features_hardmask.append(features_batch_hardmask.cpu().numpy())
                            val_labels.append(label.cpu().numpy())

                    running_loss += loss.item() * img0.size(0)

                # --- Epoch 階段結束的指標和記錄 ---
                epoch_loss = running_loss / self.dataset_sizes[phase]

                if phase == "train":
                    epoch_loss_sim = running_loss_sim / self.dataset_sizes[phase]

                    self.writer.add_scalar("training/loss_total", epoch_loss, epoch)
                    self.writer.add_scalar("training/loss_sim", epoch_loss_sim, epoch)
                    self.writer.add_scalar("training/learning_rate_w", self.optimizer_w.param_groups[0]['lr'], epoch)

                    # [可選] 在每個 epoch 結束時也記錄一次最終的稀疏度
                    # 這可以提供一個更平滑的 epoch-level 視圖
                    with torch.no_grad():
                        total_elements_epoch = 0
                        non_zero_elements_epoch = 0
                        for s in self.model.s_params.values():
                            m = torch.sigmoid(current_alpha * s) # 使用該 epoch 最後的 alpha
                            hard_mask = (m > self.sparsity_threshold).float()
                            total_elements_epoch += hard_mask.numel()
                            non_zero_elements_epoch += hard_mask.sum().item()
                        
                        if total_elements_epoch > 0:
                            epoch_sparsity = 1.0 - (non_zero_elements_epoch / total_elements_epoch)
                        else:
                            epoch_sparsity = 0.0
                        
                        self.writer.add_scalar("training/epoch_sparsity", epoch_sparsity, epoch)
                        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} | Sparsity: {epoch_sparsity:.4f}")

                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} (Sim: {epoch_loss_sim:.4f})")
                
                elif phase == "val":
                    knn_features_hardmask = np.concatenate(val_features_hardmask, axis=0)
                    knn_features_dict = {"hard": knn_features_hardmask}
                    knn_labels = np.concatenate(val_labels, axis=0)

                    for mask_type, knn_features in knn_features_dict.items():
                        train_features, test_features, train_labels, test_labels = train_test_split(knn_features, knn_labels, test_size=0.5, random_state=0)
                        knn = KNeighborsClassifier(n_neighbors=5)
                        knn.fit(train_features, train_labels)
                        predictions = knn.predict(test_features)
                        knn_accuracy = accuracy_score(test_labels, predictions)

                        if mask_type == "hard":
                            self.writer.add_scalar("validation/loss", epoch_loss, epoch) 
                            self.writer.add_scalar("validation/knn_accuracy", knn_accuracy, epoch)
                            print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {epoch_loss:.4f} | KNN Accuracy: {knn_accuracy:.4f}")

                            if knn_accuracy > best_knn_accuracy:
                                best_knn_accuracy = knn_accuracy
                                self.writer.add_scalar("validation/best_knn_accuracy", best_knn_accuracy, epoch)
                                print(f"Saving best model at epoch {epoch+1} with KNN Accuracy: {best_knn_accuracy:.4f}")
                                self.save_model(self.model, type="best")

                        elif mask_type == "soft":
                            self.writer.add_scalar("validation/soft_knn_accuracy", knn_accuracy, epoch)
                            print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {epoch_loss:.4f} | dense KNN Accuracy: {knn_accuracy:.4f}")

            # 在每個 epoch 結束後更新權重的學習率
            self.scheduler_w.step()

            print(f"Saving last model state at epoch {epoch+1}")
            self.save_model(self.model, type="last")
            print()

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        self.writer.flush()
        self.writer.close()


    def varience_sparsity_eval(
        self,
        batch_size=128,
        workers=0,
        dataset_dir=".\\LabelTool",
        alpha=1.0,
        # --- ADS-SSL Specific Parameters ---
    ):
        # --- Dataset Initialization ---
        self.dataset_initialize(
            DATASET_DIR=dataset_dir, BATCH_SIZE=batch_size, WORKERS=workers
        )
        self.alpha = alpha

        last_sparsity = -1
        for threshold in range(0, 1001):
            self.model.sparsity_threshold = threshold / 1000.0

            # --- Inference ---
            val_features_hardmask = []
            val_features_softmask = []
            val_labels = []

            # --- Calculate current sparsity ---
            total_elements = 0
            non_zero_elements = 0
            for s in self.model.s_params.values():
                m = torch.sigmoid(self.alpha * s)  # 使用該 epoch 最後的 alpha
                hard_mask = (m > self.model.sparsity_threshold).float()
                total_elements += hard_mask.numel()
                non_zero_elements += hard_mask.sum().item()
            
            if total_elements > 0:
                current_sparsity = 1.0 - (non_zero_elements / total_elements)
            else:
                current_sparsity = 0.0

            if abs(last_sparsity - current_sparsity) <= 0.01:
                continue

            last_sparsity = current_sparsity

            with torch.no_grad():
                for img0, img1, label in self.dataloaders["val"]:
                    img0, img1, label = img0.to(self.device), img1.to(self.device), label.to(self.device)
                    
                    features_batch_hardmask = self.model.inference(img0, already_hard_mask=False, threshold=self.model.sparsity_threshold, alpha=self.alpha)
                    features_batch_softmask = self.model.inference(img0, already_hard_mask=False, dense=True)
                    val_features_hardmask.append(features_batch_hardmask.cpu().numpy())
                    val_features_softmask.append(features_batch_softmask.cpu().numpy())
                    val_labels.append(label.cpu().numpy())

            knn_features_hardmask = np.concatenate(val_features_hardmask, axis=0)
            knn_features_softmask = np.concatenate(val_features_softmask, axis=0)
            knn_features_dict = {"hard": knn_features_hardmask, "soft": knn_features_softmask}
            knn_labels = np.concatenate(val_labels, axis=0)

            for mask_type, knn_features in knn_features_dict.items():
                train_features, test_features, train_labels, test_labels = train_test_split(knn_features, knn_labels, test_size=0.5, random_state=0)
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(train_features, train_labels)
                predictions = knn.predict(test_features)
                knn_accuracy = accuracy_score(test_labels, predictions)

                if mask_type == "hard":
                    print(f"Validation KNN Accuracy (Mask Type: {mask_type}): {knn_accuracy:.5f}, Current Sparsity: {current_sparsity:.9f}, Current Threshold: {self.model.sparsity_threshold}")

    def export_specific_threshold_model(self, threshold):
        self.model.sparsity_threshold = threshold
        # self.model._apply_target_s_params_to_online()

        # 將s_params中經過sigmoid大於特定閾值的轉為1, 小於等於的轉為0
        for s in self.model.s_params.values():
            m = torch.sigmoid(s)  # 使用該 epoch 最後的 alpha
            hard_mask = (m > self.model.sparsity_threshold).float()
            s.data = hard_mask

        # --- Calculate current sparsity ---
        total_elements = 0
        non_zero_elements = 0
        for s in self.model.s_params.values():
            m = s  
            hard_mask = (m > 0).float()
            total_elements += hard_mask.numel()
            non_zero_elements += hard_mask.sum().item()
        
        if total_elements > 0:
            current_sparsity = 1.0 - (non_zero_elements / total_elements)
        else:
            current_sparsity = 0.0

        print(f"Current Sparsity: {current_sparsity:.9f}, Current Threshold: {self.model.sparsity_threshold}")

        self.writer = self.save_model(self.model, type="tensorboard_init")
        self.save_model(self.model, type="best", filename_prefix=f"export_{current_sparsity:.5f}_ADS_SSL_SimSiam_")

    def save_model(
        self,
        model_to_save: torch.nn.Module,
        filename_prefix="ADS_SSL_SimSiam_",
        directory="./runs",
        type="best",
    ):
        """
        保存模型权重到指定目录，或初始化 tensorboard writer。

        Args:
            models: 要保存的模型 (可以是单个模型或模型列表)。
            filename_prefix: 文件名前缀。
            directory: 保存目录。
            type: "init", "early", "last", "best", or "tensorboard_init"。
        """
        # (這部分與原程式碼基本相同，只修改了前綴)
        assert type in ["init", "early", "last", "best", "tensorboard_init"], "type an only be 'init', 'best', 'last', or 'tensorboard_init'"
        os.makedirs(directory, exist_ok=True)

        if type == "tensorboard_init":
            i = 0
            run_dir_name = filename_prefix
            while os.path.exists(os.path.join(directory, run_dir_name + str(i))):
                i += 1
            final_run_dir = os.path.join(directory, run_dir_name + str(i))
            os.makedirs(final_run_dir)
            print(f"TensorBoard log directory created at: {final_run_dir}")
            return SummaryWriter(final_run_dir)

        if not hasattr(self, "writer") or self.writer is None:
            print("Warning: TensorBoard writer not initialized. Cannot determine save directory.")
            filepath = f"{type}.pt"
        else:
            filepath = os.path.join(self.writer.log_dir, f"{type}.pt")

        try:
            # 保存整個模型，包括權重和 s_params
            torch.save(model_to_save.state_dict(), filepath)
        except Exception as e:
            print(f"Error saving model {type} to {filepath}: {e}")
