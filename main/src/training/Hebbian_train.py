import time
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD
from torchvision import transforms
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Optional

# 導入資料集處理 (假設路徑不變)
from src.processing.CIFAR10 import CIFAR10_Dataset
# 導入我們上一部定義好的 Hebbian Sparse SimSiam
# 請確保 src/module/hebbian_SimSiam_Module.py 包含我們之前討論的 Hebbian_SimSiam 類別
from src.module.hebbian_SimSiam_Module import Hebbian_SimSiam
from src.module.hebbian_SimSiam_Module import HebbianSparseLayer

class Hebbian_SSL_Trainer:
    def __init__(
        self,
        pretrained_model_class=models.resnet18,
        pretrained_weight=None,
        load_weight: str = "",
        base_lr=0.03,
        target_sparsity=0.8,
        use_erk=True,          # Ablation Toggle
        protect_highway=True   # Ablation Toggle
    ) -> None:
        self.base_lr = base_lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 準備 Backbone (Dense)
        self.backbone = pretrained_model_class(weights=pretrained_weight)
        model_type = "shufflenet" if pretrained_model_class == models.shufflenet_v2_x0_5 else "resnet"

        # 2. 初始化 Hebbian_SimSiam 模型
        # 這會自動將 conv/linear 層替換為 HebbianSparseLayer
        if model_type == "shufflenet":
            self.model = Hebbian_SimSiam(
                self.backbone, 
                model_type='shufflenet',
                encoder_output_dim=1024,
                target_sparsity=target_sparsity,
                use_erk=use_erk,
                protect_highway=protect_highway
            ).to(self.device)
        elif model_type == "resnet":
            self.model = Hebbian_SimSiam(
                self.backbone, 
                model_type='resnet',
                encoder_output_dim=512,
                projector_inner_dim=2048,
                target_sparsity=target_sparsity,
                use_erk=use_erk,
                protect_highway=protect_highway
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


        # 3. 載入權重 (若有)
        if load_weight != "":
            print(f"Loading weights from {load_weight}")
            # 因為 Hebbian mask 是 buffer，直接 load_state_dict 即可自動處理
            state_dict = torch.load(load_weight, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded.")

        # 資料增強 (SimSiam 標準設定)
        self.data_transforms = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.2, 1)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ]),
            "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ]),
        }

        print("Initialized Hebbian-SSL Model.")
        print(f"Target Sparsity: {target_sparsity}")
        print("Use device:", self.device)

    def dataset_initialize(self, DATASET_DIR, BATCH_SIZE, WORKERS):
        self.image_datasets = {
            x: CIFAR10_Dataset(split=x, transform=self.data_transforms[x])
            for x in ["train", "val"]
        }
        self.dataloaders = {
            x: DataLoader(
                self.image_datasets[x],
                batch_size=BATCH_SIZE,
                shuffle=(x == "train"),
                num_workers=WORKERS,
                pin_memory=True,
                drop_last=(x == "train") # SimSiam 訓練時避免最後一個 batch 太小影響 BN
            )
            for x in ["train", "val"]
        }
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ["train", "val"]}

    def train(
        self,
        num_epochs=100,
        batch_size=64,
        workers=0,
        dataset_dir=".\\LabelTool",
        # --- Hebbian Specific Parameters ---
        init_grow_ratio=0.2,    # 初始結構交換率 (例如 20% 的連接會重生)
        hebbian_freq=50,        # 每幾步計算一次 Pearson Correlation (節省算力)
        warmup_epochs=5,        # 結構固定不變的預熱期 (讓權重先收斂一點)
        momentum=0.9,           # SGD Momentum
        weight_decay=5e-4,      # SGD Weight Decay
    ):
        # 初始化資料
        self.dataset_initialize(dataset_dir, batch_size, workers)
        
        # 計算實際學習率 (Linear Scaling Rule)
        lr = self.base_lr * batch_size / 256
        
        # 優化器：只優化權重 (Mask 由 Hebbian 邏輯控制)
        optimizer = SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
        
        # 學習率排程
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Tensorboard
        self.writer = self.save_model(self.model, type="tensorboard_init")
        
        # 記錄參數
        hparam_dict = {
            "method": "Hebbian-SSL",
            "epochs": num_epochs,
            "bs": batch_size,
            "lr": lr,
            "target_sparsity": self.model.target_sparsity,
            "init_grow_ratio": init_grow_ratio
        }
        self.writer.add_hparams(hparam_dict, {})

        best_knn_acc = 0.0
        start_time = time.time()

        print(f"Start Training... Total Epochs: {num_epochs}")

        for epoch in range(num_epochs):
            self.model.train()
            
            # --- 1. 計算當前 Epoch 的生長率 (Cosine Decay) ---
            # 隨著訓練進行，結構越來越穩定
            # Hebbian 探索期拉長：讓拓樸更新持續到總 Epoch 的 80%
            exploration_epochs = int(num_epochs * 0.8)
            if epoch < warmup_epochs:
                current_grow_ratio = 0.0 # Warmup 期間不改變結構
            elif epoch > exploration_epochs:
                current_grow_ratio = 0.0 # 訓練末期固定結構，純訓練權重
            else:
                # 從 init_grow_ratio 降到 0
                decay_progress = (epoch - warmup_epochs) / (exploration_epochs - warmup_epochs)
                current_grow_ratio = init_grow_ratio * 0.5 * (1 + math.cos(math.pi * decay_progress))

            running_loss = 0.0
            
            pbar = tqdm(self.dataloaders["train"], desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for i, (x1, x2, _) in enumerate(pbar):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                # --- 2. 控制 Hebbian 統計開關 ---
                # 只有在特定的 step 才開啟 hook 計算相關性，大幅提升速度
                # 並且在 warmup 期間或者 grow_ratio 很低時也可以選擇不計算
                should_compute_hebbian = (i % hebbian_freq == 0) and (current_grow_ratio > 0.001)
                self.model.set_hebbian_enable(should_compute_hebbian)

                optimizer.zero_grad()
                
                # SimSiam Forward
                p1, p2, z1, z2 = self.model(x1, x2)
                
                # SimSiam Loss: -(p * z).sum(dim=1).mean()
                loss = -(F.normalize(p1, dim=1) * F.normalize(z2, dim=1)).sum(dim=1).mean() * 0.5 + \
                       -(F.normalize(p2, dim=1) * F.normalize(z1, dim=1)).sum(dim=1).mean() * 0.5
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # --- 3. Epoch 結束：更新拓撲結構 ---
            if current_grow_ratio > 0.001:
                # [診斷] 更新前：拍快照
                old_mask_snapshot = self._capture_mask_snapshot()

                # 根據累積的 Hebbian Score 進行 Prune & Grow
                # Hebbian V5: 回傳變動掩碼以便清除優化器動量
                topology_changes = self.model.update_topology(grow_ratio=current_grow_ratio)

                # Hebbian V5: 清理優化器動量 (Momentum Reset)
                if optimizer.state:
                    for layer, (grow_mask, drop_mask) in topology_changes.items():
                        change_mask = grow_mask | drop_mask
                        # 直接操作 optimizer.state 中的對應權重
                        weight_param = layer.layer.weight
                        if weight_param in optimizer.state:
                            state = optimizer.state[weight_param]
                            if 'momentum_buffer' in state:
                                state['momentum_buffer'][change_mask] = 0.0

                # [診斷] 更新後：計算變動率
                change_rate = self.check_topology_change(old_mask_snapshot)
                
                # [診斷] 記錄到 TensorBoard (非常重要，畫出來才能看到趨勢)
                self.writer.add_scalar("Structure/ChangeRate", change_rate, epoch)
                
                # 更新完後，可以選擇 reset optimizer state 對應新長出來的權重 (RigL 做法)
                # 但 PyTorch SGD 對於 0 值的權重更新也沒問題，這裡保持簡單
            
            # Update Learning Rate
            scheduler.step()

            # --- 3.5 紀錄可解釋性變化
            if epoch % 10 == 0:
                self.check_dead_channels()
                self.check_signal_alignment() # 注意這要在 backward 之後，zero_grad 之前
                self.check_simsiam_collapse()
            
            # --- 4. Logging & Validation ---
            avg_loss = running_loss / len(self.dataloaders["train"])
            self.writer.add_scalar("training/loss", avg_loss, epoch)
            self.writer.add_scalar("training/learning_rate", optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar("Structure/GrowRatio", current_grow_ratio, epoch)

            # 計算當前 Encoder 真實稀疏度並記錄到 TensorBoard
            total_params = 0
            zero_params = 0
            for name, param in self.model.named_parameters():
                if 'encoder' in name and 'weight' in name and 'bn' not in name and 'downsample.1' not in name and param.dim() > 1:
                    param_numel = param.numel()
                    param_zeros = (param.data.abs() < 1e-7).sum().item()
                    total_params += param_numel
                    zero_params += param_zeros
            global_actual_sparsity = zero_params / total_params if total_params > 0 else 0
            self.writer.add_scalar("training/epoch_sparsity", global_actual_sparsity, epoch)

            # 驗證 KNN Accuracy
            val_acc = self.evaluate_knn(epoch)
            self.writer.add_scalar("validation/knn_accuracy", val_acc, epoch)
            
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | KNN Acc: {val_acc:.4f} | Grow: {current_grow_ratio:.4f} | Sparsity: {global_actual_sparsity:.4f}")

            # Save Models
            self.save_model(self.model, type="last")
            if val_acc > best_knn_acc:
                best_knn_acc = val_acc
                self.writer.add_scalar("validation/best_knn_accuracy", best_knn_acc, epoch)
                self.save_model(self.model, type="best")
                print(f"New Best Model! Acc: {best_knn_acc:.4f}")

        total_time = time.time() - start_time
        print(f"Training Finished. Total Time: {total_time/3600:.2f} hours.")
        self.writer.close()

    def evaluate_knn(self, epoch):
        """
        使用 KNN 評估 SimSiam 的特徵品質。
        因為我們全程都是硬稀疏 (Mask=0/1)，所以不需要像 CS 那樣區分 Soft/Hard。
        """
        self.model.eval()
        
        # 為了 KNN 評估，我們需要提取特徵
        # 我們只使用 encoder 的輸出 (projector 前) 或者 projector 的輸出
        # 這裡 SimSiam 通常使用 encoder output (y1) 或 projector output (z1)
        # 為了方便，我們假設 model.encoder 已經被替換成稀疏版本，我們直接用它
        
        def get_features(loader):
            features_list = []
            labels_list = []
            with torch.no_grad():
                for img, _, label in loader:
                    img = img.to(self.device)
                    # 只過 Encoder
                    features = self.model.encoder(img).mean([2, 3]) # Global Avg Pool
                    features = F.normalize(features, dim=1)
                    features_list.append(features.cpu().numpy())
                    labels_list.append(label.numpy())
            return np.concatenate(features_list), np.concatenate(labels_list)

        # 這裡為了效率，我們可以用 validation set 的一部分當 gallery，一部分當 query
        # 或者標準做法：用 train set 當 gallery (記憶體消耗大)，val set 當 query
        # 這裡簡化：用 Val Set 切一半做 KNN (快速驗證用)
        
        features, labels = get_features(self.dataloaders["val"])
        
        # Split for KNN
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=42)
        
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        
        return acc
    
    def validate_lottery(
        self,
        mask_source_path: str,  # 訓練好的模型路徑 (best.pt)
        weight_init_path: Optional[str] = None, # 初始權重路徑 (init.pt)，若為 None 則隨機初始化
        num_epochs=100,
        batch_size=128,
        workers=0,
        dataset_dir=".\\LabelTool",
    ):
        print("=== Starting Lottery Ticket Validation ===")
        
        # 1. 載入訓練好的 Mask
        print(f"Loading mask from {mask_source_path}...")
        trained_state = torch.load(mask_source_path, map_location=self.device)
        
        # 我們只需要載入 buffer (mask)，不需要載入權重
        # 但為了方便，我們先全部載入，稍後再重置權重
        self.model.load_state_dict(trained_state, strict=False)
        
        # 備份 Mask (因為接下來重置權重可能會影響到某些層，雖然 buffer 通常不受影響，但保險起見)
        final_masks = {}
        for name, m in self.model.named_modules():
            if hasattr(m, 'mask'):
                final_masks[name] = m.mask.clone().detach()

        # 2. 重置權重 (Rewinding)
        if weight_init_path and os.path.exists(weight_init_path):
            print(f"Rewinding weights to initialization: {weight_init_path}")
            init_state = torch.load(weight_init_path, map_location=self.device)
            # 這裡我們只載入 parameter (weight, bias)，不載入 buffer (mask)
            # 過濾掉所有包含 'mask' 或 'hebbian_score' 的 key
            clean_init_state = {k: v for k, v in init_state.items() 
                                if 'mask' not in k and 'hebbian_score' not in k}
            self.model.load_state_dict(clean_init_state, strict=False)
        else:
            print("Re-initializing weights randomly (Random Ticket)...")
            # 重新初始化所有 Conv 和 Linear
            def init_weights(m):
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
            self.model.apply(init_weights)

        # 3. 強制恢復 Mask (確保結構是訓練好的那組)
        # 並將權重套用 Mask (因為重置後權重變成了 dense)
        print("Applying fixed mask to reset weights...")
        for name, m in self.model.named_modules():
            if name in final_masks:
                m.mask.copy_(final_masks[name]) # 恢復 mask
                m.enable_hebbian = False       # 關閉 Hebbian 統計
                with torch.no_grad():
                    m.layer.weight.data *= m.mask # 確保初始權重符合稀疏結構

        # 4. 開始訓練 (邏輯與 train 類似，但移除了 update_topology)
        self.dataset_initialize(dataset_dir, batch_size, workers)
        lr = self.base_lr * batch_size / 256
        optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # 建立新的 Log 目錄
        self.writer = self.save_model(self.model, type="tensorboard_init", filename_prefix="Lottery_Check_")

        best_acc = 0.0
        
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            
            pbar = tqdm(self.dataloaders["train"], desc=f"Lottery Epoch {epoch+1}/{num_epochs}", leave=False)
            for x1, x2, _ in pbar:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                optimizer.zero_grad()
                p1, p2, z1, z2 = self.model(x1, x2)
                loss = -(F.normalize(p1, dim=1) * F.normalize(z2, dim=1)).sum(dim=1).mean() * 0.5 + \
                       -(F.normalize(p2, dim=1) * F.normalize(z1, dim=1)).sum(dim=1).mean() * 0.5
                loss.backward()
                
                # [關鍵] 每次 step 都要確保 mask 以外的梯度為 0 (雖然權重已 mask，但 optimizer 有 momentum)
                # 這裡我們可以簡單地再次乘上 mask
                with torch.no_grad():
                    for m in self.model.modules():
                        if hasattr(m, 'mask'):
                            if m.layer.weight.grad is not None:
                                m.layer.weight.grad *= m.mask

                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # [關鍵] 這裡 **不呼叫** update_topology，結構固定
            
            scheduler.step()
            val_acc = self.evaluate_knn(epoch)
            self.writer.add_scalar("Lottery/Val_Acc", val_acc, epoch)
            print(f"Epoch {epoch+1} | Loss: {running_loss/len(self.dataloaders['train']):.4f} | KNN: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model(self.model, type="best", filename_prefix="Lottery_")

        print(f"Lottery Validation Finished. Best Acc: {best_acc:.4f}")
        self.writer.close()

    def save_model(self, model, type="last", filename_prefix="Hebbian_SSL_", directory="./runs"):
        os.makedirs(directory, exist_ok=True)
        
        if type == "tensorboard_init":
            import datetime
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(directory, f"{filename_prefix}{now}")
            os.makedirs(log_dir, exist_ok=True)
            return SummaryWriter(log_dir)
        
        filepath = os.path.join(self.writer.log_dir, f"{type}.pt")
        # 儲存 state_dict，其中包含了 weights (dense parameters) 和 buffers (mask, hebbian_score)
        torch.save(model.state_dict(), filepath)
        return None
    
    def check_dead_channels(self):
        print("\n=== [Diagnostic] Dead Channel Check ===")
        total_dead_layers = 0
        
        for name, m in self.model.named_modules():
            if isinstance(m, HebbianSparseLayer):
                # mask shape: (Out, In, K, K)
                # 檢查輸出通道 (Output Channels) 是否全死
                # view(Out, -1).sum(1) -> 每個 Output Channel 的連接總數
                out_connectivity = m.mask.view(m.mask.size(0), -1).sum(dim=1)
                dead_outputs = (out_connectivity == 0).sum().item()
                
                # 檢查輸入通道 (Input Channels) 是否全死 (對於 Conv2d)
                # view(Out, In, -1).sum(0).sum(-1) -> 每個 Input Channel 的連接總數
                if m.mask.dim() == 4:
                    in_connectivity = m.mask.abs().sum(dim=(0, 2, 3))
                    dead_inputs = (in_connectivity == 0).sum().item()
                else:
                    # Linear
                    in_connectivity = m.mask.abs().sum(dim=0)
                    dead_inputs = (in_connectivity == 0).sum().item()

                total_out = m.mask.size(0)
                total_in = m.mask.size(1)
                
                if dead_outputs > 0 or dead_inputs > 0:
                    print(f"Layer {name}:")
                    print(f"  - Dead Outputs: {dead_outputs}/{total_out} ({dead_outputs/total_out:.2%})")
                    print(f"  - Dead Inputs : {dead_inputs}/{total_in} ({dead_inputs/total_in:.2%})")
                    total_dead_layers += 1
        
        if total_dead_layers == 0:
            print(">> All layers are healthy (No fully disconnected channels).")
        else:
            print(f">> WARNING: {total_dead_layers} layers have disconnected channels!")

    def check_signal_alignment(self):
        print("\n=== [Diagnostic] Signal Alignment Check ===")
        # 需要先清空梯度並計算一次 Backward 才能拿到 grad
        # 這裡假設是在 optimizer.step() 之前或剛 backward 完調用
        
        correlations = []
        
        for name, m in self.model.named_modules():
            if isinstance(m, HebbianSparseLayer) and m.layer.weight.grad is not None:
                # 只看 Mask=0 (潛在連接) 的部分，因為這是生長決策的依據
                # 或者看 Mask=1 (現有連接) 的部分，看剪枝決策
                
                # 這裡我們看整體的趨勢
                with torch.no_grad():
                    grad_mag = m.layer.weight.grad.abs().view(-1)
                    hebbian = m.hebbian_score.view(-1)
                    
                    # 簡單計算 Pearson 相關係數
                    # (雖然 Rank Correlation 更好，但 Pearson 夠快)
                    if grad_mag.std() > 0 and hebbian.std() > 0:
                        # Stack 並計算 corrcoef
                        stack = torch.stack([grad_mag, hebbian])
                        corr = torch.corrcoef(stack)[0, 1].item()
                        correlations.append(corr)
                        
                        # 如果相關性極低或負的，印出來
                        if corr < 0.0: 
                            print(f"Layer {name}: Correlation = {corr:.4f} (NEGATIVE ALIGNMENT!)")
        
        if len(correlations) > 0:
            avg_corr = sum(correlations) / len(correlations)
            print(f">> Average Gradient-Hebbian Correlation: {avg_corr:.4f}")
        else:
            print(">> No gradients found to compare.")

    def check_simsiam_collapse(self):
        self.model.eval()
        print("\n=== [Diagnostic] SimSiam Feature Collapse Check ===")
        
        # 拿一個 batch
        try:
            x1, _, _ = next(iter(self.dataloaders['train']))
            x1 = x1.to(self.device)
            
            with torch.no_grad():
                # 假設 model forward 回傳 p1, p2, z1, z2
                # 我們看 z1 (Projector output)
                y1 = self.model.encoder(x1).mean([2, 3]) # Encoder output
                z1 = self.model.projector(y1)            # Projector output
                
                # 歸一化前的 std
                z1_std = F.normalize(z1, dim=1).std(dim=0).mean().item()
                
                print(f">> Output Std (normalized): {z1_std:.4f}")
                
                # SimSiam 論文指出，如果 std < 1/sqrt(d)，通常就是坍塌了
                # 這裡 d 是 projector output dim (例如 2048)
                # 1/sqrt(2048) approx 0.022
                if z1_std < 0.01:
                    print(">> CRITICAL WARNING: Model has likely COLLAPSED!")
        except Exception as e:
            print(f"Skip collapse check due to error: {e}")

    def _capture_mask_snapshot(self):
        """
        [輔助函數] 獲取當前所有稀疏層的 Mask 快照
        """
        snapshot = {}
        for name, m in self.model.named_modules():
            # 判斷是否為我們的稀疏層 (根據你的 import 修改類別名稱)
            if hasattr(m, 'mask') and isinstance(m.mask, torch.Tensor):
                # 必須使用 .clone().detach()，否則會存到參考(Reference)而不是數值
                snapshot[name] = m.mask.clone().detach()
        return snapshot

    def check_topology_change(self, old_masks_snapshot):
        """
        [診斷工具] 計算 Mask 變動率 (Topology Turnover Rate)
        Args:
            old_masks_snapshot: 由 _capture_mask_snapshot() 產生的字典
        """
        total_changed_params = 0
        total_params = 0
        
        # 用來記錄各層的變動情況 (可選)
        layer_stats = []

        for name, m in self.model.named_modules():
            if name in old_masks_snapshot:
                old_mask = old_masks_snapshot[name]
                new_mask = m.mask
                
                # 計算差異：XOR 運算 (原本是0變1，或原本是1變0)
                # abs(new - old) 在 0/1 mask 下等同於 XOR
                diff = torch.abs(new_mask - old_mask).sum().item()
                numel = new_mask.numel()
                
                total_changed_params += diff
                total_params += numel
                
                # 如果該層變動很大，可以記錄下來
                if diff > 0:
                    layer_rate = diff / numel
                    layer_stats.append((name, layer_rate))

        if total_params == 0:
            return 0.0

        global_change_rate = total_changed_params / total_params
        
        print(f"\n=== [Diagnostic] Topology Change Rate: {global_change_rate:.4%} ===")
        # 如果你想看細節，可以取消下面這行的註解
        # for lname, lrate in layer_stats[:5]: print(f"  - {lname}: {lrate:.4%}")
        
        return global_change_rate

