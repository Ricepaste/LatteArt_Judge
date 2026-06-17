# main/src/module/hebbian_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# --- 1. 基礎赫布稀疏層 (支援 Pearson Correlation 與 Grouped Conv) ---
class HebbianSparseLayer(nn.Module):
    def __init__(self, layer, sparsity, hebbian_decay=0.9, subsample_rate=0.1):
        """
        args:
            layer: 原始的 nn.Conv2d 或 nn.Linear
            sparsity: 目標稀疏度 (例如 0.9 代表剪掉 90%)
            subsample_rate: 計算相關性時採樣的比例，節省顯存
        """
        super().__init__()
        self.layer = layer
        self.sparsity = sparsity
        self.hebbian_decay = hebbian_decay
        self.subsample_rate = subsample_rate
        
        # 註冊 buffer (Mask & Score)
        self.register_buffer('mask', torch.ones_like(layer.weight))
        self.register_buffer('hebbian_score', torch.zeros_like(layer.weight))
        
        # 新增 Hebbian V3: 輸入特徵啟用率追蹤 (針對每個輸入通道 Firing Rate)
        if isinstance(layer, nn.Conv2d):
            in_channels = layer.weight.shape[1] * layer.groups
        else:
            in_channels = layer.weight.shape[1]
        self.register_buffer('input_firing_score', torch.zeros(in_channels))
        
        # 初始化 Mask (Magnitude Pruning)
        self._init_mask()
        
        # 註冊 Hook
        self.hook_handle = self.register_forward_hook(self._hebbian_hook)
        self.enable_hebbian = True # 控制開關

    def _init_mask(self):
        k = int((1 - self.sparsity) * self.layer.weight.numel())
        if k == 0: return
        # 初始保留絕對值最大的權重
        threshold = torch.topk(torch.abs(self.layer.weight).view(-1), k).values[-1]
        self.mask = (torch.abs(self.layer.weight) >= threshold).float()
        with torch.no_grad():
            self.layer.weight.data *= self.mask

    def forward(self, x):
        # 應用 Mask 進行前向傳播
        w_sparse = self.layer.weight * self.mask
        if isinstance(self.layer, nn.Linear):
            return F.linear(x, w_sparse, self.layer.bias)
        elif isinstance(self.layer, nn.Conv2d):
            return F.conv2d(x, w_sparse, self.layer.bias, 
                            self.layer.stride, self.layer.padding, 
                            self.layer.dilation, self.layer.groups)

    def _compute_pearson_corr(self, x, y):
        """
        計算 Pearson 相關係數，支援 Grouped Conv。
        """
        # 1. 快速 Subsampling (使用 Boolean Mask 代替 randperm)
        if self.subsample_rate < 1.0 and self.training:
            num_samples = x.size(0)
            mask = torch.rand(num_samples, device=x.device) < self.subsample_rate
            x = x[mask]
            y = y[mask]

        if x.size(0) < 2: return torch.zeros_like(self.layer.weight)

        # 2. 標準化 (Center & Scale)
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)
        
        # 使用更有利的 GPU 計算標準差
        x_std = torch.sqrt((x_centered**2).mean(dim=0, keepdim=True) + 1e-8)
        y_std = torch.sqrt((y_centered**2).mean(dim=0, keepdim=True) + 1e-8)

        x_norm = x_centered / x_std
        y_norm = y_centered / y_std

        # 3. 計算相關矩陣
        N = x_norm.size(0)
        
        if x_norm.dim() == 2:
            corr = torch.matmul(y_norm.t(), x_norm) / N
        elif x_norm.dim() == 3:
            # Case 2: Grouped Conv (N, Groups, Feat)
            x_g = x_norm.permute(1, 0, 2) # (G, N, Di)
            y_g = y_norm.permute(1, 0, 2) # (G, N, Do)
            corr = torch.bmm(y_g.transpose(1, 2), x_g) / N
            
        return corr

    def _hebbian_hook(self, module, input, output):
        if not self.training or not self.enable_hebbian:
            return
        
        x = input[0].detach()
        y = output.detach()
        
        # Hebbian V3: 計算 Firing Rate (特徵大於 0 的比例)
        # Hebbian V8: 計算特徵方差 (Variance, 用於找出高 SNR 訊號)
        if x.dim() == 4:
            firing_rate = (x > 0).float().mean(dim=(0, 2, 3))
            x_var = x.var(dim=(0, 2, 3), unbiased=False)
            y_var = y.var(dim=(0, 2, 3), unbiased=False)
        else:
            firing_rate = (x > 0).float().mean(dim=0)
            x_var = x.var(dim=0, unbiased=False)
            y_var = y.var(dim=0, unbiased=False)
            
        # 初始化方差追蹤器 (第一次觸發時)
        if getattr(self, 'input_variance', None) is None:
            self.register_buffer('input_variance', torch.ones_like(x_var))
            self.register_buffer('output_variance', torch.ones_like(y_var))

        if isinstance(self.layer, nn.Conv2d):
            k_size = self.layer.kernel_size
            stride = self.layer.stride
            padding = self.layer.padding
            groups = self.layer.groups
            
            # 使用 unfold 展開輸入
            x_unfold = F.unfold(x, k_size, dilation=1, padding=padding, stride=stride)
            
            # 轉換形狀 (B*H*W, C_total)
            x_flat = x_unfold.permute(0, 2, 1).reshape(-1, x_unfold.size(1))
            y_flat = y.permute(0, 2, 3, 1).reshape(-1, y.size(1))
            
            if groups > 1:
                cin_per_group = x_unfold.size(1) // groups
                cout_per_group = y_flat.size(1) // groups
                x_grouped = x_flat.view(x_flat.size(0), groups, cin_per_group)
                y_grouped = y_flat.view(y_flat.size(0), groups, cout_per_group)
                corr = self._compute_pearson_corr(x_grouped, y_grouped)
            else:
                corr = self._compute_pearson_corr(x_flat, y_flat)
            
            current_corr = corr.reshape_as(self.layer.weight)

        elif isinstance(self.layer, nn.Linear):
            x_flat = x.view(-1, x.size(-1))
            y_flat = y.view(-1, y.size(-1))
            current_corr = self._compute_pearson_corr(x_flat, y_flat)
            current_corr = current_corr.reshape_as(self.layer.weight)

        # 更新動量分數 (Hebbian V3)
        if hasattr(self, 'hebbian_score') and current_corr is not None:
            self.hebbian_score = self.hebbian_decay * self.hebbian_score + \
                                 (1 - self.hebbian_decay) * torch.abs(current_corr)
        
        if hasattr(self, 'input_firing_score'):
            self.input_firing_score = self.hebbian_decay * self.input_firing_score + \
                                      (1 - self.hebbian_decay) * firing_rate
                                      
        if hasattr(self, 'input_variance'):
            self.input_variance = self.hebbian_decay * self.input_variance + \
                                  (1 - self.hebbian_decay) * x_var
            self.output_variance = self.hebbian_decay * self.output_variance + \
                                   (1 - self.hebbian_decay) * y_var

    def update_topology(self, grow_ratio=0.2):
        """
        更新拓樸結構 (Hebbian V3)：
        1. 剪枝：純權重絕對值剪枝。
        2. 生長 (Dual-Engine)：
           - 25% 基於反赫布 (Anti-Hebbian)：Pearson 絕對值最小。
           - 75% 基於資訊熵 (Entropy)：輸入 Firing Rate 最接近 0.5。
           若剛被剪掉則改為隨機生長。
        """
        with torch.no_grad():
            num_active = int(self.mask.sum().item())
            num_swap = int(num_active * grow_ratio)
            if num_swap <= 0: return

            # --- 1. 剪枝 (Prune): 純權重強度 ---
            w_abs = torch.abs(self.layer.weight) * self.mask
            w_for_pruning = w_abs.clone()
            w_for_pruning[self.mask == 0] = float('inf')
            
            _, drop_idx = torch.topk(-w_for_pruning.view(-1), num_swap)
            drop_mask = torch.zeros_like(self.mask, dtype=torch.bool)
            drop_mask.view(-1)[drop_idx] = True

            # --- 2. 生長 (Grow): Hebbian V4 (2D Anti-Hebbian x 1D Entropy Joint Score) ---
            # 目標：尋找絕對相關性最小 (最正交/新鮮) 且資訊熵最高 (Firing Rate 接近 0.5) 的連接
            dead_mask = (self.mask == 0)
            potential_pool = dead_mask | drop_mask
            
            proposed_grow_mask = torch.zeros_like(self.mask, dtype=torch.bool)

            if num_swap > 0:
                # 1. 取得 1D Entropy 分數
                # 正規化到 [0, 1]，0.5 (最大熵) 時分數為 1，全 0 或全 1 時分數為 0
                entropy_1d = 1.0 - torch.abs(self.input_firing_score - 0.5) * 2.0
                
                # 2. 準備 2D 反赫布分數 (-abs(Corr))
                # 轉換為正向乘數，1.0 代表「完全不相關/正交」，0.0 代表「完全線性相關」
                anti_hebbian_2d = 1.0 - self.hebbian_score.clone()
                
                # 3. 準備 2D 方差分數 (Variance/SNR Mask) 
                if isinstance(self.layer, nn.Conv2d):
                    groups = self.layer.groups
                    C_out, C_in_pg = self.layer.weight.shape[:2]
                    C_out_pg = C_out // groups
                    
                    entropy_2d = entropy_1d.view(groups, 1, C_in_pg, 1, 1).expand(groups, C_out_pg, C_in_pg, *self.layer.weight.shape[2:])
                    entropy_2d = entropy_2d.reshape_as(self.layer.weight)
                    
                    x_var_pg = self.input_variance.view(groups, 1, C_in_pg, 1, 1).expand(groups, C_out_pg, C_in_pg, *self.layer.weight.shape[2:])
                    y_var_pg = self.output_variance.view(groups, C_out_pg, 1, 1, 1).expand(groups, C_out_pg, C_in_pg, *self.layer.weight.shape[2:])
                    variance_2d = torch.sqrt(x_var_pg * y_var_pg).reshape_as(self.layer.weight)
                else:
                    entropy_2d = entropy_1d.view(1, -1).expand_as(self.layer.weight)
                    
                    x_var_2d = self.input_variance.view(1, -1).expand_as(self.layer.weight)
                    y_var_2d = self.output_variance.view(-1, 1).expand_as(self.layer.weight)
                    variance_2d = torch.sqrt(x_var_2d * y_var_2d)
                
                # Hebbian V8: 聯合成績 (Variance-Weighted Anti-Hebbian)
                # 預設使用連乘設計，但提供消融實驗 (Ablation Study) 支援
                
                # 從環境變數讀取消融設定 (預設全開)
                use_anti_hebb = os.environ.get("ABLATION_ANTI_HEBB", "1") == "1"
                use_variance = os.environ.get("ABLATION_VARIANCE", "1") == "1"
                use_entropy = os.environ.get("ABLATION_ENTROPY", "1") == "1"
                use_positive_hebb_only = os.environ.get("ABLATION_POSITIVE_HEBB_ONLY", "0") == "1"
                use_random_growth = os.environ.get("ABLATION_RANDOM_GROWTH", "0") == "1"
                
                if use_random_growth:
                    # 純隨機生長 (SET)，對齊所有的網絡拓撲保護與 ERK 分佈
                    # 生成與權重形狀相同的隨機分數，並在潛在池中進行選擇
                    joint_score = torch.rand_like(self.layer.weight)
                elif use_positive_hebb_only:
                    # 完全使用正赫布 (Positive Hebbian) 作為生長依據，不使用 joint score 評分 (不乘以 variance 和 entropy)
                    joint_score = self.hebbian_score.clone()
                else:
                    joint_score = torch.ones_like(anti_hebbian_2d)
                    if use_anti_hebb:
                        joint_score *= anti_hebbian_2d
                    if use_variance:
                        joint_score *= variance_2d
                    if use_entropy:
                        joint_score *= entropy_2d
                
                # 排除非潛在池
                joint_score[~potential_pool] = -float('inf')
                
                _, grow_idx = torch.topk(joint_score.view(-1), num_swap)
                proposed_grow_mask.view(-1)[grow_idx] = True
            
            # --- 3. 衝突排除與隨機回退 (Just-Pruned Revival) ---
            revived_mask = proposed_grow_mask & drop_mask
            num_revived = int(revived_mask.sum().item())
            
            final_grow_mask = proposed_grow_mask & (~revived_mask)
            
            if num_revived > 0:
                random_pool_mask = dead_mask & (~proposed_grow_mask)
                random_indices = torch.where(random_pool_mask.view(-1))[0]
                
                if random_indices.numel() > 0:
                    num_to_grow_randomly = min(num_revived, random_indices.numel())
                    rand_vals = torch.rand(random_indices.numel(), device=self.mask.device)
                    _, rand_top_k = torch.topk(rand_vals, num_to_grow_randomly)
                    final_grow_mask.view(-1)[random_indices[rand_top_k]] = True

            # --- 4. 應用的變動 ---
            self.mask[drop_mask] = 0
            self.mask[final_grow_mask] = 1
            
            # Hebbian V5: 活化初始化 (Active Init)
            # 不再設為 0.0，而是給予符合該層標準差的微小隨機值
            if final_grow_mask.any():
                # 使用 Kaiming Uniform 邏輯估算標準差
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.layer.weight)
                std = math.sqrt(2.0 / fan_in) if fan_in > 0 else 0.01
                
                # 生成新權重
                new_weights = torch.randn_like(self.layer.weight) * std * 0.1 # 稍微小一點，避免初期衝擊太大
                self.layer.weight.data[final_grow_mask] = new_weights[final_grow_mask]
            
            # 確保被剪掉的真的歸零
            self.layer.weight.data *= self.mask
            
            # Hebbian V7: 修正 Bug。原本這裡會清空所有 mask=1 的分數，導致分數無法跨 Epoch 累積。
            # 現在只清空「剛長出來」的連線分數，讓穩定連線的分數能持續積累。
            self.hebbian_score[final_grow_mask] = 0.0
            
            return final_grow_mask, drop_mask

# --- 2. ShuffleNet 專用適配層 ---
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class HebbianInvertedResidual(nn.Module):
    def __init__(self, original_block, sparsity=0.9):
        super().__init__()
        self.stride = original_block.stride
        self.branch1 = self._convert_to_hebbian(original_block.branch1, sparsity)
        self.branch2 = self._convert_to_hebbian(original_block.branch2, sparsity)

    def _convert_to_hebbian(self, module, sparsity):
        new_module = nn.Sequential()
        if len(list(module.children())) == 0:
            return module
            
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                new_module.add_module(name, HebbianSparseLayer(child, sparsity))
            else:
                new_module.add_module(name, child)
        return new_module

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)