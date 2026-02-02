# main/src/module/hebbian_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 基礎赫布稀疏層 (支援 Pearson Correlation 與 Grouped Conv) ---
class HebbianSparseLayer(nn.Module):
    def __init__(self, layer, sparsity, hebbian_decay=0.99, subsample_rate=0.1):
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
        計算 Pearson 相關係數 (含標準化)，支援 Grouped Conv。
        Input x, y shape: (N_samples, [Groups], Features)
        """
        # 1. Subsampling (沿著 N_samples 維度)
        if self.subsample_rate < 1.0 and self.training:
            num_samples = x.size(0)
            perm = torch.randperm(num_samples, device=x.device)[:int(num_samples * self.subsample_rate)]
            x = x[perm]
            y = y[perm]

        # 2. 標準化 (Center & Scale) over dim=0 (Batch)
        # 加上極小值 1e-8 避免除以零
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)
        
        x_norm = x_centered / (x.std(dim=0, keepdim=True) + 1e-8)
        y_norm = y_centered / (y.std(dim=0, keepdim=True) + 1e-8)

        # 3. 計算相關矩陣
        N = x_norm.size(0)
        
        if x_norm.dim() == 2:
            # Case 1: Standard Linear / Dense Conv (N, Feat)
            # Correlation = (Y^T * X) / N
            corr = torch.matmul(y_norm.t(), x_norm) / N
            
        elif x_norm.dim() == 3:
            # Case 2: Grouped Conv (N, Groups, Feat)
            # 我們需要對每個 Group 分別做 (Y_g^T * X_g)
            # Permute to (Groups, N, Feat) 以利用 bmm
            x_g = x_norm.permute(1, 0, 2) # (G, N, Di)
            y_g = y_norm.permute(1, 0, 2) # (G, N, Do)
            
            # bmm: (G, N, Do)^T * (G, N, Di) -> (G, Do, N) * (G, N, Di) -> (G, Do, Di)
            # 注意: transpose(1, 2) 是交換後兩個維度
            corr = torch.bmm(y_g.transpose(1, 2), x_g) / N
            
            # 結果是 (Groups, Cout_per_group, Cin_per_group * K * K)
            
        return corr

    def _hebbian_hook(self, module, input, output):
        if not self.training or not self.enable_hebbian:
            return
        
        x = input[0].detach()
        y = output.detach()

        if isinstance(self.layer, nn.Conv2d):
            k_size = self.layer.kernel_size
            stride = self.layer.stride
            padding = self.layer.padding
            groups = self.layer.groups
            
            # 使用 unfold 將卷積輸入展平: (B, Cin*K*K, L)
            x_unfold = F.unfold(x, k_size, dilation=1, padding=padding, stride=stride)
            
            # 轉置為 (N_samples, Total_Features) -> (B*L, Cin*K*K)
            x_flat = x_unfold.permute(0, 2, 1).contiguous().view(-1, x_unfold.size(1))
            
            # y: (B, Cout, H, W) -> (B, Cout, L) -> (B, L, Cout) -> (B*L, Cout)
            y_flat = y.view(y.size(0), y.size(1), -1).permute(0, 2, 1).contiguous().view(-1, y.size(1))
            
            if groups > 1:
                # --- Grouped Convolution 處理 ---
                # x_flat: (N, Cin * K * K). 
                # Cin = groups * Cin_per_group.
                # Reshape x to (N, groups, Cin_per_group * K * K)
                
                # 注意：unfold 的輸出通道順序通常是 (Cin, K, K)。
                # PyTorch Group Conv 假設 Cin 是按組排列的。
                cin_per_group = x_unfold.size(1) // groups
                x_grouped = x_flat.view(x_flat.size(0), groups, cin_per_group)
                
                # y_flat: (N, Cout). Cout = groups * Cout_per_group
                cout_per_group = y_flat.size(1) // groups
                y_grouped = y_flat.view(y_flat.size(0), groups, cout_per_group)
                
                corr = self._compute_pearson_corr(x_grouped, y_grouped)
                
                # corr shape: (Groups, Cout_pg, Cin_pg * K * K)
                # Weight shape: (Cout, Cin_pg, K, K) -> (Groups * Cout_pg, Cin_pg, K, K)
                # 我們可以 Flatten 前兩個維度來匹配
                
            else:
                # --- Standard Convolution ---
                corr = self._compute_pearson_corr(x_flat, y_flat)
            
            # Reshape 回權重形狀
            current_corr = corr.reshape_as(self.layer.weight)

        elif isinstance(self.layer, nn.Linear):
            x_flat = x.view(-1, x.size(-1))
            y_flat = y.view(-1, y.size(-1))
            current_corr = self._compute_pearson_corr(x_flat, y_flat)
            current_corr = current_corr.reshape_as(self.layer.weight)

        # 更新動量分數 (取絕對值)
        self.hebbian_score = self.hebbian_decay * self.hebbian_score + \
                             (1 - self.hebbian_decay) * torch.abs(current_corr)

    def update_topology(self, grow_ratio=0.2):
        with torch.no_grad():
            num_active = int(self.mask.sum().item())
            num_swap = int(num_active * grow_ratio)
            if num_swap == 0: return

            # 1. Prune (Drop 最小權重) -> 改為 Drop 最小 Pearson Score
            # active_weights = torch.abs(self.layer.weight) * self.mask
            # active_weights[self.mask == 0] = float('inf')
            # threshold_drop = torch.topk(-active_weights.view(-1), num_swap).values[-1]
            # drop_mask = (active_weights != float('inf')) & (-active_weights >= threshold_drop)

            # --- 1. 改良版 Prune (混合分數) ---
            # A. 權重分數 (歸一化到 0~1 以便相加)
            w_abs = torch.abs(self.layer.weight)
            w_score = w_abs / (w_abs.max() + 1e-8)
            # B. Hebbian 分數 (本來就在 0~1 之間，因為是 Pearson Correlation)
            # 注意：hebbian_score 是 buffer，已經是動量平均過的
            h_score = self.hebbian_score
            # C. 混合分數 (lambda 可調，例如 0.5)
            # 如果你希望 Hebbian 影響力大一點，可以設高一點，但不能完全捨棄 w_score
            hybrid_score = w_score + 0.5 * h_score
            # 只在 active 的連接中找最小的
            hybrid_score = hybrid_score * self.mask
            hybrid_score[self.mask == 0] = float('inf') # 避免選到已經是 0 的
            threshold_drop = torch.topk(-hybrid_score.view(-1), num_swap).values[-1]
            drop_mask = (hybrid_score != float('inf')) & (-hybrid_score >= threshold_drop)

            # 2. Grow (基於 Pearson Score 復活)
            candidate_scores = self.hebbian_score * (1 - self.mask)
            threshold_grow = torch.topk(candidate_scores.view(-1), num_swap).values[-1]
            grow_mask = (candidate_scores >= threshold_grow)

            # 3. Apply
            self.mask[drop_mask] = 0
            self.mask[grow_mask] = 1
            self.layer.weight.data[grow_mask] = 0.0 # 新生長權重歸零
            self.hebbian_score[self.mask == 1] = 0.0 # 重置分數
            self.layer.weight.data *= self.mask

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