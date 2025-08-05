import torch
import torch.nn as nn
import torch.nn.functional as F
import copy # 用於深度複製模型
from contextlib import contextmanager
from src.module.SimSiam_Module import SimSiam, SimSiamLoss_unsymmetric

class SparseSimSiam(SimSiam):
    def __init__(self, *args, **kwargs):
        super(SparseSimSiam, self).__init__(*args, **kwargs)

        # s_params 的創建和初始化 (為線上網路服務)
        self.s_params = nn.ParameterDict()
        self._create_s_params_for_online() # [修正] s_params 現在只關聯線上網路的結構

        # [新增] 目標網路的 EMA 權重副本
        # 這些是普通的 nn.Module，它們的權重將透過 EMA 更新，不參與梯度計算
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        
        # 將目標網路的權重設置為不可訓練，它們只通過 EMA 更新
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

        # [新增] EMA 動量係數
        self.momentum = 0.5 # 參考 BYOL/DINO 的典型值

    def _create_s_params_for_online(self):
        """為線上網路的 encoder 和 projector 創建 s 參數。"""
        # 為 encoder 中的卷積層創建 s 參數
        for name, module in self.encoder.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                s_name = name.replace('.', '_')
                self.s_params[s_name] = nn.Parameter(torch.ones_like(module.weight)) # [修正] s 初始為1
        
        # 為 projector 中的線性層創建 s 參數
        for name, module in self.projector.named_modules():
             if isinstance(module, nn.Linear):
                s_name = f"projector_{name.replace('.', '_')}"
                self.s_params[s_name] = nn.Parameter(torch.ones_like(module.weight)) # [修正] s 初始為1
        
        # 初始化 s 參數
        self._initialize_s_params(mean=1.0, std=0.01) # [修正] 初始化 s_params 的均值為 1

    def _initialize_s_params(self, mean=1.0, std=0.01):
        """初始化 s 參數。"""
        for s in self.s_params.values():
            nn.init.normal_(s, mean=mean, std=std)

    # [新增] EMA 更新目標網路權重的方法
    @torch.no_grad()
    def _update_target_network_ema(self):
        """
        使用 EMA 更新目標網路的權重。
        注意：s_params 不在這裡更新，它們通過梯度下降更新。
        """
        # 更新 Encoder
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)
        
        # 更新 Projector
        for param_q, param_k in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)

    @contextmanager
    def _sparse_target_weight_context(self, alpha):
        """
        這個上下文管理器現在的作用是：
        1. 臨時替換目標網路 (target_encoder, target_projector) 的權重。
        2. 替換後的權重是 EMA 更新後的密集權重，再結合可學習的稀疏遮罩 's'。
        3. 這些替換後的權重是 'detach' 的，確保不傳遞梯度給原始的 EMA 權重。
        4. 但是，遮罩 'm' 來自 's_params'，而 's_params' 仍是可訓練的，梯度會流向它。
        """
        # 獲取目標網路的模組列表 (與線上網路結構相同)
        target_modules_and_names = []
        for name, module in self.target_encoder.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                target_modules_and_names.append((f"{name.replace('.', '_')}", module))
        for name, module in self.target_projector.named_modules():
            if isinstance(module, nn.Linear):
                target_modules_and_names.append((f"projector_{name.replace('.', '_')}", module))

        original_target_weights = {}
        try:
            for s_name_prefix, target_module in target_modules_and_names:
                if s_name_prefix in self.s_params: # 確保該層有對應的 s 參數
                    # 1. 保存目標網路原始（EMA 更新後）的權重
                    original_target_weights[s_name_prefix] = target_module.weight.data
                    
                    # 2. 計算稀疏權重
                    # 從 target_module 獲取權重（EMA 更新後的），然後 detach
                    w_ema_detached = target_module.weight.detach() 
                    s = self.s_params[s_name_prefix] # 使用線上網路的 s_params
                    m = torch.sigmoid(alpha * s)
                    sparse_weight_for_forward = w_ema_detached * m
                    
                    # 3. 臨時替換目標模組的權重
                    target_module.weight.data = sparse_weight_for_forward
            
            yield # 執行 with 語句塊中的代碼 (即 target network 的 forward)

        finally:
            # 退出上下文：恢復目標網路的原始權重（EMA 更新後的）
            for s_name_prefix, target_module in target_modules_and_names:
                if s_name_prefix in original_target_weights:
                    target_module.weight.data = original_target_weights[s_name_prefix]


    def forward(self, x1, x2, alpha, momentum=None):
        """
        模型的前向傳播。
        
        Args:
            x1 (Tensor): 第一個增強視圖 (用於線上網路)。
            x2 (Tensor): 第二個增強視圖 (用於目標網路)。
            alpha (float): Sigmoid 陡峭度係數。
        """
        # --- 線上分支 (Online Branch) ---
        # 線上網路的權重是正常訓練的密集權重
        y1_online = self.encoder(x1).mean([2, 3])
        z1_online = self.projector(y1_online)
        p1 = self.predictor(z1_online)

        # --- 目標分支 (Target Branch) ---
        # 1. 在前向傳播前，先更新目標網路的 EMA 權重
        if momentum is not None:
            self.momentum = momentum
        self._update_target_network_ema()

        # 2. 進入稀疏上下文，此時目標網路的權重會被臨時替換為稀疏且 detach 的版本
        with self._sparse_target_weight_context(alpha):
            # 確保整個目標網路的計算都在 torch.no_grad() 中進行
            # 這是因為 target_encoder 和 target_projector 的 EMA 權重本身就不需要梯度
            # 而 s_params 的梯度會通過 'm' 得到
            with torch.no_grad(): # [修正] 確保這裡的 no_grad 覆蓋所有計算
                y2_target_embedding = self.target_encoder(x2).mean([2, 3])
                z2_target = self.target_projector(y2_target_embedding)

        # SimSiam 損失的計算將在訓練腳本中進行
        # 返回 p1 和 z2_target
        return p1, z2_target