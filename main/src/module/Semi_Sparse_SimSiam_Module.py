import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from src.module.SimSiam_Module import SimSiam, SimSiamLoss_unsymmetric

class SparseSimSiam(SimSiam):
    def __init__(self, *args, **kwargs):
        super(SparseSimSiam, self).__init__(*args, **kwargs)

        # s_params 的創建和初始化保持不變
        self.s_params = nn.ParameterDict()
        self._create_s_params()
        self._initialize_s_params()

    def _create_s_params(self):
        """遍歷模型，為帶權重的層創建 s 參數。"""
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                s_name = name.replace('.', '_')
                self.s_params[s_name] = nn.Parameter(torch.zeros_like(module.weight))
    
    def _initialize_s_params(self, mean=1.0, std=0.01):
        """初始化 s 參數。"""
        for s in self.s_params.values():
            nn.init.normal_(s, mean=mean, std=std)

    @contextmanager
    def sparse_target_mode(self, alpha):
        """
        這就是我們的核心武器：一個上下文管理器。
        """
        original_weights = {}
        try:
            # 進入上下文：替換權重
            for name, module in self.named_modules():
                s_name = name.replace('.', '_')
                if isinstance(module, (nn.Conv2d, nn.Linear)) and s_name in self.s_params:
                    # 1. 保存原始權重
                    original_weights[name] = module.weight.data
                    
                    # 2. 計算稀疏權重，並確保它不帶權重的梯度
                    w_detached = module.weight.detach() # 關鍵！
                    s = self.s_params[s_name]
                    m = torch.sigmoid(alpha * s)
                    sparse_weight = w_detached * m
                    
                    # 3. 直接替換模組的權重
                    module.weight.data = sparse_weight
            
            yield # 執行 with 語句塊中的代碼 (即 target network 的 forward)

        finally:
            # 退出上下文：恢復原始權重
            for name, module in self.named_modules():
                 if name in original_weights:
                    module.weight.data = original_weights[name]


    def forward(self, x1, x2, alpha):
        """
        修正後的前向傳播，現在它優雅且正確。
        """
        # --- 線上分支 (Online Branch) ---
        # 正常傳播，權重是原始的密集權重。
        # 注意：s_params 作為模型的一部分，其梯度會被正常計算。
        y1_online = self.encoder(x1).mean([2, 3])
        z1_online = self.projector(y1_online)
        p1 = self.predictor(z1_online)

        # --- 目標分支 (Target Branch) ---
        # 進入我們的稀疏目標模式上下文
        with self.sparse_target_mode(alpha):
            # 在這個區塊內，模型的權重被臨時替換了
            # 模型原生的 forward 方法可以被安全調用，因為它現在操作的是
            # 我們提供的、已經分離了權重梯度的稀疏權重。
            y2_target_embedding = self.encoder(x2).mean([2, 3])
            z2_target = self.projector(y2_target_embedding)

        # 損失計算可以在外部完成，也可以在這裡完成
        return p1, z2_target