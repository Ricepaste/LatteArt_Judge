# 假設上面的 class 定義已經 import 或寫在同一檔案中
from torch import Tensor, tensor
import torch.nn as nn
from src.module.SimSiam_Module import SimSiam
from src.module.Random_layers import HebbianInvertedResidual, SETSparseLayer
from torchvision.models.shufflenetv2 import InvertedResidual

class SET_SimSiam(SimSiam):
    def __init__(
        self,
        pretrained_model,
        model_type='shufflenet', # <--- 新增參數
        encoder_output_dim=1024,
        projector_inner_dim=256,
        target_sparsity=0.8, # 目標稀疏度
        sparsify_projector=False # 通常 SSL 不稀疏化 projector，保持 dense 效果較好
    ):
        # 1. 呼叫原始 SimSiam 的 init，這會建立 self.encoder, self.projector 等
        super().__init__(pretrained_model, 
                         model_type=model_type, 
                         encoder_output_dim=encoder_output_dim, 
                         projector_inner_dim=projector_inner_dim
                         )
        
        self.target_sparsity = target_sparsity
        
        # 2. 替換 Encoder 中的層為 Hebbian Layer
        print(f"Converting Encoder to Hebbian Sparse (Sparsity: {target_sparsity})...")
        self._replace_layers_recursively(self.encoder, target_sparsity)
        
        # 3. (選用) 替換 Projector/Predictor
        if sparsify_projector:
            print("Converting Projector/Predictor to Hebbian Sparse...")
            self._replace_layers_recursively(self.projector, target_sparsity)
            self._replace_layers_recursively(self.predictor, target_sparsity)

    def _replace_layers_recursively(self, module, sparsity):
        """
        遞歸遍歷並替換層：
        1. InvertedResidual -> HebbianInvertedResidual
        2. Conv2d/Linear -> HebbianSparseLayer
        """
        for name, child in module.named_children():
            # 優先處理 ShuffleNet Block
            if isinstance(child, InvertedResidual):
                setattr(module, name, HebbianInvertedResidual(child, sparsity))
            
            # 處理一般卷積與全連接
            elif isinstance(child, (nn.Conv2d, nn.Linear)):
                # 為了避免替換掉已經被包在 HebbianInvertedResidual 裡面的層，
                # 我們只處理直接暴露在 Sequential 下的層
                setattr(module, name, SETSparseLayer(child, sparsity))
            
            else:
                # 繼續遞歸 (例如 Sequential, ModuleList)
                self._replace_layers_recursively(child, sparsity)

    def update_topology(self, grow_ratio=0.2):
        """
        外部訓練迴圈呼叫此函數來更新結構
        """
        count = 0
        for m in self.modules():
            if isinstance(m, SETSparseLayer):
                m.update_topology(grow_ratio)
                count += 1
            # HebbianInvertedResidual 不需要特別處理，
            # 因為 self.modules() 會遞歸找到它裡面的 HebbianSparseLayer
            
    def set_hebbian_enable(self, enable: bool):
        """
        控制是否計算 Hebbian Score (例如只在特定 Step 開啟以節省計算)
        """
        for m in self.modules():
            if isinstance(m, SETSparseLayer):
                m.enable_hebbian = enable

class SparseSimSiam_online(SET_SimSiam):
    # 繼承 SparseSimSiam 而不是原始 SimSiam
    def __init__(self, *args, **kwargs):
        # 這裡會呼叫 SparseSimSiam 的 init，自動完成層替換
        super(SparseSimSiam_online, self).__init__(*args, **kwargs)

    def forward(self, x1):
        # Forward 邏輯不變
        y1 = self.encoder(x1).mean([2, 3])
        z1 = self.projector(y1)
        p1 = self.predictor(z1)
        return p1