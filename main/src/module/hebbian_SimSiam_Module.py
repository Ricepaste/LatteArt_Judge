# main/src/module/hebbian_SimSiam_Module.py
from torch import Tensor, tensor
import torch.nn as nn
from src.module.SimSiam_Module import SimSiam, SimSiam_online
from src.module.hebbian_layers import HebbianInvertedResidual, HebbianSparseLayer
# 引入必要的 Block 定義以便識別
from torchvision.models.shufflenetv2 import InvertedResidual
from torchvision.models.resnet import BasicBlock, Bottleneck

class Hebbian_SimSiam(SimSiam):
    def __init__(
        self,
        pretrained_model,
        model_type='shufflenet', # <--- 新增參數
        encoder_output_dim=1024,
        projector_inner_dim=256,
        target_sparsity=0.8,
        sparsify_projector=False
    ):
        # 1. 呼叫 Refactored SimSiam 的 init，傳遞 model_type
        super().__init__(
            pretrained_model=pretrained_model, 
            model_type=model_type,
            encoder_output_dim=encoder_output_dim, 
            projector_inner_dim=projector_inner_dim
        )
        
        self.target_sparsity = target_sparsity
        
        # 2. 替換 Encoder 中的層為 Hebbian Layer
        print(f"Converting Encoder ({model_type}) to Hebbian Sparse (Sparsity: {target_sparsity})...")
        self._replace_layers_recursively(self.encoder, target_sparsity)
        
        # 3. (選用) 替換 Projector/Predictor
        if sparsify_projector:
            print("Converting Projector/Predictor to Hebbian Sparse...")
            self._replace_layers_recursively(self.projector, target_sparsity)
            self._replace_layers_recursively(self.predictor, target_sparsity)

    def _replace_layers_recursively(self, module, sparsity):
        """
        遞歸遍歷並替換層：
        1. InvertedResidual (ShuffleNet) -> HebbianInvertedResidual
        2. BasicBlock/Bottleneck (ResNet) -> 遞歸進入 (不需特殊 Wrapper，直接替換內部 Conv)
        3. Conv2d/Linear -> HebbianSparseLayer
        """
        for name, child in module.named_children():
            # A. 針對 ShuffleNet Block 的特殊處理
            # 因為 ShuffleNet 需要特殊的 channel_shuffle 操作，所以使用 Wrapper
            if isinstance(child, InvertedResidual):
                setattr(module, name, HebbianInvertedResidual(child, sparsity))
            
            # B. 針對 ResNet Block 的處理
            # ResNet 的 BasicBlock/Bottleneck 不需要特殊 Wrapper，
            # 只要遞歸進去替換內部的 Conv2d 即可保持 Residual Connection 結構不變
            elif isinstance(child, (BasicBlock, Bottleneck)):
                self._replace_layers_recursively(child, sparsity)

            # C. 處理一般卷積與全連接 (這是遞歸的終止條件)
            elif isinstance(child, (nn.Conv2d, nn.Linear)):
                # 注意：这里会覆盖掉原本的层，但保留参数
                # ResNet 的 downsample 層通常包含在 Sequential 中，也會被這裡的遞歸涵蓋
                setattr(module, name, HebbianSparseLayer(child, sparsity))
            
            else:
                # D. 繼續遞歸 (例如 nn.Sequential, ModuleList, 或 ResNet 的 downsample)
                self._replace_layers_recursively(child, sparsity)

    def update_topology(self, grow_ratio=0.2):
        """
        外部訓練迴圈呼叫此函數來更新結構
        """
        count = 0
        for m in self.modules():
            if isinstance(m, HebbianSparseLayer):
                m.update_topology(grow_ratio)
                count += 1
            # HebbianInvertedResidual 內部包含 HebbianSparseLayer，會被 self.modules() 自動遍歷到
            
    def set_hebbian_enable(self, enable: bool):
        """
        控制是否計算 Hebbian Score
        """
        for m in self.modules():
            if isinstance(m, HebbianSparseLayer):
                m.enable_hebbian = enable

class SparseSimSiam_online(Hebbian_SimSiam):
    # 繼承 Hebbian_SimSiam，同時擁有 SimSiam 的功能
    def __init__(self, *args, **kwargs):
        # 確保參數透傳 (包含 model_type)
        super(SparseSimSiam_online, self).__init__(*args, **kwargs)

    def forward(self, x1):
        # Forward 邏輯與 SimSiam_online 相同
        y1 = self.encoder(x1).mean([2, 3])
        z1 = self.projector(y1)
        p1 = self.predictor(z1)
        return p1