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
        model_type='shufflenet',
        encoder_output_dim=1024,
        projector_inner_dim=256,
        target_sparsity=0.8,
        sparsify_projector=False,
        use_erk=True,          # Ablation Toggle: ERK 分佈
        protect_highway=True   # Ablation Toggle: 1x1 與殘差保護
    ):
        # 1. 呼叫 Refactored SimSiam 的 init，傳遞 model_type
        super().__init__(
            pretrained_model=pretrained_model, 
            model_type=model_type,
            encoder_output_dim=encoder_output_dim, 
            projector_inner_dim=projector_inner_dim
        )
        
        self.target_sparsity = target_sparsity
        self.use_erk = use_erk
        self.protect_highway = protect_highway
        
        # 2. 替換 Encoder 中的層為 Hebbian Layer
        if self.use_erk or self.protect_highway:
            print(f"Converting Encoder ({model_type}) to Hebbian Sparse (Target Global Sparsity: {target_sparsity}, ERK: {use_erk}, Protect Highway: {protect_highway})...")
            self._set_layer_sparsities(self.encoder, target_sparsity)
        else:
            print(f"Converting Encoder ({model_type}) to Hebbian Sparse V5 (Uniform Sparsity: {target_sparsity})...")
            self._replace_layers_recursively(self.encoder, target_sparsity)
        
        # 3. (選用) 替換 Projector/Predictor
        if sparsify_projector:
            print("Converting Projector/Predictor to Hebbian Sparse...")
            self._set_layer_sparsities(self.projector, target_sparsity)
            self._set_layer_sparsities(self.predictor, target_sparsity)

    def _set_layer_sparsities(self, root_module, global_target_sparsity):
        """
        Hebbian V6: 實作 ERK (Erdos-Renyi Kernel) 稀疏度分配與殘差保護
        """
        # 1. 蒐集所有候選層，並標註那些是 1x1 或 downsample
        sparsifiable_layers = []
        protected_layers = []
        
        for name, m in root_module.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # 例外：ResNet 第一層 (stem) 永遠保持稠密
                if name == 'conv1' and hasattr(root_module, 'layer1'):
                    continue
                
                is_1x1 = False
                if isinstance(m, nn.Conv2d):
                    if m.kernel_size == (1, 1) or m.kernel_size == 1:
                        is_1x1 = True
                
                is_downsample = "downsample" in name
                
                # Hebbian V6 核心：保護殘差公路 (1x1 或 downsample 分支)
                if self.protect_highway and (is_1x1 or is_downsample):
                    print(f"  [Guard] Protecting highway layer: {name}")
                    protected_layers.append((name, m))
                else:
                    sparsifiable_layers.append((name, m))

        # 2. 執行 ERK 稀疏度計算 (Erdos-Renyi Kernel)
        if self.use_erk:
            # 對於剩下的層，根據參數規模分配稀疏度
            total_params = 0
            erk_scores = []
            for name, m in sparsifiable_layers:
                if isinstance(m, nn.Conv2d):
                    n_in, n_out, kh, kw = m.weight.shape
                    total_params += m.weight.numel()
                    score = (n_in + n_out + kh + kw) / (n_in * n_out * kh * kw)
                    erk_scores.append(score)
                else: # Linear
                    n_out, n_in = m.weight.shape
                    total_params += m.weight.numel()
                    score = (n_in + n_out) / (n_in * n_out)
                    erk_scores.append(score)
            
            for i, (name, m) in enumerate(sparsifiable_layers):
                if len(sparsifiable_layers) > 0 and total_params > 0:
                    param_ratio = m.weight.numel() / (total_params / len(sparsifiable_layers))
                    layer_sparsity = global_target_sparsity + (0.05 if param_ratio > 1.2 else -0.05)
                else:
                    layer_sparsity = global_target_sparsity
                layer_sparsity = max(0.1, min(0.95, layer_sparsity)) 
                self._replace_single_layer(root_module, name, m, layer_sparsity)
        else:
            # V5 Fallback: Uniform sparsity on sparsifiable layers
            for name, m in sparsifiable_layers:
                self._replace_single_layer(root_module, name, m, global_target_sparsity)

        # 3. 處理被保護的層 (Dense)
        for name, m in protected_layers:
            self._replace_single_layer(root_module, name, m, 0.0)

    def _replace_single_layer(self, root_module, name, child, sparsity):
        """
        輔助函式：將單一層替換為 HebbianSparseLayer
        """
        parent_name = name.rsplit('.', 1)
        if len(parent_name) > 1:
            parent = dict(root_module.named_modules())[parent_name[0]]
            leaf_name = parent_name[1]
        else:
            parent = root_module
            leaf_name = name
        
        # Hebbian V5/V6: 使用 subsample_rate=1.0 確保統計精確度
        print(f"  Replacing {name} with Hebbian (Sparsity: {sparsity:.4f}, Sample: 1.0)")
        setattr(parent, leaf_name, HebbianSparseLayer(child, sparsity, subsample_rate=1.0))

    def _replace_layers_recursively(self, module, sparsity):
        """ (Legacy) 舊版本的齊頭式替換 """
        self._set_layer_sparsities(module, sparsity)

    def update_topology(self, grow_ratio=0.2):
        """
        外部訓練迴圈呼叫此函數來更新結構
        回傳字典 {module: (grow_mask, drop_mask)} 以便優化器進行動量清除
        """
        topology_changes = {}
        for m in self.modules():
            if isinstance(m, HebbianSparseLayer):
                grow_mask, drop_mask = m.update_topology(grow_ratio)
                topology_changes[m] = (grow_mask, drop_mask)
        return topology_changes
            
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