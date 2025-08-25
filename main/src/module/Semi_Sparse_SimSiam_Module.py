import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from src.module.SimSiam_Module import SimSiam

# 為了讓 isinstance 能夠工作，我們需要從 torchvision 導入這些類
# 請確保您的環境中安裝了 torchvision
try:
    from torchvision.models.shufflenetv2 import InvertedResidual, channel_shuffle # type: ignore
except ImportError:
    print("Warning: Could not import InvertedResidual or channel_shuffle from torchvision.")
    print("The model may not work correctly with ShuffleNetV2.")
    # 定義一個假的類，以防導入失敗，避免程序崩潰，但功能會受限
    class InvertedResidual(nn.Module): pass
    def channel_shuffle(x, groups): return x


class SparseSimSiam(SimSiam):
    def __init__(self, *args, **kwargs):
        super(SparseSimSiam, self).__init__(*args, **kwargs)

        self.s_params = nn.ParameterDict()
        # 遞歸地為 self.encoder 和 self.projector 創建 s_params
        self._create_s_params_recursively(self.encoder, "encoder")
        self._create_s_params_recursively(self.projector, "projector")
        
        # 初始化 s_params 的均值為 1，以確保初始時為準密集模型
        self._initialize_s_params(mean=0.5, std=0.01)

        # 創建目標網路的 EMA 權重副本
        self.dense_target_encoder = copy.deepcopy(self.encoder)
        self.dense_target_projector = copy.deepcopy(self.projector)
        
        # 將目標網路的權重設置為不可訓練
        for param in self.dense_target_encoder.parameters():
            param.requires_grad = False
        for param in self.dense_target_projector.parameters():
            param.requires_grad = False

        # --- 新增: 創建目標網路的稀疏參數 (s_params) 的 EMA 副本 ---
        self.target_s_params = nn.ParameterDict()
        # 根據線上網路的 s_params 結構創建目標網路的 s_params
        for key, s_param in self.s_params.items():
            # 使用 .data.clone() 複製數據，並設置 requires_grad=False，確保其不可訓練
            self.target_s_params[key] = nn.Parameter(s_param.data.clone(), requires_grad=False)
        # --- 新增結束 ---

        # 設置預設的 EMA 動量係數
        self.momentum = 0.996
        self.mask_momentum = 0.996

    def _create_s_params_recursively(self, module_container, prefix):
        """遞歸地為容器內的所有可稀疏化層創建 s 參數。"""
        for name, module in module_container.named_children():
            full_name = f"{prefix}_{name}"
            s_key = full_name.replace('.', '_')
            
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.s_params[s_key] = nn.Parameter(torch.ones_like(module.weight))
            # 如果子模組仍然是個容器，就遞歸進去
            elif len(list(module.children())) > 0:
                self._create_s_params_recursively(module, full_name)

    def _initialize_s_params(self, mean=1.0, std=0.01):
        """初始化 s 參數。"""
        for s in self.s_params.values():
            nn.init.normal_(s, mean=mean, std=std)

    @torch.no_grad()
    def _update_target_network_ema(self):
        """使用 EMA 更新目標網路的權重和稀疏化參數 s。"""
        # 更新權重（現有邏輯）
        for param_q, param_k in zip(self.encoder.parameters(), self.dense_target_encoder.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)
        
        for param_q, param_k in zip(self.projector.parameters(), self.dense_target_projector.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)

        # --- 新增: 使用 EMA 更新目標網路的稀疏參數 (s_params) ---
        for key in self.s_params.keys():
            s_q = self.s_params[key]         # 線上網路的 s 參數
            s_k = self.target_s_params[key]  # 目標網路的 s 參數
            s_k.data.mul_(self.mask_momentum).add_(s_q.data, alpha=1 - self.mask_momentum)
        # --- 新增結束 ---
    
    def _forward_sparse_recursively(self, module_container, current_input, prefix, alpha, weights_source, s_params_source):
        """
        通用的遞歸式稀疏前向傳播函數。
        
        Args:
            module_container: 當前要處理的模組 (例如 self.target_encoder 或 self.encoder)。
                              用於遍歷結構和識別模組類型。
            current_input: 當前的輸入張量。
            prefix: 用於查找 s_params 的名稱前綴。
            alpha: Sigmoid 陡峭度係數。
            weights_source: 包含權重的模組容器 (例如 self.dense_target_encoder 或 self.encoder)。
                            實際的權重會從這裡獲取。
            s_params_source: 包含 s 參數的 ParameterDict (例如 self.s_params 或 self.target_s_params)。
                             稀疏遮罩會從這裡計算。
        """
        # 特殊處理 InvertedResidual 塊 (ShuffleNetV2)
        if isinstance(module_container, InvertedResidual):
            if module_container.stride == 1:
                x1, x2 = current_input.chunk(2, dim=1)
                # 對 branch2 進行遞歸，並傳入正確的分割後輸入 x2
                branch2_out = self._forward_sparse_recursively(
                    module_container.branch2, x2, f"{prefix}_branch2", alpha, 
                    getattr(weights_source, 'branch2'), s_params_source # 從 weights_source 獲取對應的 branch 模組
                )
                out = torch.cat((x1, branch2_out), dim=1)
            else:
                # 對兩個分支都進行遞歸
                branch1_out = self._forward_sparse_recursively(
                    module_container.branch1, current_input, f"{prefix}_branch1", alpha, 
                    getattr(weights_source, 'branch1'), s_params_source # 從 weights_source 獲取對應的 branch 模組
                )
                branch2_out = self._forward_sparse_recursively(
                    module_container.branch2, current_input, f"{prefix}_branch2", alpha, 
                    getattr(weights_source, 'branch2'), s_params_source # 從 weights_source 獲取對應的 branch 模組
                )
                out = torch.cat((branch1_out, branch2_out), dim=1)
            
            return channel_shuffle(out, 2)

        # 通用遍歷邏輯
        # 我們遍歷 `module_container.named_children()` 以獲取結構和模組類型，
        # 但實際的 `weight` 和 `bias` 來自 `weights_source`。
        temp_input = current_input
        for name, module in module_container.named_children():
            full_name = f"{prefix}_{name}"
            s_key = full_name.replace('.', '_')
            
            # 從 weights_source 獲取對應的模組，以提取其參數 (weight, bias)
            source_module = getattr(weights_source, name)

            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = source_module.weight
                bias = source_module.bias
                
                if s_key in s_params_source:
                    s = s_params_source[s_key]
                    mask = torch.sigmoid(alpha * s)
                    weight = weight * mask # 將權重與稀疏遮罩相乘
                
                if isinstance(module, nn.Conv2d):
                    temp_input = F.conv2d(temp_input, weight, bias, module.stride, 
                                             module.padding, module.dilation, module.groups)
                elif isinstance(module, nn.Linear):
                    temp_input = F.linear(temp_input, weight, bias)
            
            elif len(list(module.children())) > 0:
                 # 遞歸調用子容器。傳遞正確的 source_module 作為 weights_source
                 temp_input = self._forward_sparse_recursively(
                     module, temp_input, full_name, alpha, source_module, s_params_source
                 )
            
            else:
                # 對於非稀疏層 (如 BatchNorm, ReLU 等)，直接應用它們
                temp_input = module(temp_input)
        return temp_input


    def forward(self, x1, x2, alpha, momentum=None):
        """訓練時使用的前向傳播方法。"""
        # --- 線上分支 (Online Branch) ---
        # 線上編碼器使用自己的可訓練權重 (self.encoder) 和可訓練 s_params (self.s_params)
        y1_online = self._forward_sparse_recursively(
            self.encoder, x1, "encoder", alpha, self.encoder, self.s_params
        ).mean([2, 3])
        # 線上投影器使用自己的可訓練權重 (self.projector) 和可訓練 s_params (self.s_params)
        z1_online = self._forward_sparse_recursively(
            self.projector, y1_online, "projector", alpha, self.projector, self.s_params
        )
        p1 = self.predictor(z1_online)

        # --- 目標分支 (Target Branch) ---
        if momentum is not None:
            self.momentum = momentum
        self._update_target_network_ema() # 此方法現在會更新 dense_target_weights 和 target_s_params

        with torch.no_grad(): # 目標分支不參與梯度計算
            # 目標編碼器使用 EMA 權重 (self.dense_target_encoder) 和 EMA 稀疏參數 (self.target_s_params)
            y2_target = self._forward_sparse_recursively(
                self.encoder, x2, "encoder", alpha, self.dense_target_encoder, self.target_s_params
            ).mean([2, 3])
            # 目標投影器使用 EMA 權重 (self.dense_target_projector) 和 EMA 稀疏參數 (self.target_s_params)
            z2_target = self._forward_sparse_recursively(
                self.projector, y2_target, "projector", alpha, self.dense_target_projector, self.target_s_params
            )

        return p1, z2_target

    @torch.no_grad()
    def inference(self, x, use_hard_mask=True, threshold=0.5):
        """使用訓練好的線上網路權重和學到的稀疏結構來提取特徵。"""
        # 推論時，我們使用線上網路的最終權重和學到的 s_params。
        # 目標網路及其 EMA 參數僅用於訓練過程。
        
        inference_alpha = 100.0 # 一個大 alpha 值可以讓 sigmoid 更接近 0/1

        def inference_recursive(module_container, current_input, prefix):
            if isinstance(module_container, InvertedResidual):
                if module_container.stride == 1:
                    x1, x2 = current_input.chunk(2, dim=1)
                    branch2_out = inference_recursive(module_container.branch2, x2, f"{prefix}_branch2")
                    out = torch.cat((x1, branch2_out), dim=1)
                else:
                    branch1_out = inference_recursive(module_container.branch1, current_input, f"{prefix}_branch1")
                    branch2_out = inference_recursive(module_container.branch2, current_input, f"{prefix}_branch2")
                    out = torch.cat((branch1_out, branch2_out), dim=1)
                return channel_shuffle(out, 2)

            # Note: For inference, we use the original module's weights and the learned s_params.
            # So, source_module for weights is 'module' itself.
            temp_input = current_input
            for name, module in module_container.named_children():
                full_name = f"{prefix}_{name}"
                s_key = full_name.replace('.', '_')
                
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    weight = module.weight # 推論時直接使用模組本身的權重
                    bias = module.bias
                    
                    if s_key in self.s_params:
                        s = self.s_params[s_key]
                        m = torch.sigmoid(inference_alpha * s) # 使用大 alpha 得到接近 0/1 的軟遮罩
                        mask = (m > threshold).float() if use_hard_mask else m
                        weight = weight * mask
                    
                    if isinstance(module, nn.Conv2d):
                        temp_input = F.conv2d(temp_input, weight, bias, module.stride, 
                                                 module.padding, module.dilation, module.groups)
                    elif isinstance(module, nn.Linear):
                        temp_input = F.linear(temp_input, weight, bias)
                
                elif len(list(module.children())) > 0:
                     temp_input = inference_recursive(module, temp_input, full_name)
                
                else:
                    temp_input = module(temp_input)
            return temp_input

        # 執行推論
        features = inference_recursive(self.encoder, x, "encoder")
        return features.mean([2, 3])