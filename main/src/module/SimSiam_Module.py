# main/src/module/SimSiam_Module.py
from torch import Tensor, tensor
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
import warnings # 引入warnings以便在輸出維度不匹配時發出警告


class SimSiam(nn.Module):
    def __init__(
        self,
        pretrained_model,
        model_type='shufflenet',  # <--- 新增參數：指定模型類型
        encoder_output_dim=1024,
        projector_inner_dim=256,
    ):
        super(SimSiam, self).__init__()

        # --- 1. 根據 model_type 構建 encoder ---
        if model_type == 'shufflenet':
            # 原本的 ShuffleNetV2_0_5 結構
            # 預期的 encoder_output_dim 可能是 1024
            if encoder_output_dim != 1024:
                warnings.warn(f"For 'shufflenet' type, encoder_output_dim is typically 1024. Received: {encoder_output_dim}")

            self.encoder = nn.Sequential(
                pretrained_model.conv1,
                pretrained_model.maxpool,
                pretrained_model.stage2,
                pretrained_model.stage3,
                pretrained_model.stage4,
                pretrained_model.conv5,
            )

        elif model_type == 'resnet':
            # ResNet18/34 等標準 PyTorch ResNet 結構
            # 預期的 encoder_output_dim 應該是 512
            if encoder_output_dim != 512:
                warnings.warn(f"For standard 'resnet' type (e.g., resnet18), encoder_output_dim is typically 512. Received: {encoder_output_dim}")

            # 將 ResNet 的所有特徵層組合成 encoder，忽略最後的 avgpool 和 fc 層
            self.encoder = nn.Sequential(
                pretrained_model.conv1,
                pretrained_model.bn1,
                pretrained_model.relu,
                pretrained_model.maxpool,
                pretrained_model.layer1,
                pretrained_model.layer2,
                pretrained_model.layer3,
                pretrained_model.layer4,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Must be 'shufflenet' or 'resnet'.")

        # --- 2. 構建 projector (與原先相同) ---
        # build a 3-layer projector
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_output_dim, projector_inner_dim, bias=False),
            nn.BatchNorm1d(projector_inner_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(projector_inner_dim, projector_inner_dim, bias=False),
            nn.BatchNorm1d(projector_inner_dim),
            nn.ReLU(inplace=True),  # second layer
            nn.Linear(projector_inner_dim, projector_inner_dim, bias=False),
            nn.BatchNorm1d(projector_inner_dim, affine=False),  # third layer
        )  # output layer

        # --- 3. 構建 predictor (與原先相同) ---
        """
        according to the original paper, 
        predictor's output and projector's output vector should be the same size to calculate loss.
        Meanwhile, the predictor's inner dimmension should be 1/4 of predictor's output dimmension.
        """
        predictor_output_dim = projector_inner_dim
        predictor_inner_dim = predictor_output_dim // 4
        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(predictor_output_dim, predictor_inner_dim, bias=False),
            nn.BatchNorm1d(predictor_inner_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(predictor_inner_dim, predictor_output_dim),
        )  # output layer

    def forward(self, x1, x2):
        # 這裡的 .mean([2, 3]) 執行了 Global Average Pooling (GAP)
        # GAP 是 SimSiam 在將特徵圖輸入 Projector 前的標準操作。
        y1 = self.encoder(x1).mean([2, 3]) 
        y2 = self.encoder(x2).mean([2, 3])
        z1 = self.projector(y1)
        z2 = self.projector(y2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


class SimSiam_online(SimSiam):
    # 繼承SimSiam類別
    def __init__(self, *args, **kwargs):
        # 確保所有參數正確傳遞給 SimSiam.__init__
        super(SimSiam_online, self).__init__(*args, **kwargs)

    def forward(self, x1):
        # ... (其餘部分不變)
        y1 = self.encoder(x1).mean([2, 3])
        z1 = self.projector(y1)
        p1 = self.predictor(z1)
        return p1


class SimSiam_target(SimSiam):
    def __init__(self, *args, **kwargs):
        temp_online = kwargs.pop("online")
        assert isinstance(temp_online, SimSiam_online), "online network is required"
        # 確保所有參數正確傳遞給 SimSiam.__init__ (包括 model_type 和 encoder_output_dim)
        super(SimSiam_target, self).__init__(*args, **kwargs)
        self.target = temp_online  # 直接共享參數

    def forward(self, x1):
        with torch.no_grad():  # 確保 target 網路不參與梯度計算
            y1 = self.target.encoder(x1).mean([2, 3])
            z1 = self.target.projector(y1)
        return z1.detach() 


class SimSiamLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def D(self, p, z):
        # p: NxC
        # z: NxC
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1)  # dot product & negation

    def calculate_L1(self, p1, z2):
        """計算 L1 = D(p1, z2.detach())"""
        return self.D(p1, z2)

    def calculate_L2(self, p2, z1):
        """計算 L2 = D(p2, z1.detach())"""
        return self.D(p2, z1)

    def forward(self, p1, p2, z1, z2):
        """計算總損失"""
        l1 = self.calculate_L1(p1, z2)
        l2 = self.calculate_L2(p2, z1)
        loss = 0.5 * (l1.mean() + l2.mean())  # 計算平均損失
        return loss

    def forward_components(self, p1, p2, z1, z2):
        """返回 L1 和 L2 的 batch 平均損失"""
        l1_batch_mean = self.calculate_L1(p1, z2).mean()
        l2_batch_mean = self.calculate_L2(p2, z1).mean()
        return l1_batch_mean, l2_batch_mean


class SimSiamLoss_unsymmetric(nn.Module):
    def __init__(self):
        super(SimSiamLoss_unsymmetric, self).__init__()

    def forward(self, p1, z2):
        # normalize projection output
        p1 = nn.functional.normalize(p1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # negative cosine similarity
        loss = -(p1 * z2).sum(dim=1).mean()

        return loss