from torch import Tensor, tensor
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch


class SimSiam(nn.Module):
    def __init__(
        self,
        pretrained_model,
        encoder_output_dim=1024,
        projector_inner_dim=256,
    ):
        super(SimSiam, self).__init__()

        # create the encoder
        self.encoder = nn.Sequential(
            pretrained_model.conv1,
            pretrained_model.maxpool,
            pretrained_model.stage2,
            pretrained_model.stage3,
            pretrained_model.stage4,
            pretrained_model.conv5,
        )

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

        """
        according to the original paper, 
        predictor's output and projector's output vector should be the same size to calculate loss.
        Meanwhile, the predictor's inner dimmension should be 1/4 of predictor's output dimmension.
        """
        predictor_output_dim = projector_inner_dim
        predictor_inner_dim = predictor_output_dim // 4
        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(projector_inner_dim, predictor_inner_dim, bias=False),
            nn.BatchNorm1d(predictor_inner_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(predictor_inner_dim, predictor_output_dim),
        )  # output layer

    def forward(self, x1, x2):
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
        # temp_proj = kwargs.pop("shared_projector", None)
        # temp_pred = kwargs.pop("shared_predictor", None)
        super(SimSiam_online, self).__init__(*args, **kwargs)
        # self.projector = temp_proj
        # self.predictor = temp_pred

    def forward(self, x1):
        y1 = self.encoder(x1).mean([2, 3])
        z1 = self.projector(y1)
        p1 = self.predictor(z1)
        return p1


class SimSiam_target(SimSiam):
    def __init__(self, *args, **kwargs):
        temp_online = kwargs.pop("online")
        assert isinstance(temp_online, SimSiam_online), "online network is required"
        super(SimSiam_target, self).__init__(*args, **kwargs)
        self.target = temp_online  # 直接共享參數

    def forward(self, x1):
        with torch.no_grad():  # 確保 target 網路不參與梯度計算
            y1 = self.target.encoder(x1).mean([2, 3])
            z1 = self.target.projector(y1)
        return z1.detach().requires_grad_()  # 額外 detach() 避免警告


class SimSiamLoss(nn.Module):
    def __init__(self):
        super(SimSiamLoss, self).__init__()

    def forward(self, p1, p2, z1, z2):
        # normalize projection output
        p1 = nn.functional.normalize(p1, dim=1)
        p2 = nn.functional.normalize(p2, dim=1)
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # negative cosine similarity
        loss = -(p1 * z2).sum(dim=1).mean() / 2 - (p2 * z1).sum(dim=1).mean() / 2

        return loss


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
