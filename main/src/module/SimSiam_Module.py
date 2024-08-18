import torch.nn as nn
import torch.nn.functional as F
import copy


class SimSiam(nn.Module):
    def __init__(self, pretrained_model, dim=2048, pred_dim=512):
        super(SimSiam, self).__init__()

        # create the encoder
        self.encoder = nn.Sequential(
            pretrained_model.features, pretrained_model.avgpool
        )

        # build a 3-layer projector
        prev_dim = 1280
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            nn.Linear(prev_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False),  # third layer
        )  # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim),
        )  # output layer

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


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
