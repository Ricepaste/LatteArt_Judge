import torch.nn as nn
import copy


class OnlineNetwork(nn.Module):
    def __init__(self, pretrained_model):
        super(OnlineNetwork, self).__init__()
        self.encoder = nn.Sequential(
            pretrained_model.features, pretrained_model.avgpool
        )
        self.projector = nn.Sequential(
            nn.Linear(1280, 256),  # 根據 encoder 的輸出大小調整
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )
        self.predictor = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Linear(256, 128)
        )

    def forward(self, x):
        z = self.encoder(x)
        p = self.projector(z)
        q = self.predictor(p)
        return z, p, q


class TargetNetwork(nn.Module):
    def __init__(self, online_network, momentum=0.99):
        super(TargetNetwork, self).__init__()
        self.momentum = momentum
        self.online_network = online_network
        self.target_network = copy.deepcopy(online_network)
        for param in self.target_network.parameters():
            param.requires_grad = False

    def update_target_network(self):
        for online_param, target_param in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            target_param.data = (
                self.momentum * target_param.data
                + (1.0 - self.momentum) * online_param.data
            )

    def forward(self, x):
        z = self.target_network.encoder(x)
        p = self.target_network.projector(z)
        return z, p
