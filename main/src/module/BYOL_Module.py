import torch.nn as nn
import torch.nn.functional as F
import copy


class OnlineNetwork(nn.Module):
    def __init__(self, pretrained_model):
        super(OnlineNetwork, self).__init__()
        self.encoder = nn.Sequential(
            pretrained_model.features, pretrained_model.avgpool
        )
        self.projector = nn.Sequential(
            nn.Flatten(),  # 添加展平層
            nn.Linear(1280, 256),  # 根據 encoder 的輸出大小調整
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )
        self.predictor = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Linear(256, 128)
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.projector(z1)
        p2 = self.projector(z2)
        q1 = self.predictor(p1)
        q2 = self.predictor(p2)
        return (q1, q2)


class TargetNetwork(nn.Module):
    def __init__(self, online_network, momentum=0.99):
        super(TargetNetwork, self).__init__()
        self.momentum = momentum
        self.online_network = online_network
        self.target_network = copy.deepcopy(online_network)
        for param in self.target_network.parameters():
            param.requires_grad = False

    def update_target_network(self):
        """
        需要在每個 epoch 結束後呼叫此函數，更新 target network 的參數
        """
        for online_param, target_param in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            target_param.data = (
                self.momentum * target_param.data
                + (1.0 - self.momentum) * online_param.data
            )

    def forward(self, x1, x2):
        z1 = self.target_network.encoder(x1)
        z2 = self.target_network.encoder(x2)
        p1 = self.target_network.projector(z1)
        p2 = self.target_network.projector(z2)
        return (p1.detach(), p2.detach())  # 確保 target 的輸出不反向傳播梯度


class BYOLLoss(nn.Module):
    def __init__(self):
        super(BYOLLoss, self).__init__()

    def forward(self, online_output, target_output):
        q1, q2 = online_output
        p1_target, p2_target = target_output

        loss1 = F.mse_loss(q1, p1_target)
        loss2 = F.mse_loss(q2, p2_target)

        return loss1 + loss2
