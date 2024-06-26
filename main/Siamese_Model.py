import torch.nn as nn
import torch.nn.functional as F
import torch


class SNN(nn.Module):

    def __init__(self, pretrained_model):
        super(SNN, self).__init__()
        features_and_avgpool = nn.Sequential(
            pretrained_model.features, pretrained_model.avgpool
        )
        self.feature_extractor = features_and_avgpool
        self.classfier = nn.Sequential(
            nn.Linear(2 * 1280 * 1 * 1, 10000),
            nn.ReLU(inplace=True),
            nn.Linear(10000, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        x1 = x1.view(-1, 1280 * 1 * 1)
        x2 = x2.view(-1, 1280 * 1 * 1)
        x = torch.cat((x1, x2), dim=1)
        x = self.classfier(x)
        return x


# torch.nn.FeatureAlphaDropout(p=0.2, inplace=False),

"""
# change the last layer of the b3 model to fit our problem
model._modules['classifier'] = torch.nn.Sequential(
    # torch.nn.FeatureAlphaDropout(p=0.2, inplace=False),
    torch.nn.Linear(1280, 20000),
    torch.nn.Sigmoid(),
    torch.nn.Dropout(p=0.2, inplace=False),
    torch.nn.Linear(20000, 5000),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(p=0.2, inplace=False),
    torch.nn.Linear(5000, 100),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(p=0.2, inplace=False),
    torch.nn.Linear(100, 1),
    # torch.nn.Linear(1280, 1),
)
"""
