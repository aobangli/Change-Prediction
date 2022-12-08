import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, n):
        super(LinearModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)
        return out
