import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)
        return out
