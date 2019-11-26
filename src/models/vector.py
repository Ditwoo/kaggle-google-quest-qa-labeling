import torch
import torch.nn as nn
from collections import OrderedDict


class LinearModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1):
        super(LinearModel, self).__init__()

        hidden_dim = 512
        dropout_rate = 0.2
        self.layers = nn.Sequential(OrderedDict([
            ("input", nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout_rate),
            )),
            ("head", nn.Linear(hidden_dim, out_dim))
        ]))

    def forward(self, x):
        return self.layers(x)
