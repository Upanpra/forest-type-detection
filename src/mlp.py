from typing import List

import torch.nn.functional as F
from torch import nn


class NeuralNets(nn.Module):

    def __init__(self,
         input_channels: int =3,
        num_classes: int = 3,
                 num_hidden_features: int = 192,
                 num_hidden_layers: int = 4,

                 ):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Linear(input_channels, num_hidden_features))
        layers.append(nn.GELU())
        for _ in range(num_hidden_layers - 2):
            layers.append(nn.BatchNorm1d(num_hidden_features))  # TODO try to get layernorm working for more parity w/ convnext
            layers.append(nn.Linear(input_channels, num_hidden_features))
            layers.append(nn.GELU())

        layers.append(
            nn.BatchNorm1d(num_hidden_features))  # TODO try to get layernorm working for more parity w/ convnext
        layers.append(nn.Linear(input_channels, num_hidden_features))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_hidden_features, num_classes)

    def forward(self, x):
        # x is image of shape b, c, h, w
        b, c, h, w = x.shape
        x = x.reshape([b, -1])
        x = self.features(x)
        x =
        return x
