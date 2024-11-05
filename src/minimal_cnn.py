"""
A minimal CNN implementation that contains a single conv layer and single non-linearity

To instantiate these models, pass in the number of classes:
model = min_cnn(num_classes=3)
dropout_model = min_cnn_dropout(num_classes=3)
"""

from functools import partial
from typing import Callable, Optional, Any, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from src.ops.misc import Conv2dNormActivation


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class MinCNN(nn.Module):
    def __init__(
        self,
        input_channels: int =44,
        firstconv_output_channels: int = 128,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        drop_prob: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        layers.append(
            Conv2dNormActivation(
                input_channels,
                firstconv_output_channels,
                kernel_size=3,
                stride=1,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=nn.GELU,
                bias=True,
            )
        )

        layers.append(torch.nn.Dropout(drop_prob))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastconv_output_channels = firstconv_output_channels
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def min_cnn(drop_prob=0.0, **kwargs):
    return MinCNN(drop_prob=drop_prob, **kwargs)


min_cnn_dropout = partial(min_cnn, drop_prob=0.2)
