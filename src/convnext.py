"""
A minimal ConvNext implementation adapted from torch vision
https://pytorch.org/vision/main/_modules/torchvision/models/convnext.html

To instantiate these models, pass in the number of classes:
conv_next_single_block_model = convnext_single_block_hyperspectral(num_classes=3)
dropout_conv_next_single_block_model = convnext_single_block_hyperspectral_dropout(num_classes=3)
conv_next_model = convnext_minimal_hyperspectral(num_classes=3)
dropout_conv_next_model = convnext_minimal_hyperspectral_dropout(num_classes=3)
"""

from functools import partial
from typing import Callable, Optional, Tuple, Union, Any, List, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from src.ops.misc import Conv2dNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        drop_prob: float = 0.0
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True),
            torch.nn.Dropout(p=drop_prob),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        input_channels: int =3,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        drop_prob: float = 0.0,
        final_layer_drop_prob: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                input_channels,
                firstconv_output_channels,
                kernel_size=3,  # change from convnext original of 4
                stride=1,  # change from convnext original of 4
                padding=1,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        layers.append(torch.nn.Dropout(drop_prob))

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                if stochastic_depth_prob > 0:
                    # adjust stochastic depth probability based on the depth of the stage block
                    sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0 + 1e-6)
                else:
                    sd_prob = 0.0
                stage.append(block(cnf.input_channels, layer_scale, sd_prob, drop_prob=drop_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.final_dropout = torch.nn.Dropout(final_layer_drop_prob)
        if final_layer_drop_prob > 0.0:
            print(f"Using final layer dropout with drop probability of {final_layer_drop_prob}")
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
        x = self.final_dropout(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _convnext(
    block_setting: List[CNBlockConfig],
    input_channels: int = 3,
    stochastic_depth_prob: float = 0.0,
    drop_prob: float = 0.0,
    weights: Optional[str]=None,
    **kwargs: Any,
) -> ConvNeXt:

    model = ConvNeXt(block_setting, input_channels=input_channels, stochastic_depth_prob=stochastic_depth_prob, drop_prob=drop_prob, **kwargs)

    if weights is not None:
        model.load_state_dict(torch.load(weights))

    return model


def convnext_tiny(*, weights: Optional[str] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Tiny model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Tiny_Weights
        :members:
    """
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


# TODO: Train these 4 models
def convnext_minimal_hyperspectral(*, weights: Optional[str] = None, drop_prob: float = 0.0, **kwargs: Any) -> ConvNeXt:
    """

    An adapted, minimal ConvNeXt model architecture for 114-channel hyperspectral data.
    Implementation adapted from torchvision based on the paper `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights str: The path to pretrained weights to use. By default, no pre-trained weights are used.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Tiny_Weights
        :members:
    """
    block_setting = [
        CNBlockConfig(128, 192, 3),
        CNBlockConfig(192, None, 1),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.0)
    return _convnext(block_setting, stochastic_depth_prob=stochastic_depth_prob, drop_prob=drop_prob, weights=weights, **kwargs)


def convnext_single_block_hyperspectral(*, weights: Optional[str] = None, drop_prob: float = 0.0, **kwargs: Any) -> ConvNeXt:
    """

    An adapted, minimal ConvNeXt model architecture for 114-channel hyperspectral data.
    Implementation adapted from torchvision based on the paper `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights str: The path to pretrained weights to use. By default, no pre-trained weights are used.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Tiny_Weights
        :members:
    """
    block_setting = [
        CNBlockConfig(128, None, 1),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.0)
    return _convnext(block_setting, stochastic_depth_prob=stochastic_depth_prob, drop_prob=drop_prob, weights=weights, **kwargs)


convnext_minimal_hyperspectral_dropout = partial(convnext_minimal_hyperspectral, drop_prob=0.2)
convnext_single_block_hyperspectral_dropout = partial(convnext_single_block_hyperspectral, drop_prob=0.2)
convnext_minimal_hyperspectral_final_layer_dropout = partial(convnext_minimal_hyperspectral, final_drop_prob=0.2, drop_prob=0.0)
