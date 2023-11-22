from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_nn import BinaryNN, BinarySequential
from .ste_activation import STEClippedIdentity, _STEClippedIdentity, STELayer


class BNNLayer(BinaryNN, ABC):
    """Binary layer, Courbariaux et al. 2016, https://arxiv.org/abs/1602.02830
    """
    LAYER: nn.Module
    STE_BINARIZATION = _STEClippedIdentity.apply

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = self.LAYER(bias=None, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.clip_weights()  # Clipping weights onto [-1, 1]
        return self._forward(x, self.binarized_weight)

    def real_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.clip_weights()  # Clipping weights onto [-1, 1]
        return self._forward(x, self.weight)

    def clip_weights(self):
        self.layer.weight.data = torch.clip(self.weight.data, -1, 1)

    @abstractmethod
    def _forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        pass

    def _specific_numel_binary(self):
        return self.weight.numel()

    @property
    def weight(self):
        return self.layer.weight

    @property
    def binarized_weight(self):
        return self.STE_BINARIZATION(self.weight)


class BNNLinear(BNNLayer):
    LAYER = nn.Linear

    def _forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight, bias=None)


class BNNConv2d(BNNLayer):
    LAYER = nn.Conv2d

    def _forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.layer._conv_forward(x, weight, bias=None)


class BNNConvBlock(BinaryNN):
    STE_BINARIZATION = STEClippedIdentity

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[Tuple[int], int], do_batchnorm: bool = True):
        super().__init__()
        self.conv = BNNConv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same", padding_mode="replicate")
        self.do_batchnorm = do_batchnorm
        if self.do_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.ste = self.STE_BINARIZATION()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.do_batchnorm:
            x = self.bn(x)
        return self.ste(x)

    def clip_weights(self):
        self.conv.clip_weights()

    @property
    def weight(self):
        return self.conv.weight

    @property
    def binarized_weight(self):
        return self.conv.binarized_weight


class BNNConv(BinaryNN):
    """Implentation of Binarized Neural Network, Courbariaux et al. 2016, https://arxiv.org/abs/1602.02830
    for a non shrinking convolutional layer.
    """

    def __init__(self, kernel_size: Union[Tuple[int], int], channels: List[int], do_batchnorm: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.channels = channels
        self.layers = BinarySequential()
        self.do_batchnorm = do_batchnorm

        for chin, chout in zip(channels[:-1], channels[1:]):
            self.layers.append(BNNConvBlock(chin, chout, kernel_size=kernel_size, do_batchnorm=self.do_batchnorm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def real_forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, (BNNLayer, STELayer)):
                x = layer.real_forward(x)
            else:
                x = layer(x)
        return x

    @property
    def in_channels(self):
        return self.channels[:-1]

    @property
    def out_channels(self):
        return self.channels[1:]
