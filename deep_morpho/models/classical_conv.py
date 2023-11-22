from typing import Tuple, List, Callable
import torch
import torch.nn as nn
import numpy as np

from .binary_nn import BinaryNN


class ConvNetLastLinear(BinaryNN):

    def __init__(
        self,
        kernel_size: int,
        channels: List[int],
        n_classes: int,
        input_size: Tuple[int, int, int],
        do_maxpool: bool = False,
        activation_constructor: Callable = nn.ReLU,
        **kwargs
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = [input_size[0]] + channels + [n_classes]
        self.n_classes = n_classes
        self.input_size = np.array(input_size)
        self.input_dense = np.array(input_size)
        self.do_maxpool = do_maxpool
        self.activation_constructor = activation_constructor

        conv_layers = []
        for (chin, chout) in zip(self.channels[:-3], self.channels[1:-2]):
            conv_layers.append(nn.Conv2d(chin, chout, kernel_size, padding="same",))
            conv_layers.append(activation_constructor())
            if do_maxpool:
                conv_layers.append(nn.MaxPool2d(2))
                self.input_dense[1:] = self.input_dense[1:] // 2

        self.conv_layers = nn.Sequential(*conv_layers)

        # self.conv_layers = nn.Sequential(*
        #     [nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size, padding="same", **kwargs) for i in range(len(self.channels) - 2)]
        # )
        self.flatten = nn.Flatten()

        linears = []

        linears.append(nn.Linear(in_features=self.channels[-3] * np.prod(self.input_dense[1:]), out_features=self.channels[-2]))
        linears.append(activation_constructor())
        linears.append(nn.Linear(in_features=self.channels[-2], out_features=self.channels[-1]))

        self.linear_layers = nn.Sequential(*linears)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)

        return x



class ConvNetBinaryConnectCifar10(BinaryNN):
    """ Conv Net used to train on CIFAR 10 in the paper Binary Connect
    https://proceedings.neurips.cc/paper/2015/file/3e15cc11f979ed25912dff5b0669f2cd-Paper.pdf
    Note: we do not use the SVM last layer, and we do not use the batch norm.
    """
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        n_classes: int,
        activation_constructor: Callable = nn.ReLU,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        alpha = .1
        epsilon = 1e-4

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128, eps=epsilon, momentum=alpha),
            activation_constructor(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128, eps=epsilon, momentum=alpha),
            activation_constructor(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256, eps=epsilon, momentum=alpha),
            activation_constructor(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same"),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256, eps=epsilon, momentum=alpha),
            activation_constructor(),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512, eps=epsilon, momentum=alpha),
            activation_constructor(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512, eps=epsilon, momentum=alpha),
            activation_constructor(),
        )

        self.flatten = nn.Flatten()

        self.linear_block = nn.Sequential(
            nn.Linear(in_features=512 * input_size[1] // (2 ** 3) * input_size[2] // (2 ** 3), out_features=1024),
            nn.BatchNorm1d(1024, eps=epsilon, momentum=alpha),
            activation_constructor(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024, eps=epsilon, momentum=alpha),
            activation_constructor(),
            nn.Linear(in_features=1024, out_features=n_classes),
            nn.BatchNorm1d(n_classes, eps=epsilon, momentum=alpha),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.linear_block(x)

        return x


class MLPBatchNormClassical(BinaryNN):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        channels: List[int],
        n_classes: int,
        activation_constructor: Callable = nn.ReLU,
        alpha: float = .15,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.alpha = alpha
        self.epsilon = epsilon

        self.flatten = nn.Flatten()

        linear_blocks = []

        linear_blocks.append(
            nn.Sequential(
                nn.Linear(in_features=input_size[0] * input_size[1] * input_size[2], out_features=channels[0]),
                nn.BatchNorm1d(channels[0], eps=epsilon, momentum=alpha),
                activation_constructor(),
            )
        )

        for idx, neurons in enumerate(channels[1:]):
            linear_blocks.append(
                nn.Sequential(
                    nn.Linear(in_features=channels[idx], out_features=neurons),
                    nn.BatchNorm1d(neurons, eps=epsilon, momentum=alpha),
                    activation_constructor(),
                )
            )

        self.linear_blocks = nn.Sequential(*linear_blocks)

        self.linear_last = nn.Sequential(
            nn.Linear(in_features=channels[-1], out_features=n_classes),
            nn.BatchNorm1d(n_classes, eps=epsilon, momentum=alpha),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear_blocks(x)
        x = self.linear_last(x)

        return x

    @property
    def num_units(self) -> List[int]:
        return self.channels


class MLPBinaryConnectMNIST(BinaryNN):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        n_classes: int,
        activation_constructor: Callable = nn.ReLU,
        num_units: int = 2048,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        alpha = .15
        epsilon = 1e-4

        # num_units = 2048
        n_hidden_layers = 3

        self.flatten = nn.Flatten()

        linear_blocks = []

        linear_blocks.append(
            nn.Sequential(
                nn.Linear(in_features=input_size[0] * input_size[1] * input_size[2], out_features=num_units),
                nn.BatchNorm1d(num_units, eps=epsilon, momentum=alpha),
                activation_constructor(),
            )
        )

        for _ in range(n_hidden_layers - 1):
            linear_blocks.append(
                nn.Sequential(
                    nn.Linear(in_features=num_units, out_features=num_units),
                    nn.BatchNorm1d(num_units, eps=epsilon, momentum=alpha),
                    activation_constructor(),
                )
            )

        self.linear_blocks = nn.Sequential(*linear_blocks)

        self.linear_last = nn.Sequential(
            nn.Linear(in_features=num_units, out_features=n_classes),
            nn.BatchNorm1d(n_classes, eps=epsilon, momentum=alpha),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear_blocks(x)
        x = self.linear_last(x)

        return x
