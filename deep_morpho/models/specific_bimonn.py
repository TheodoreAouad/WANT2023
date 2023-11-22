from abc import ABC
from typing import List, Dict, Tuple
import copy

import torch
import torch.nn as nn
import numpy as np

from .bisel import BiSEL
from .binary_nn import BinaryNN, BinarySequential
from .dense_lui import DenseLUI
from .layers_not_binary import DenseLuiNotBinary
from ..initializer import InitBiseEnum, BimonnInitializer, BiselInitializer


# TODO: Merge with Bimonn class
class BimonnDenseBase(BinaryNN, ABC):
    last_layer: DenseLUI

    def __init__(
        self,
        channels: List[int],
        input_size: int,
        n_classes: int,
        initializer_method: InitBiseEnum = InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
        initializer_args: Dict = {"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", },
        input_mean: float = .5,
        apply_last_activation: bool = True,
        *args,
        **kwargs
    ):
        super().__init__()

        self.input_size = input_size
        self.n_classes = n_classes
        self.channels = [np.prod(input_size)] + channels + [n_classes]
        self.apply_last_activation = apply_last_activation

        self.initializer_method = initializer_method

        self.initializer_args = initializer_args
        self.initializer_args["input_mean"] = .5  # All layers except the first one

        self.input_mean = input_mean

        self.first_init_args = {k: v for k, v in initializer_args.items() if k != "input_mean"}
        self.first_init_args["input_mean"] = input_mean

        self.initializer_fn = BimonnInitializer.get_init_class(initializer_method, atomic_element="bisel")

        self.flatten = nn.Flatten()
        # self.layers = []
        self.layers = BinarySequential()

        if len(self.channels) > 2:  # If len==2, then there is only one layer.
            # self.dense1 = DenseLUI(
            #     in_channels=self.channels[0],
            #     out_channels=self.channels[1],
            #     initializer=self.initializer_fn(**self.first_init_args),
            #     **kwargs
            # )
            self.layers.append(DenseLUI(
                in_channels=self.channels[0],
                out_channels=self.channels[1],
                initializer=self.initializer_fn(**self.first_init_args),
                **kwargs
            ))


        for idx, (chin, chout) in enumerate(zip(self.channels[1:-2], self.channels[2:-1]), start=2):
            # setattr(self, f"dense{idx}", DenseLUI(
            #     in_channels=chin,
            #     out_channels=chout,
            #     initializer=self.initializer_fn(**initializer_args),
            #     **kwargs
            # ))
            # self.layers.append(getattr(self, f"dense{idx}"))
            self.layers.append(DenseLUI(
                in_channels=chin,
                out_channels=chout,
                initializer=self.initializer_fn(**initializer_args),
                **kwargs
            ))

        last_kwargs = copy.deepcopy(kwargs)
        if not self.apply_last_activation:
            last_kwargs["threshold_mode"]["activation"] = "identity"

        self.layers.append(self.last_layer(
            in_channels=self.channels[-2],
            out_channels=self.channels[-1],
            initializer=self.initializer_fn(**initializer_args),
            **last_kwargs
        ))

    @property
    def classification_layer(self):
        return self.layers[-1]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_save(self, x):
        """ Saves all intermediary outputs.
        Args:
            x (torch.Tensor): input images of size (batch_size, channel, width, height)

        Returns:
            list: list of dict. list[layer][output_lui_channel], list[layer][output_bisel_inchan, output_bisel_outchan]
        """
        output = {"input": x}
        cur = self.layers[0].forward_save(x)
        output[0] = cur
        for layer_idx, layer in enumerate(self.layers[1:], start=1):
            cur = layer.forward_save(cur['output'])
            output[layer_idx] = cur
        output["output"] = cur["output"]
        return output

    @classmethod
    def default_args(cls) -> Dict[str, dict]:
        """Return the default arguments of the model, in the format of argparse.ArgumentParser"""
        res = super().default_args()
        res.update({
            k: v for k, v in DenseLUI.default_args().items()
            if k not in res
            and k not in [
                "initializer", "in_channels", "out_channels", "bise_module", "lui_module", "groups",
            ]
        })
        return res


class BimonnDense(BimonnDenseBase):
    last_layer = DenseLUI


class BimonnDenseNotBinary(BimonnDenseBase):
    last_layer = DenseLuiNotBinary


class BiselMaxPoolBlock(BinaryNN):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int],
        initializer: BiselInitializer,
        *args,
        **kwargs
    ):
        super().__init__()
        self.bisel = BiSEL(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            initializer=initializer,
            *args, **kwargs
        )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.bisel(x)
        x = self.maxpool(x)
        return x

    def forward_save(self, x):
        """ Saves all intermediary outputs.
        Args:
            x (torch.Tensor): input images of size (batch_size, channel, width, height)

        Returns:
            list: list of dict. list[layer][output_lui_channel], list[layer][output_bisel_inchan, output_bisel_outchan]
        """
        output = {"input": x}
        cur = self.bisel.forward_save(x)
        output["bisel"] = cur
        cur = self.maxpool(cur["output"])
        output["maxpool"] = cur
        output["output"] = cur
        return output


class BimonnBiselDenseBase(BinaryNN, ABC):
    last_layer: DenseLUI
    def __init__(
        self,
        kernel_size: Tuple[int],
        channels: List[int],
        input_size: Tuple[int],
        n_classes: int,
        initializer_bise_method: InitBiseEnum = InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
        initializer_bise_args: Dict = {"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": .5},
        initializer_lui_method: InitBiseEnum = InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_RANDOM_BIAS,
        initializer_lui_args: Dict = None,
        input_mean: float = .5,
        apply_last_activation: bool = True,
        *args,
        **kwargs
    ):
        super().__init__()
        assert len(channels) >= 2, "At least 2 layers are required, one for BiSEL, one for DenseLUI"
        self.input_dense = np.array(input_size)
        self.input_size = np.array(input_size)
        self.n_classes = n_classes
        self.channels = [input_size[0]] + channels + [n_classes]
        self.apply_last_activation = apply_last_activation

        self.initializer_bise_method = initializer_bise_method
        self.initializer_bise_args = initializer_bise_args
        self.initializer_lui_method = initializer_lui_method if initializer_lui_method is not None else initializer_bise_method
        self.initializer_lui_args = initializer_lui_args if initializer_lui_args is not None else copy.deepcopy(initializer_bise_args)
        self.input_mean = input_mean

        self.first_init_args = {k: v for k, v in self.initializer_bise_args.items() if k != "input_mean"}
        self.first_init_args["input_mean"] = input_mean

        self.initializer_bise_fn = BimonnInitializer.get_init_class(self.initializer_bise_method, atomic_element="bisel")
        self.initializer_lui_fn = BimonnInitializer.get_init_class(self.initializer_lui_method, atomic_element="bisel")

        self.bisel_layers = BinarySequential()

        self.bisel_layers.append(BiselMaxPoolBlock(
            in_channels=self.channels[0],
            out_channels=self.channels[1],
            kernel_size=kernel_size,
            initializer=BiselInitializer(
                bise_initializer=self.initializer_bise_fn(**self.first_init_args),
                lui_initializer=self.initializer_lui_fn(**self.initializer_lui_args),
            ),
            *args, **kwargs
        ))


        self.input_dense[1:] = np.array(self.input_dense[1:]) // 2

        for idx, (chin, chout) in enumerate(zip(self.channels[1:-3], self.channels[2:-2]), start=2):
            self.bisel_layers.append(BiselMaxPoolBlock(
                in_channels=chin,
                out_channels=chout,
                kernel_size=kernel_size,
                initializer=BiselInitializer(
                    bise_initializer=self.initializer_bise_fn(**self.initializer_bise_args),
                    lui_initializer=self.initializer_lui_fn(**self.initializer_lui_args),
                ),
                *args, **kwargs
            ))
            self.input_dense[1:] = self.input_dense[1:] // 2

        self.flatten = nn.Flatten()
        self.dense_layers = BinarySequential()

        self.dense_layers.append(DenseLUI(
            in_channels=self.channels[-3] * np.prod(self.input_dense[1:]),
            out_channels=self.channels[-2],
            initializer=self.initializer_bise_fn(**initializer_bise_args),
            **kwargs
        ))

        last_kwargs = copy.deepcopy(kwargs)
        if not self.apply_last_activation:
            last_kwargs["threshold_mode"]["activation"] = "identity"

        self.dense_layers.append(self.last_layer(
            in_channels=self.channels[-2],
            out_channels=self.channels[-1],
            initializer=self.initializer_bise_fn(**initializer_bise_args),
            **last_kwargs
        ))

    @property
    def layers(self):
        return BinarySequential(self.bisel_layers, self.flatten, self.dense_layers, binary_mode=self.binary_mode)

    @property
    def classification_layer(self):
        return self.dense_layers[-1]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.bisel_layers(x)
        x = self.flatten(x)
        x = self.dense_layers(x)

        return x

    def forward_save(self, x):
        """ Saves all intermediary outputs.
        Args:
            x (torch.Tensor): input images of size (batch_size, channel, width, height)

        Returns:
            list: list of dict. list[layer][output_lui_channel], list[layer][output_bisel_inchan, output_bisel_outchan]
        """
        output = {"input": x}
        cur = self.bisel_layers[0].forward_save(x)
        output[0] = cur
        for layer_idx, layer in enumerate(self.bisel_layers[1:], start=1):
            cur = layer.forward_save(cur['output'])
            output[layer_idx] = cur

        cur_idx = len(self.bisel_layers)

        x = self.flatten(cur['output'])
        cur = self.dense_layers[0].forward_save(x)
        output[cur_idx] = cur
        for layer_idx, layer in enumerate(self.dense_layers[1:], start=cur_idx + 1):
            cur = layer.forward_save(cur['output'])
            output[layer_idx] = cur

        output["output"] = cur["output"]
        return output

    @classmethod
    def default_args(cls) -> Dict[str, dict]:
        """Return the default arguments of the model, in the format of argparse.ArgumentParser"""
        res = super().default_args()
        res.update({
            k: v for k, v in DenseLUI.default_args().items()
            if k not in res
            and k not in [
                "initializer", "in_channels", "out_channels", "bise_module", "lui_module", "groups",
            ]
        })
        return res


class BimonnBiselDense(BimonnBiselDenseBase):
    last_layer = DenseLUI


class BimonnBiselDenseNotBinary(BimonnBiselDenseBase):
    last_layer = DenseLuiNotBinary
