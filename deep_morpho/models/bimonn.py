from abc import ABC
from typing import List, Tuple, Union, Dict, Optional, Any, Callable
import inspect
import copy

import numpy as np
import torch.nn as nn

from .bisel import BiSEL, SyBiSEL, BiSELBase
from .layers_not_binary import BiSELNotBinary, SyBiSELNotBinary
from .binary_nn import BinaryNN, BinarySequential
from ..initializer import BimonnInitializer, InitBimonnEnum, BimonnInitInputMean, InitBiseEnum


class BiMoNN(BinaryNN):

    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        channels: List[int],
        atomic_element: Union[str, List[str]] = 'bisel',
        initializer_method: InitBimonnEnum = InitBimonnEnum.INPUT_MEAN,
        initializer_args: Dict = {
            "bise_init_method": InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
            "lui_init_method": InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_RANDOM_BIAS,
            "bise_init_args": {"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto"},
            "input_mean": .5,
        },
        **kwargs,
    ):
        super().__init__()
        self.length = max([len(value) if isinstance(value, list) else 0 for value in [kernel_size, atomic_element] + list(kwargs.values())])
        self.length = max(len(channels) - 1, self.length)
        self.kernel_size = self._init_kernel_size(kernel_size)
        self.atomic_element = self._init_atomic_element(atomic_element)
        self.channels = self._init_attr("channels", channels)

        self._check_args()

        self.initializer_method = initializer_method
        self.initializer_args = initializer_args if initializer_args is not None else self._default_init_args()
        self.initalizer: BimonnInitializer = self.create_initializer(**self.initializer_args)

        # kwargs['channels'] = channels

        for attr, value in kwargs.items():
            setattr(self, attr, self._init_attr(attr, value))

        self.bisel_initializers = self.initalizer.generate_bisel_initializers(self)

        # self.layers = []
        self.layers = BinarySequential()
        self.bises_idx = []
        self.bisecs_idx = []
        self.bisels_idx = []
        for idx in range(len(self)):
            layer = self._make_layer(idx)
            self.layers.append(layer)
            # setattr(self, f'layer{idx+1}', layer)

    def _check_args(self):
        assert isinstance(self.kernel_size, list), "kernel_size must be a list of int"
        assert isinstance(self.kernel_size[0], (tuple, int)), f"kernel_size[0] is {type(self.kernel_size[0])} but must be int, or tuple of ints"

        assert isinstance(self.atomic_element, list), "atomic_element must be a list of str"
        assert isinstance(self.atomic_element[0], str), f"atomic_element[0] is {type(self.atomic_element[0])} but must be str"

        assert isinstance(self.channels, list), "channels must be a list of int"
        assert isinstance(self.channels[0], int), f"channels[0] is {type(self.channels[0])} but must be int"

    def _default_init_args(self):
        if self.atomic_element[0] == "bisel":
            return {
                "input_mean": 0.5,
                "bise_init_method": InitBiseEnum.CUSTOM_HEURISTIC,
                "bise_init_args": {"init_bias_value": 1},
            }

        elif self.atomic_element[0] == "sybisel":
            return {
                "input_mean": 0,
                "bise_init_method": InitBiseEnum.CUSTOM_CONSTANT,
                "bise_init_args": {"mean_weight": "auto"},
            }

    def create_initializer(self, **kwargs):
        if self.initializer_method.value == InitBimonnEnum.IDENTICAL.value:
            return BimonnInitializer(**kwargs)

        elif self.initializer_method.value == InitBimonnEnum.INPUT_MEAN.value:
            # print(kwargs)
            return BimonnInitInputMean(**kwargs)

        raise NotImplementedError("Initializer not recognized.")

    @property
    def bises(self):
        return [self.layers[idx] for idx in self.bises_idx]

    @property
    def bisecs(self):
        return [self.layers[idx] for idx in self.bisecs_idx]

    @property
    def bisels(self):
        return [self.layers[idx] for idx in self.bisels_idx]

    def forward(self, x):
        output = self.layers[0](x)
        for layer in self.layers[1:]:
            output = layer(output)
        return output

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

    # deprecated
    def get_bise_selems(self) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
        """Go through all BiSE indexes and shows the learned selem and operation. If None are learned, puts value
        None.

        Returns:
            (dict, dict): the dictionary of learned selems and operations with indexes as keys.
        """
        selems = {}
        operations = {}
        v1, v2 = 0, 1
        for idx in self.bises_idx:
            if v1 is not None:
                selems[idx], operations[idx] = self.layers[idx].find_selem_and_operation(v1, v2)
                v1, v2 = self.layers[idx].get_outputs_bounds(v1, v2)
            else:
                selems[idx], operations[idx] = None, None
        return selems, operations

    def __len__(self):
        return self.length
        # return len(self.atomic_element)
        # return len(self.layers)

    def _init_kernel_size(self, kernel_size: List[Union[Tuple, int]]):
        if isinstance(kernel_size, list):
            res = []
            for size in kernel_size:
                if isinstance(size, int):
                    res.append((size, size))
                else:
                    res.append(size)
            return res

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        return [kernel_size for _ in range(len(self))]


    def _init_channels(self, channels: List[int]):
        # self.out_channels = channels[1:]
        # self.in_channels = channels[:-1]
        self.channels = channels
        return self.channels

    @property
    def out_channels(self):
        return self.channels[1:]

    @property
    def in_channels(self):
        return self.channels[:-1]

    def _init_input_mean(self, input_mean: Union[float, List[float]]):
        if isinstance(input_mean, list):
            return input_mean
        return [input_mean] + [0 if self.atomic_element[idx - 1] == "sybisel" else .5 for idx in range(1, len(self))]

    def _init_atomic_element(self, atomic_element: Union[str, List[str]]):
        if isinstance(atomic_element, list):
            return [s.lower() for s in atomic_element]

        return [atomic_element.lower() for _ in range(len(self))]

    def _init_attr(self, attr_name, attr_value):
        if attr_name == "kernel_size":
            # return self._init_kernel_size(attr_value)
            return self.kernel_size

        if attr_name == "atomic_element":
            return self._init_atomic_element(attr_value)

        if attr_name == "channels":
            return self._init_channels(attr_value)

        if attr_name == "input_mean":
            return self._init_input_mean(attr_value)

        if isinstance(attr_value, list):
            return attr_value

        return [attr_value for _ in range(len(self))]
        # return [attr_value for _ in range(len(self.kernel_size))]

    def is_not_default(self, key: str) -> bool:
        return key in self.__dict__.keys()

    def bisels_kwargs_idx(self, idx):
        return dict(
            **{'shared_weights': None, "in_channels": self.in_channels[idx], "out_channels": self.out_channels[idx]},
            **{k: getattr(self, k)[idx] for k in self.bisels_args if self.is_not_default(k)}
        )

    @property
    def bises_args(self):
        return BiSEL._bises_args()

    @property
    def bisels_args(self):
        return set(self.bises_args).union(
            ['constant_P_lui', "lui_kwargs",]
        )

    def _make_layer(self, idx):
        if self.atomic_element[idx] == 'bisel':
            layer = BiSEL(initializer=self.bisel_initializers[idx], **self.bisels_kwargs_idx(idx))
            self.bisels_idx.append(idx)

        elif self.atomic_element[idx] == 'sybisel':
            layer = SyBiSEL(initializer=self.bisel_initializers[idx], **self.bisels_kwargs_idx(idx))
            self.bisels_idx.append(idx)

        return layer

    @classmethod
    def default_args(cls) -> Dict[str, dict]:
        """Return the default arguments of the model, in the format of argparse.ArgumentParser"""
        res = super().default_args()
        res.update({
            k: v for k, v in BiSELBase.default_args().items()
            if k not in res
            and k not in [
                "initializer", "in_channels", "out_channels", "bise_module", "lui_module",
            ]
        })
        return res


class BiMoNNClassifier(BiMoNN, ABC):

    def __init__(
        self,
        n_classes: int,
        input_size: Tuple[int],
        channels: List[int],
        *args,
        **kwargs
    ):
        assert len(input_size) == 3, "input_size shape must be (n_chan, width, length)"
        assert isinstance(n_classes, int), f"n_classes is {n_classes} but must be an integer"

        channels = [input_size[0]] + channels
        super().__init__(*args, channels=channels, **kwargs)

        self.n_classes = n_classes
        self.input_size = input_size



    def forward(self, x):
        output = super().forward(x)
        batch_size = output.shape[0]
        output = output.squeeze()
        if batch_size == 1:
            output = output.unsqueeze(0)
        return output

    def forward_save(self, x):
        output = super().forward_save(x)
        batch_size = output["output"].shape[0]
        output["output"] = output["output"].squeeze()
        if batch_size == 1:
            output["output"] = output["output"].unsqueeze(0)
        return output


class BiMoNNClassifierLastLinearBase(BiMoNNClassifier):
    classif_layer_fn: Callable

    def __init__(
        self,
        n_classes: int,
        input_size: Tuple[int],
        final_bisel_kwargs: Dict = None,
        apply_last_activation: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, n_classes=n_classes, input_size=input_size, **kwargs)

        self.repr_size = input_size[1:]
        self.apply_last_activation = apply_last_activation

        last_layer_idx = len(self.layers) - 1


        self.bisel_kwargs = copy.deepcopy(self.bisels_kwargs_idx(last_layer_idx)) if final_bisel_kwargs is None else final_bisel_kwargs
        self.bisel_kwargs["in_channels"] = self.out_channels[-1]
        self.bisel_kwargs["out_channels"] = n_classes
        self.bisel_kwargs["kernel_size"] = self.repr_size
        self.bisel_kwargs["padding"] = 0

        last_initializer = self.initalizer.generate_bisel_initializer_layer(self, layer_idx=last_layer_idx)

        if not self.apply_last_activation:
            self.bisel_kwargs["threshold_mode"]["activation"] = "identity"
        classification_layer = self.classif_layer_fn(initializer=last_initializer, **self.bisel_kwargs)

        # self.in_channels.append(self.out_channels[-1])
        # self.out_channels.append(n_classes)
        self.kernel_size.append(self.repr_size)

        self.layers.append(classification_layer)
        self.bisels_idx.append(len(self.layers) - 1)

    @property
    def classification_layer(self):
        return self.layers[-1]


class BiMoNNClassifierLastLinear(BiMoNNClassifierLastLinearBase):
    def __init__(
        self,
        # kernel_size: List[Union[Tuple, int]],
        # n_classes: int,
        # input_size: Tuple[int],
        atomic_element: Union[str, List[str]],
        # final_bisel_kwargs: Dict = None,
        *args,
        **kwargs
    ):
        self.classif_layer_fn = SyBiSEL if atomic_element in ["sybisel", ["sybisel"]] else BiSEL
        super().__init__(
            # kernel_size=kernel_size,
            # n_classes=n_classes,
            # input_size=input_size,
            # final_bisel_kwargs=final_bisel_kwargs,
            atomic_element=atomic_element,
            *args,
            **kwargs
        )


class BiMoNNClassifierLastLinearNotBinary(BiMoNNClassifierLastLinearBase):
    def __init__(
        self,
        # kernel_size: List[Union[Tuple, int]],
        # n_classes: int,
        atomic_element: Union[str, List[str]],
        # input_size: Tuple[int],
        # final_bisel_kwargs: Dict = None,
        *args,
        **kwargs
    ):
        self.classif_layer_fn=SyBiSELNotBinary if atomic_element in ["sybisel", ["sybisel"]] else BiSELNotBinary
        super().__init__(
            # kernel_size=kernel_size,
            # n_classes=n_classes,
            # input_size=input_size,
            # final_bisel_kwargs=final_bisel_kwargs,
            atomic_element=atomic_element,
            *args,
            **kwargs
        )



class BiMoNNClassifierMaxPoolBase(BiMoNNClassifier):
    classif_layer_fn: Callable
    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        n_classes: int,
        input_size: Tuple[int],
        final_bisel_kwargs: Dict = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, kernel_size=kernel_size, n_classes=n_classes, input_size=input_size, **kwargs)

        # self.channels = [input_size[0]] + self.channels + [n_classes]

        # assert len(input_size) == 3, "input_size shape must be (n_chan, width, length)"

        self.repr_size = input_size[1:]
        self.maxpool_layers = []
        for idx in range(len(self)):
            self.repr_size = (self.repr_size[0] // 2, self.repr_size[1] // 2)
            self.maxpool_layers.append(nn.MaxPool2d((2, 2)))
            # setattr(self, f"maxpool_{idx}", self.maxpool_layers[-1])


        # Compatibility python 3.7
        layers2 = []
        for bisel, maxpool in zip(self.layers, self.maxpool_layers):
            layers2 += BinarySequential(bisel, maxpool)
        self.layers = BinarySequential(*layers2)
        # self.layers = sum([[bisel, maxpool] for bisel, maxpool in zip(self.layers, self.maxpool_layers)], start=[])

        self.bisels_idx = [2*bisel_idx for bisel_idx in self.bisels_idx]

        self.bisel_kwargs = self.bisels_kwargs_idx(0) if final_bisel_kwargs is None else final_bisel_kwargs
        self.bisel_kwargs["in_channels"] = self.channels[-1]
        self.bisel_kwargs["out_channels"] = n_classes
        self.bisel_kwargs["kernel_size"] = self.repr_size
        self.bisel_kwargs["padding"] = 0

        self.classification_layer = self.classif_layer_fn(**self.bisel_kwargs)

        # self.in_channels.append(self.out_channels[-1])
        # self.out_channels.append(n_classes)
        self.kernel_size.append(self.repr_size)

        self.layers.append(self.classification_layer)
        self.bisels_idx.append(len(self.layers) - 1)


class BiMoNNClassifierMaxPool(BiMoNNClassifierMaxPoolBase):
    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        n_classes: int,
        input_size: Tuple[int],
        atomic_element: Union[str, List[str]],
        final_bisel_kwargs: Dict = None,
        *args,
        **kwargs
    ):
        self.classif_layer_fn = SyBiSEL if atomic_element in ["sybisel", ["sybisel"]] else BiSEL
        super().__init__(
            kernel_size=kernel_size,
            n_classes=n_classes,
            input_size=input_size,
            final_bisel_kwargs=final_bisel_kwargs,
            atomic_element=atomic_element,
            *args, **kwargs
        )


class BiMoNNClassifierMaxPoolNotBinary(BiMoNNClassifierMaxPoolBase):
    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        n_classes: int,
        input_size: Tuple[int],
        atomic_element: Union[str, List[str]],
        final_bisel_kwargs: Dict = None,
        *args,
        **kwargs
    ):
        self.classif_layer_fn = SyBiSELNotBinary if atomic_element in ["sybisel", ["sybisel"]] else BiSELNotBinary
        super().__init__(
            kernel_size=kernel_size,
            n_classes=n_classes,
            input_size=input_size,
            final_bisel_kwargs=final_bisel_kwargs,
            atomic_element=atomic_element,
            *args, **kwargs
        )
