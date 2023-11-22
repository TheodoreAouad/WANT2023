from typing import Union, Tuple, Dict
from enum import Enum

import torch
import numpy as np

from .bise_base import BiseBiasOptimEnum
# from .bise_old import BiSE, InitBiseEnum, SyBiSE, BiseBiasOptimEnum
# from .bise_old2 import BiSE as BiSE_OLD2

from .bise import BiSE, SyBiSE
from .lui import LUI, SyLUI
from .binary_nn import BinaryNN
from ..initializer import BiselInitIdentical, BiselInitializer, InitBiseHeuristicWeights, InitBiseEnum


def invert_euclidian(x, div1, div2):
    """x = mq + r
    Returns rq + m
    """
    m = x // div1
    r = x % div1
    return div2 * r + m


def regroup_input_lui(x, n_chin, n_chout):
    """With the BiSe implementation of the group convolution, the input is recalibrated to be in the right order.
    Example:
       Given in channels (1, 2, 3), out_channels (1, 2), the group output channels are (1, 1, 2, 2, 3, 3). After
       regrouping, the output channels are (1, 2, 3, 1, 2, 3), ready for the next group convolution.
    """
    index = invert_euclidian(torch.arange(n_chout*n_chin), n_chin, n_chout).to(x.device)
    return torch.index_select(x, 1, index)



class InitBiselEnum(Enum):
    DIFFERENT_INIT = 0


class BiSELBase(BinaryNN):

    def __init__(
        self,
        bise_module: BiSE,
        lui_module: LUI,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh"},
        constant_P_lui: bool = False,
        initializer: BiselInitializer = BiselInitIdentical(InitBiseHeuristicWeights(init_bias_value=1, input_mean=0.5)),
        lui_kwargs: Dict = {},
        **bise_kwargs
    ):
        super().__init__()
        self.bise_module = bise_module
        self.lui_module = lui_module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.threshold_mode = self._init_threshold_mode(threshold_mode)
        self.constant_P_lui = constant_P_lui
        self.bise_kwargs = bise_kwargs

        self.initializer = initializer

        self.lui_kwargs = lui_kwargs
        self.lui_kwargs['constant_activation_P'] = self.constant_P_lui
        self.lui_kwargs = {k: v for k, v in bise_kwargs.items() if k not in self.lui_kwargs and k in self.lui_args}

        self.bises = self._init_bises()
        self.luis = self._init_luis()

        self.initializer.post_initialize(self)

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["weight", "activation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode

    def _init_bises(self):
        bise_initializer = self.initializer.get_bise_initializers(self)
        bises = self.bise_module(
            kernel_size=self.kernel_size, threshold_mode=self.threshold_mode, initializer=bise_initializer,
            in_channels=self.in_channels, out_channels=self.out_channels * self.in_channels, groups=self.in_channels,
            **self.bise_kwargs
        )
        return bises

    def _init_luis(self):
        luis = []
        lui_initializer = self.initializer.get_lui_initializers(self)
        luis = self.lui_module(
            in_channels=self.in_channels * self.out_channels,
            out_channels=self.out_channels,
            threshold_mode=self.threshold_mode,
            initializer=lui_initializer,
            groups=self.out_channels,
            **self.lui_kwargs,
        )
        return luis


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bise_res = self.bises(x)
        recalibrated = regroup_input_lui(bise_res, self.in_channels, self.out_channels)
        lui_res = self.luis(recalibrated)

        return lui_res

    def forward_save(self, x):
        output = {}

        bise_res = self.bises(x)
        recalibrated = regroup_input_lui(bise_res, self.in_channels, self.out_channels)
        lui_res = self.luis(recalibrated)

        for chout in range(self.out_channels):
            output[chout] = lui_res[:, chout, ...]
            for chin in range(self.in_channels):
                output[chin, chout] = bise_res[:, self.convert_chin_chout_bise_chan(chin, chout), ...]

        output['output'] = lui_res

        return output

    def convert_chin_chout_bise_chan(self, chin: int, chout: int):
        return chin * self.out_channels + chout

    def get_activation_P_bise(self, chin: int, chout: int):
        return self.bises.activation_P[self.convert_chin_chout_bise_chan(chin, chout)]

    def get_activation_P_lui(self, chout: int):
        return self.luis.activation_P[chout]

    def get_bias_bise(self, chin: int, chout: int):
        return self.bises.bias[self.convert_chin_chout_bise_chan(chin, chout)]

    def get_bias_lui(self, chout: int):
        return self.luis.bias[chout]

    def get_weight_bise(self, chin: int, chout: int):
        return self.bises.weight[self.convert_chin_chout_bise_chan(chin, chout), 0, ...]

    def get_closest_operation_bise(self, chin: int, chout: int):
        return self.bises.closest_operation[self.convert_chin_chout_bise_chan(chin, chout)]
    
    def get_closest_selem_dist_bise(self, chin: int, chout: int):
        return self.bises.closest_selem_dist[self.convert_chin_chout_bise_chan(chin, chout)]

    def get_learned_operation_bise(self, chin: int, chout: int):
        return self.bises.learned_operation[self.convert_chin_chout_bise_chan(chin, chout)]

    def get_weight_param_bise(self, chin: int, chout: int):
        return self.bises.weight_param[self.convert_chin_chout_bise_chan(chin, chout), 0, ...]

    def get_learned_selem_bise(self, chin: int, chout: int):
        return self.bises.learned_selem[self.convert_chin_chout_bise_chan(chin, chout), 0, ...]

    def get_learned_selem_lui(self, chout: int):
        return self.luis.learned_selem[chout, 0, ...]

    def get_closest_selem_bise(self, chin: int, chout: int):
        return self.bises.closest_selem[self.convert_chin_chout_bise_chan(chin, chout), 0, ...]

    def get_closest_selem_lui(self, chout: int):
        return self.luis.closest_selem[chout, 0, ...]

    def get_weight_grad_bise(self, chin: int, chout: int):
        grad_weights = self.bises.weights_handler.grad
        if grad_weights is None:
            return None
        return grad_weights[self.convert_chin_chout_bise_chan(chin, chout), 0, ...]

    def is_activated_bise(self, chin: int, chout: int):
        return self.bises.is_activated[self.convert_chin_chout_bise_chan(chin, chout)]

    def is_activated_lui(self, chout: int):
        return self.luis.is_activated[chout]

    def get_bias_grad_bise(self, chin: int, chout: int):
        grad_biases = self.bises.bias.grad
        if grad_biases is None:
            return None
        return grad_biases[self.convert_chin_chout_bise_chan(chin, chout)]

    def get_bias_grad_lui(self, chout: int):
        grad_biases = self.luis.bias.grad
        if grad_biases is None:
            return None
        return grad_biases[chout]

    def get_weight_grad_lui(self, chin: int, chout: int):
        grad_weights = self.luis.bias.grad
        if grad_weights is None:
            return None
        return grad_weights[chin, chout]

    @property
    def weight(self) -> torch.Tensor:
        """ Returns the convolution weights, of shape (out_channels, in_channels, W, L).
        """
        # return torch.cat([layer.weight for layer in self.bises], axis=1)
        with torch.no_grad():
            return torch.stack([
                torch.stack(
                    [self.get_weight_bise(chin, chout) for chin in range(self.in_channels)]
                ) for chout in range(self.out_channels)
            ])


    @property
    def activation_P_bise(self) -> torch.Tensor:
        """ Returns the activation P parameter, of shape (out_channels, in_channels).
        """
        # return torch.cat([layer.weight for layer in self.bises], axis=1)
        return torch.stack([
            torch.stack(
                [self.get_activation_P_bise(chin, chout) for chin in range(self.in_channels)]
            ) for chout in range(self.out_channels)
        ])

    @property
    def bias_bise(self) -> torch.Tensor:
        """ Returns the convolution biases, of shape (out_channels, in_channels).
        """
        return torch.stack([
            torch.stack(
                [self.get_bias_bise(chin, chout) for chin in range(self.in_channels)]
            ) for chout in range(self.out_channels)
        ])

    @property
    def weights(self) -> torch.Tensor:
        return self.weight

    def get_weight_lui(self, chout: int):
        return self.luis.weight[chout, ...]

    def get_coef_lui(self, chout: int):
        return self.luis.coefs[chout, ...]

    @property
    def coef(self):
        return self.luis.coefs

    @property
    def bias_lui(self):
        return self.luis.bias

    # @property
    # def weight_P_bise(self) -> torch.Tensor:
    #     """ Returns the weights P of the bise layers, of shape (out_channels, in_channels).
    #     """
    #     return torch.stack([layer.weight_P for layer in self.bises], axis=-1)


    @property
    def activation_P_lui(self) -> torch.Tensor:
        """ Returns the activations P of the lui layer, of shape (out_channels).
        """
        return self.luis.activation_P


    # @property
    # def bias_bise(self) -> torch.Tensor:
    #     """ Returns the bias of the bise layers, of shape (out_channels, in_channels).
    #     """
    #     return torch.stack([layer.bias for layer in self.bises], axis=-1)

    @property
    def bias_bises(self) -> torch.Tensor:
        return self.bias_bise

    @property
    def bias_luis(self) -> torch.Tensor:
        return self.bias_lui

    @property
    def coefs(self) -> torch.Tensor:
        """ Returns the coefficients of the linear operation of LUI, of shape (out_channels, in_channels).
        """
        return self.coef

    @property
    def bises_args(self):
        return self._bises_args()

    @staticmethod
    def _bises_args():
        return [
            'kernel_size', 'weight_P', 'threshold_mode', 'activation_P', 'in_channels',
            'out_channels', "constant_activation_P", "constant_weight_P", "closest_selem_method",
            "bias_optim_mode", "bias_optim_args", "weights_optim_mode", "weights_optim_args",
        ]

    @property
    def lui_args(self):
        return set(self.bises_args).difference(["padding"]).union(["in_channels"])

    @classmethod
    def default_args(cls) -> Dict[str, dict]:
        res = super().default_args()
        res.update({k: v for k, v in BiSE.default_args().items() if k not in res})

        return res

    @property
    def n_activated(self):
        return self.bises.n_activated + self.luis.n_activated

    @property
    def percentage_activated(self):
        return (self.bises.n_activated + self.luis.n_activated) / (self.bises.n_elements + self.luis.n_elements)


class BiSEL(BiSELBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh"},
        constant_P_lui: bool = False,
        initializer: BiselInitializer = BiselInitIdentical(InitBiseHeuristicWeights(init_bias_value=1, input_mean=0.5)),
        lui_kwargs: Dict = {},
        **bise_kwargs
    ):
        super().__init__(
            bise_module=BiSE,
            lui_module=LUI,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            threshold_mode=threshold_mode,
            constant_P_lui=constant_P_lui,
            initializer=initializer,
            # init_bias_value_bise=init_bias_value_bise,
            # init_bias_value_lui=init_bias_value_lui,
            # input_mean=input_mean,
            # init_weight_mode=init_weight_mode,
            lui_kwargs=lui_kwargs,
            **bise_kwargs
        )


class SyBiSEL(BiSELBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh_symetric"},
        constant_P_lui: bool = False,
        # init_bias_value_bise: float = 0,
        # init_bias_value_lui: float = 0,
        # input_mean: float = 0,
        # init_weight_mode: InitBiseEnum = InitBiseEnum.CUSTOM_CONSTANT,
        initializer: BiselInitializer = BiselInitIdentical(InitBiseHeuristicWeights(init_bias_value=1, input_mean=0.5)),
        # initializer_method: InitBiseEnum = InitBiseEnum.CUSTOM_CONSTANT,
        # initializer_args: Dict = {"input_mean": 0, "mean_weight": "auto"},
        bias_optim_mode: BiseBiasOptimEnum = BiseBiasOptimEnum.RAW,
        lui_kwargs: Dict = {},
        **bise_kwargs
    ):
        super().__init__(
            bise_module=SyBiSE,
            lui_module=SyLUI,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias_optim_mode=bias_optim_mode,
            threshold_mode=threshold_mode,
            constant_P_lui=constant_P_lui,
            initializer=initializer,
            lui_kwargs=lui_kwargs,
            **bise_kwargs
        )
