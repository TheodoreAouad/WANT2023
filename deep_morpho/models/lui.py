from typing import Union, Dict

import torch
import numpy as np

from ..initializer import BiseInitializer, InitBiseHeuristicWeights, InitSybiseConstantVarianceWeights, InitBiseEnum

from .bise_base import BiSEBase, ClosestSelemEnum, BiseBiasOptimEnum, SyBiSEBase


class BiSELUIExtender:

    def forward(self, x):
        if self.force_identity:
            return x
        return super().forward(x)

    @property
    def activation_P(self):
        if self.force_identity:
            return torch.ones_like(self.activation_threshold_layer.P_, requires_grad=False)
        return super().activation_P

    @property
    def bias(self):
        if self.force_identity:
            return torch.zeros_like(super().bias, requires_grad=False)
        return super().bias

    def init_weights(self):
        if self.force_identity:
            return
        return super().init_weights()

    def init_bias(self):
        if self.force_identity:
            return
        return super().init_bias()

    @property
    def coefs(self):
        return self.weight[..., 0, 0]

    def update_binary_sets(self, *args, **kwargs):
        return self.update_binary_selems(*args, **kwargs)

    def find_selem_and_operation_chan(self, *args, **kwargs):
        if self.force_identity:
            return np.array([True]), 'dilation'
        return super().find_selem_and_operation_chan(*args, **kwargs)

    def find_set_and_operation_chan(self, *args, **kwargs):
        return self.find_selem_and_operation_chan(*args, **kwargs)

    def find_closest_selem_and_operation_chan(self, *args, **kwargs):
        if self.force_identity:
            return np.array([True]), 'dilation'
        return super().find_closest_selem_and_operation_chan(*args, **kwargs)

    def find_closest_set_and_operation_chan(self, *args, **kwargs):
        return self.find_closest_selem_and_operation_chan(*args, **kwargs)

    @property
    def closest_set(self):
        if self.force_identity:
            return np.ones_like(self.weight.detach().cpu(), dtype=bool)
            # return np.array([True])
        # if self._closest_selem is None:
        #     return None
        return self._closest_selem

    @property
    def learned_selem(self):
        if self.force_identity:
            # return np.array([True])
            return np.ones_like(self.weight.detach().cpu(), dtype=bool)
        # if self._learned_selem is None:
        #     return None
        return self._learned_selem

    @property
    def learned_operation(self):
        if self.force_identity:
            return np.ones(self.groups)
        # if self._learned_operation is None:
        #     return None
        return self._learned_operation


    @property
    def learned_set(self):
        return self.learned_selem[..., 0, 0]

    @property
    def is_activated(self):
        if self.force_identity:
            return np.ones(self.weight.shape[0], dtype=bool)
        return self._is_activated


class LUI(BiSELUIExtender, BiSEBase):

    def __init__(
        self,
        initializer: BiseInitializer = InitBiseHeuristicWeights(input_mean=.5, init_bias_value=1),
        out_channels: int = 1,
        in_channels: int = 1,
        groups: int = 1,
        bias_optim_mode: BiseBiasOptimEnum = BiseBiasOptimEnum.POSITIVE_INTERVAL_REPARAMETRIZED,
        force_identity: bool = False,
        *args,
        **kwargs
    ):
        self.force_identity = force_identity or (in_channels == out_channels == groups)
        if self.force_identity:
            bias_optim_mode = BiseBiasOptimEnum.RAW

        for key in ["do_mask_output", "padding", "kernel_size"]:
            if key in kwargs:
                del kwargs[key]

        super().__init__(
            kernel_size=(1, 1),
            initializer=initializer,
            out_channels=out_channels,
            in_channels=in_channels,
            groups=groups,
            do_mask_output=False,
            bias_optim_mode=bias_optim_mode,
            padding=0,
            *args,
            **kwargs
        )

    @classmethod
    def default_args(cls) -> Dict[str, dict]:
        """Return the default arguments of the model, in the format of argparse.ArgumentParser"""
        res = super().default_args()
        res = {
            k: v for k, v in res.items() if k not in [
                "do_mask_output", "padding"
            ]
        }
        return res

class SyLUI(BiSELUIExtender, SyBiSEBase):

    def __init__(
        self,
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh_symetric"},
        activation_P: float = 1,
        constant_activation_P: bool = False,
        # constant_weight_P: bool = False,
        shared_weights: torch.tensor = None,
        # init_bias_value: float = 0,
        # input_mean: float = 0,
        # init_weight_mode: InitBiseEnum = InitBiseEnum.CUSTOM_CONSTANT,
        initializer: BiseInitializer = InitSybiseConstantVarianceWeights(input_mean=0, mean_weight="auto"),
        in_channels: int = 1,
        out_channels: int = 1,
        groups: int = 1,
        # closest_selem_method: ClosestSelemEnum = ClosestSelemEnum.MIN_DIST_DIST_TO_BOUNDS,
        # closest_selem_args: Dict = {},
        bias_optim_mode: BiseBiasOptimEnum = BiseBiasOptimEnum.RAW,
        force_identity: bool = False,
        mean_weight_value: float = "auto",
        *args,
        **kwargs
    ):
        self.force_identity = force_identity or (in_channels == out_channels == groups)
        if self.force_identity:
            bias_optim_mode = BiseBiasOptimEnum.RAW

        super().__init__(
            kernel_size=(1, 1),
            threshold_mode=threshold_mode,
            activation_P=activation_P,
            constant_activation_P=constant_activation_P,
            # constant_weight_P=constant_weight_P,
            shared_weights=shared_weights,
            # init_bias_value=init_bias_value,
            # input_mean=input_mean,
            # init_weight_mode=init_weight_mode,
            initializer=initializer,
            out_channels=out_channels,
            in_channels=in_channels,
            groups=groups,
            do_mask_output=False,
            # # closest_selem_method=closest_selem_method,
            # # closest_selem_args=closest_selem_args,
            bias_optim_mode=bias_optim_mode,
            mean_weight_value=mean_weight_value,
            padding=0,
            *args,
            **kwargs
        )
