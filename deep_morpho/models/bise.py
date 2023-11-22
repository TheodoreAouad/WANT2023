from typing import Tuple, Union, Dict
from matplotlib.cbook import normalize_kwargs

import torch
import numpy as np

from ..initializer import BiseInitializer, InitBiseHeuristicWeights, InitSybiseConstantVarianceWeights, InitBiseEnum

from .bise_base import BiSEBase, BiseWeightsOptimEnum, ClosestSelemEnum, BiseBiasOptimEnum, SyBiSEBase, BiseWeightsOptimEnum


class BiSE(BiSEBase):
    """Given the BiSEL implementation, the BiSE always has an input chan of 1.
    """

    def __init__(
        self,
        kernel_size: Tuple,
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh"},
        activation_P: float = 1,
        constant_activation_P: bool = False,
        weights_optim_mode: BiseWeightsOptimEnum = BiseWeightsOptimEnum.THRESHOLDED,
        weights_optim_args: Dict = {},
        initializer: BiseInitializer = InitBiseHeuristicWeights(input_mean=0.5, init_bias_value=1),
        out_channels: int = 1,
        in_channels: int = 1,
        closest_selem_method: ClosestSelemEnum = ClosestSelemEnum.MIN_DIST_DIST_TO_CST,
        closest_selem_args: Dict = {},
        bias_optim_mode: BiseBiasOptimEnum = BiseBiasOptimEnum.POSITIVE,
        *args,
        **kwargs
    ):
        super().__init__(
            kernel_size=kernel_size,
            threshold_mode=threshold_mode,
            activation_P=activation_P,
            constant_activation_P=constant_activation_P,
            initializer=initializer,
            weights_optim_mode=weights_optim_mode,
            weights_optim_args=weights_optim_args,
            out_channels=out_channels,
            in_channels=in_channels,
            closest_selem_method=closest_selem_method,
            closest_selem_args=closest_selem_args,
            bias_optim_mode=bias_optim_mode,
            *args,
            **kwargs
        )

    @property
    def closest_selem(self):
        if self._closest_selem is None:
            return None
        return self._closest_selem

    @property
    def learned_selem(self):
        if self._learned_selem is None:
            return None
        return self._learned_selem



class SyBiSE(SyBiSEBase):
    def __init__(
        self,
        kernel_size: Tuple,
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh_symetric"},
        activation_P: float = 1,
        constant_activation_P: bool = False,
        # constant_weight_P: bool = True,
        # weights_optim_mode: BiseWeightsOptimEnum = BiseWeightsOptimEnum.THRESHOLDED,
        # weights_optim_args: Dict = {},
        shared_weights: torch.tensor = None,
        # init_bias_value: float = 0,
        # input_mean: float = 0,
        # mean_weight_value: float = "auto",
        # init_weight_mode: InitBiseEnum = InitBiseEnum.CUSTOM_HEURISTIC,
        # initializer_method: InitBiseEnum = InitBiseEnum.CUSTOM_CONSTANT,
        # initializer_args: Dict = {},
        initializer: BiseInitializer = InitSybiseConstantVarianceWeights(input_mean=0, mean_weight="auto"),
        out_channels: int = 1,
        in_channels: int = 1,
        do_mask_output: bool = False,
        # closest_selem_method: ClosestSelemEnum = ClosestSelemEnum.MIN_DIST,
        # closest_selem_args: Dict = {distance_agg_min},
        bias_optim_mode: BiseBiasOptimEnum = BiseBiasOptimEnum.RAW,
        padding=None,
        padding_mode: str = "replicate",
        *args,
        **kwargs
    ):
        super().__init__(
            kernel_size=kernel_size,
            threshold_mode=threshold_mode,
            activation_P=activation_P,
            constant_activation_P=constant_activation_P,
            # weights_optim_mode=weights_optim_mode,
            # weights_optim_args=weights_optim_args,
            shared_weights=shared_weights,
            initializer=initializer,
            # # init_bias_value=init_bias_value,
            # # input_mean=input_mean,
            # # init_weight_mode=init_weight_mode,
            # # mean_weight_value=mean_weight_value,
            out_channels=out_channels,
            in_channels=in_channels,
            do_mask_output=do_mask_output,
            # closest_selem_method=closest_selem_method,
            # closest_selem_args=closest_selem_args,
            bias_optim_mode=bias_optim_mode,
            padding=padding,
            padding_mode=padding_mode,
            *args,
            **kwargs
        )

    @property
    def closest_selem(self):
        if self._closest_selem is None:
            return None
        return self._closest_selem[:, 0, ...]

    @property
    def learned_selem(self):
        if self._learned_selem is None:
            return None
        return self._learned_selem[:, 0, ...]
