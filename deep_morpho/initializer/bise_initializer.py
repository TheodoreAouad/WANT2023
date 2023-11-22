from enum import Enum
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.init as init

from deep_morpho.threshold_fn import (
    tanh_threshold_inverse, sigmoid_threshold_inverse, arctan_threshold_inverse, relu_threshold_inverse, identity_threshold_inverse
)
from general.utils import uniform_sampling_bound


class InitBiseEnum(Enum):
    NORMAL = 1
    KAIMING_UNIFORM = 2
    CUSTOM_HEURISTIC = 3
    CUSTOM_CONSTANT = 4
    IDENTITY = 5
    CUSTOM_CONSTANT_RANDOM_BIAS = 6
    CUSTOM_HEURISTIC_RANDOM_BIAS = 7
    ELLIPSE = 8
    ELLIPSE_ROOT = 9
    CUSTOM_CONSTANT_DUAL = 10
    CUSTOM_CONSTANT_DUAL_RANDOM_BIAS = 11
    CUSTOM_CONSTANT_CONSTANT_WEIGHTS = 12
    CUSTOM_CONSTANT_CONSTANT_WEIGHTS_DUAL = 13
    CUSTOM_CONSTANT_CONSTANT_WEIGHTS_RANDOM_BIAS = 14
    CUSTOM_CONSTANT_CONSTANT_WEIGHTS_DUAL_RANDOM_BIAS = 15


class BiseInitializer:

    def initialize(self, module: nn.Module):
        return module.weight, module.bias


class InitWeightsThenBias(BiseInitializer):
    def initialize(self, module: nn.Module):
        self.init_weights(module)
        self.init_bias(module)

        assert not (module.weight.isnan().any()), "NaN in weights"
        assert not (module.bias.isnan().any()), "NaN in bias"
        return module.weight, module.bias

    def init_weights(self, module):
        pass

    def init_bias(self, module):
        pass


class InitBiasFixed(InitWeightsThenBias):
    def __init__(self, init_bias_value: float = None, *args, **kwargs) -> None:
        self.init_bias_value = init_bias_value

    def init_bias(self, module):
        module.set_bias(
            torch.zeros_like(module.bias) - self.init_bias_value
        )


class InitNormalIdentity(InitBiasFixed):

    def __init__(self, init_bias_value: float, mean: float = 1, std: float = .3, *args, **kwargs) -> None:
        super().__init__(init_bias_value=init_bias_value)
        self.mean = mean
        self.std = std

    @staticmethod
    def _init_normal_identity(kernel_size, chan_output, std=0.3, mean=1) -> torch.Tensor:
        weights = torch.randn((chan_output,) + kernel_size)[:, None, ...] * std - mean
        weights[..., kernel_size[0] // 2, kernel_size[1] // 2] += 2*mean
        return weights


    def init_weights(self, module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        module.set_param_from_weights(self._init_normal_identity(module.kernel_size, module.out_channels))


class InitIdentity(InitBiasFixed):
    def init_weights(self, module: nn.Module):
        module.conv.weight.data.fill_(-1)
        shape = module.conv.weight.shape
        module.conv.weight.data[..., shape[-2]//2, shape[-1]//2] = 1


class InitKaimingUniform(InitBiasFixed):
    def init_weights(self, module: nn.Module):
        module.set_param_from_weights(module.conv.weight)

    def init_bias(self, module: nn.Module):
        fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            self.init_bias_value = torch.rand_like(module.bias) * 2 * bound - bound
        else:
            self.init_bias_value = torch.zeros_like(module.bias)
        module.set_bias(self.init_bias_value)



class InitSybiseBias(InitWeightsThenBias):
    def __init__(self, init_class: BiseInitializer, input_mean: float = 0, *args, **kwargs):
        self.input_mean = input_mean
        self.init_class = init_class(*args, **kwargs)

    def get_bias_from_weights(self, module):
        self.init_bias_value = self.input_mean * module.weight.mean() * torch.tensor(module.weight.shape[1:]).prod()

    def init_weights(self, module):
        return self.init_class(module)

    def init_bias(self, module):
        return module.set_bias(
            torch.zeros_like(module.bias) - self.get_bias_from_weights(module)
        )


class InitIdentitySybise(InitSybiseBias):
    def __init__(self, *args, **kwargs):
        super().__init__(init_class=InitIdentity, *args, **kwargs)


class InitNormalIdentitySybise(InitSybiseBias):
    def __init__(self, *args, **kwargs):
        super().__init__(init_class=InitNormalIdentity, *args, **kwargs)


class InitKaimingUniformSybise(InitSybiseBias):
    def __init__(self, *args, **kwargs):
        super().__init__(init_class=InitKaimingUniform, *args, **kwargs)


class InitBiseHeuristicWeights(InitBiasFixed):
    def __init__(self, init_bias_value: float, input_mean: float = 0.5, *args, **kwargs):
        super().__init__(init_bias_value=init_bias_value)
        self.input_mean = input_mean

    def init_weights(self, module):
        nb_params = torch.tensor(module.weight.shape[1:]).prod()
        mean = self.init_bias_value / (self.input_mean * nb_params)
        std = .5
        lb = mean * (1 - std)
        ub = mean * (1 + std)
        module.set_param_from_weights(
            torch.rand_like(module.weights) * (lb - ub) + ub
        )


class InitBiseConstantVarianceWeights(InitBiasFixed):
    THRESH_STR_TO_FN = {
        "tanh": tanh_threshold_inverse,
        "sigmoid": sigmoid_threshold_inverse,
        "arctan": arctan_threshold_inverse,
        "relu": relu_threshold_inverse,
        "identity": identity_threshold_inverse,
    }

    def __init__(
        self,
        input_mean: float = 0.5,
        max_output_value: float = 0.95,
        p_for_init: float = "auto",
        *args,
        **kwargs
    ):
        r"""
        Args:
            input_mean (float): mean of the input of the bise layer.
            max_output_value (float): if p_activation = 1, the maximum desired output from the init (in other words, \xi(bias))
        """
        super().__init__(init_bias_value=0)
        self.input_mean = input_mean
        self.max_output_value = 0.95
        self.p_for_init = p_for_init

    @staticmethod
    def get_mean(p, nb_params):
        return (np.sqrt(3) + 2) / (4 * p * torch.sqrt(nb_params))

    @staticmethod
    def _get_init_p(nb_params, thresh_inv, max_output_value):
        return (np.sqrt(3) + 2) * torch.sqrt(nb_params) / (4 * 2 * thresh_inv(torch.tensor(max_output_value)))

    def get_init_p(self, nb_params, module):
        thresh_inv = self.THRESH_STR_TO_FN[module.threshold_mode['activation']]
        # return (np.sqrt(3) + 2) * torch.sqrt(nb_params) / (4 * 2 * thresh_inv(torch.tensor(self.max_output_value)))
        return self._get_init_p(nb_params, thresh_inv, self.max_output_value)

    def init_weights(self, module):
        nb_params = torch.tensor(module.weight.shape[1:]).prod()
        p = self.get_init_p(nb_params, module) if self.p_for_init == 'auto' else self.p_for_init
        # p = 1
        mean = self.get_mean(p, nb_params)
        sigma = 1 / (p ** 2 * nb_params) - mean ** 2

        diff = torch.sqrt(3 * sigma)
        lb = mean - diff
        ub = mean + diff

        new_weights = torch.rand_like(module.weights) * (lb - ub) + ub

        module.set_param_from_weights(
            new_weights
        )

        self.init_bias_value = self.input_mean * module.weight.sum((1, 2, 3))


class InitBiseConstantVarianceConstantWeights(InitBiseConstantVarianceWeights):
    """We init the LUI with a mean weights instead of a random uniform function. We take the same mean for simplicity.
    """

    def init_weights(self, module):
        nb_params = torch.tensor(module.weight.shape[1:]).prod()
        p = self.get_init_p(nb_params, module)

        mean = self.get_mean(p, nb_params)
        new_weights = torch.ones_like(module.weights) * mean
        # new_weights = torch.ones_like(module.weights) * mean / nb_params

        module.set_param_from_weights(
            new_weights
        )


        self.init_bias_value = self.input_mean * module.weight.sum((1, 2, 3))


class InitBiseConstantVarianceConstantWeightsRandomBias(InitBiseConstantVarianceConstantWeights):
    def __init__(self, ub: float = 0.0001, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ub = ub

    def init_bias(self, module):
        self.init_bias_value += float(uniform_sampling_bound(-self.ub, self.ub)) * self.init_bias_value
        module.set_bias(
            torch.zeros_like(module.bias) - self.init_bias_value
        )


class InitDualBiseConstantVarianceConstantWeights(InitBiseConstantVarianceWeights):
    """We init the LUI with a mean weights instead of a random uniform function. We take the same mean for simplicity.
    """

    def init_weights(self, module):
        nb_params = torch.tensor(module.weight.shape[1:]).prod()
        p = self.get_init_p(nb_params, module)

        mean = self.get_mean(p, nb_params)
        new_weights = torch.ones_like(module.weights) * mean / nb_params

        module.set_param_from_weights(
            new_weights
        )

        module.weights_handler.factor = mean * nb_params  # set the factor to have the right mean and variance

        self.init_bias_value = self.input_mean * module.weight.sum((1, 2, 3))


class InitDualBiseConstantVarianceConstantWeightsRandomBias(InitDualBiseConstantVarianceConstantWeights):
    def __init__(self, ub: float = 0.0001, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ub = ub

    def init_bias(self, module):
        self.init_bias_value += float(uniform_sampling_bound(-self.ub, self.ub)) * self.init_bias_value
        module.set_bias(
            torch.zeros_like(module.bias) - self.init_bias_value
        )


class InitDualBiseConstantVarianceWeights(InitBiseConstantVarianceWeights):

    def init_weights(self, module):
        nb_params = torch.tensor(module.weight.shape[1:]).prod()
        p = self.get_init_p(nb_params, module)

        mean = self.get_mean(p, nb_params)
        sigma = 1 / (p ** 2 * nb_params) - mean ** 2

        diff = torch.sqrt(3 * sigma)
        lb = mean - diff
        ub = mean + diff

        new_weights = torch.rand_like(module.weights) * (lb - ub) + ub

        module.set_param_from_weights(
            new_weights
        )

        module.weights_handler.factor = mean * nb_params  # set the factor to have the right mean and variance

        self.init_bias_value = self.input_mean * module.weight.sum((1, 2, 3))



class InitDualBiseConstantVarianceWeightsRandomBias(InitDualBiseConstantVarianceWeights):
    def __init__(self, ub: float = 0.0001, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ub = ub

    def init_bias(self, module):
        self.init_bias_value += float(uniform_sampling_bound(-self.ub, self.ub)) * self.init_bias_value
        module.set_bias(
            torch.zeros_like(module.bias) - self.init_bias_value
        )


class InitBiseConstantVarianceWeightsRandomBias(InitBiseConstantVarianceWeights):
    def __init__(self, ub: float = 0.0001, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ub = ub

    def init_bias(self, module):
        self.init_bias_value += float(uniform_sampling_bound(-self.ub, self.ub)) * self.init_bias_value
        module.set_bias(
            torch.zeros_like(module.bias) - self.init_bias_value
        )


class InitSybiseHeuristicWeights(InitWeightsThenBias):
    def __init__(self, input_mean: float = 0, mean_weight: float = "auto", init_bias_value: float = 0, *args, **kwargs) -> None:
        self.input_mean = input_mean
        self.mean_weight = mean_weight
        self.init_bias_value = init_bias_value

        self.mean = None
        self.nb_params = None

    def init_weights(self, module):
        nb_params = torch.tensor(module.weight.shape[1:]).prod()
        if self.mean_weight == 'auto':
            mean = .5
        else:
            mean = self.mean_weight
        # std = mean / (2 * np.sqrt(3))
        # lb = mean * (1 - std)
        # ub = mean * (1 + std)
        lb = mean / 2
        ub = 3 * lb
        module.set_param_from_weights(
            torch.rand_like(module.weights) * (lb - ub) + ub
        )

        self.mean = mean
        self.nb_params = nb_params

    def init_bias(self, module):
        new_value = self.input_mean * self.mean * self.nb_params
        module.set_bias(
            torch.zeros_like(module.bias) - new_value + self.init_bias_value
        )


class InitSybiseConstantVarianceWeights(InitWeightsThenBias):
    def __init__(self, input_mean: float = 0, mean_weight: float = "auto", init_bias_value: float = 0, *args, **kwargs) -> None:
        self.input_mean = input_mean
        self.mean_weight = mean_weight
        self.init_bias_value = init_bias_value

        self.mean = None
        self.nb_params = None

    def init_weights(self, module):
        p = 1
        nb_params = torch.tensor(module.weight.shape[1:]).prod()

        if self.mean_weight == "auto":
            ub = 1 / (p * torch.sqrt(nb_params))
            lb = np.sqrt(3 / 4) * ub
            mean = (lb + ub) / 2

        sigma = 1 / (p ** 2 * nb_params) - mean ** 2
        diff = torch.sqrt(3 * sigma)
        lb = mean - diff
        ub = mean + diff
        module.set_param_from_weights(
            torch.rand_like(module.weights) * (lb - ub) + ub
        )

        self.mean = mean
        self.nb_params = nb_params

    def init_bias(self, module):
        new_value = self.input_mean * self.mean * self.nb_params
        module.set_bias(
            torch.zeros_like(module.bias) - new_value + self.init_bias_value
        )


class InitSybiseConstantVarianceWeightsRandomBias(InitSybiseConstantVarianceWeights):
    def __init__(self, ub: float = 0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ub = ub

    def init_bias(self, module):
        new_value = self.input_mean * self.mean * self.nb_params * (1 + uniform_sampling_bound(-self.ub, self.ub).astype(np.float32))
        module.set_bias(
            torch.zeros_like(module.bias) - new_value + self.init_bias_value
        )


class InitSybiseConstantVarianceConstantWeights(InitSybiseConstantVarianceWeights):
    """We init the LUI with a mean weights instead of a random uniform function. We take the same mean for simplicity.
    """

    def init_weights(self, module):
        p = 1
        nb_params = torch.tensor(module.weight.shape[1:]).prod()

        if self.mean_weight == "auto":
            ub = 1 / (p * torch.sqrt(nb_params))
            lb = np.sqrt(3 / 4) * ub
            mean = (lb + ub) / 2

        new_weights = torch.ones_like(module.weights) * mean / nb_params

        module.set_param_from_weights(
            new_weights
        )

        self.mean = mean
        self.nb_params = nb_params


class InitSybiseConstantVarianceConstantWeightsRandomBias(InitSybiseConstantVarianceConstantWeights):
    def __init__(self, ub: float = 0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ub = ub

    def init_bias(self, module):
        new_value = self.input_mean * self.mean * self.nb_params * (1 + uniform_sampling_bound(-self.ub, self.ub).astype(np.float32))
        module.set_bias(
            torch.zeros_like(module.bias) - new_value + self.init_bias_value
        )


class InitSybiseHeuristicWeightsRandomBias(InitSybiseHeuristicWeights):
    def __init__(self, ub: float = 0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ub = ub

    def init_bias(self, module):
        new_value = self.input_mean * self.mean * self.nb_params * (1 + uniform_sampling_bound(-self.ub, self.ub).astype(np.float32))
        module.set_bias(
            torch.zeros_like(module.bias) - new_value + self.init_bias_value
        )


class InitBiseEllipseWeights(InitBiasFixed):
    def init_weights(self, module):
        param = torch.FloatTensor(size=module.weights_handler.param.shape)

        shape = module.weights_handler.shape[2:]
        dim = module.weights_handler.dim

        # mu
        param[:, :, :dim] = torch.tensor(shape) // 2

        # sigma_inv
        sigma_inv = torch.zeros(*param.shape[:2], dim, dim)
        sigma_inv[:, :, torch.arange(dim), torch.arange(dim)] = torch.tensor([.5 for _ in range(dim)]) + torch.randn(*param.shape[:2], dim) * .1
        param[:, :, dim: dim + dim ** 2] = sigma_inv.view(*param.shape[:2], -1)

        # a_
        param[:, :, -1] = 1

        module.set_weights_param(param)


class InitBiseEllipseWeightsRoot(InitBiasFixed):
    def init_weights(self, module):
        param = torch.FloatTensor(size=module.weights_handler.param.shape)

        shape = module.weights_handler.shape[2:]
        dim = module.weights_handler.dim

        # mu
        param[:, :, :dim] = torch.tensor(shape) // 2

        # sigma_inv
        root_sigma_inv = torch.randn(*param.shape[:2], dim, dim) * .5
        # root_sigma_inv[:, :, torch.arange(dim), torch.arange(dim)] = torch.randn(*param.shape[:2], dim)
        param[:, :, dim: dim + dim ** 2] = root_sigma_inv.view(*param.shape[:2], -1)

        # a_
        param[:, :, -1] = 1

        module.set_weights_param(param)
