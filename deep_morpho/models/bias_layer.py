from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .bise import BiSE

import torch
import torch.nn as nn

from .softplus import Softplus
from .bise_module_container import BiseModuleContainer
from general.nn.experiments.experiment_methods import ExperimentMethods


class BiasBise(nn.Module, ExperimentMethods):
    """Base class to deal with bias in BiSE like neurons. We suppose that the bias = f(param).
    """

    def __init__(self, bise_module: "BiSE", *args, **kwargs):
        super().__init__()
        self._bise_module: BiseModuleContainer = BiseModuleContainer(bise_module)
        self.param = self.init_param(*args, **kwargs)

    def forward(self,) -> torch.Tensor:
        return self.from_param_to_bias(self.param)

    def forward_inverse(self, bias: torch.Tensor,) -> torch.Tensor:
        return self.from_bias_to_param(bias)

    def from_param_to_bias(self, param: torch.Tensor,) -> torch.Tensor:
        raise NotImplementedError

    def from_bias_to_param(self, bias: torch.Tensor,) -> torch.Tensor:
        raise NotImplementedError

    @property
    def bise_module(self):
        return self._bise_module.bise_module

    @property
    def conv(self):
        return self.bise_module.conv

    @property
    def shape(self):
        return (self.conv.weight.shape[0], )

    def init_param(self, *args, **kwargs) -> torch.Tensor:
        return nn.Parameter(torch.FloatTensor(size=self.shape))

    def set_bias(self, new_bias: torch.Tensor) -> torch.Tensor:
        new_param = self.from_bias_to_param(new_bias)
        self.set_param(new_param)
        return new_param

    def set_param(self, new_param: torch.Tensor) -> torch.Tensor:
        self.param.data = new_param
        return new_param

    def set_param_from_bias(self, new_bias: torch.Tensor) -> torch.Tensor:
        assert new_bias.shape == self.bise_module.bias.shape, f"Bias must be of same shape {self.bise_module.bias.shape}"
        new_param = self.from_bias_to_param(new_bias)
        self.set_param(new_param)
        return new_param

    @property
    def grad(self):
        return self.param.grad


class BiasRaw(BiasBise):
    def from_param_to_bias(self, param: torch.Tensor) -> torch.Tensor:
        return param

    def from_bias_to_param(self, bias: torch.Tensor) -> torch.Tensor:
        return bias


class BiasSoftplus(BiasBise):

    def __init__(self, offset: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset
        self.softplus_layer = Softplus()

    def from_param_to_bias(self, param: torch.Tensor) -> torch.Tensor:
        return -self.softplus_layer(param) - self.offset

    def from_bias_to_param(self, bias: torch.Tensor) -> torch.Tensor:
        return self.softplus_layer.forward_inverse(-bias - self.offset)


class BiasBiseSoftplusReparametrized(BiasSoftplus):

    def __init__(self, *args, offset: float = 0, **kwargs):
        super().__init__(offset=offset, *args, **kwargs)

    def get_min_max_intrinsic_bias_values(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """We compute the min and max values that are really useful for the bias. We suppose that if the bias is out of
        these bounds, then it is nonsense.
        The bounds are computed to avoid having always a negative convolution output or a positive convolution output.
        """
        weights_aligned = self.bise_module.weight.reshape(self.bise_module.weight.shape[0], -1)
        weights_min = weights_aligned.min(1).values
        weights_2nd_min = weights_aligned.kthvalue(2, 1).values
        weights_sum = weights_aligned.sum(1)

        bmin = 1/2 * (weights_min + weights_2nd_min)
        bmax = weights_sum - 1/2 * weights_min

        return bmin, bmax

    def from_param_to_bias(self, param: torch.Tensor) -> torch.Tensor:
        bmin, bmax = self.get_min_max_intrinsic_bias_values()
        return -torch.clamp(self.softplus_layer(param), bmin, bmax)


class BiasBiseSoftplusProjected(BiasBiseSoftplusReparametrized):

    def from_param_to_bias(self, param: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            bmin, bmax = self.get_min_max_intrinsic_bias_values()

        return -torch.clamp(self.softplus_layer(param), bmin, bmax)
