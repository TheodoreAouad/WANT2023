from enum import Enum, auto
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..threshold_fn import *


class ThresholdLayer(nn.Module):

    def __init__(
        self,
        threshold_fn,
        threshold_inverse_fn=None,
        P_: float = 1,
        n_channels: int = 1,
        axis_channels: int = 1,
        threshold_name: str = '',
        bias: float = 0,
        constant_P: bool = False,
        binary_mode: bool = False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.axis_channels = axis_channels
        self.threshold_name = threshold_name
        self.threshold_fn = threshold_fn
        self.threshold_inverse_fn = threshold_inverse_fn
        self.bias = bias
        self.binary_mode = binary_mode

        if isinstance(P_, nn.Parameter):
            self.P_ = P_
        else:
            self.P_ = nn.Parameter(torch.tensor([P_ for _ in range(n_channels)]).float())
        if constant_P:
            self.P_.requires_grad = False

    def forward(self, x, binary_mode=None):
        if binary_mode is None:
            binary_mode = self.binary_mode

        if binary_mode:
            return x > 0

        return self.apply_threshold(x, self.P_, self.bias)

    def apply_threshold(self, x, P_, bias):
        return self.threshold_fn(
            (x + bias) * P_.view(*([1 for _ in range(self.axis_channels)] + [len(P_)] + [1 for _ in range(self.axis_channels, x.ndim - 1)]))
        )

    def forward_inverse(self, y):
        return self.apply_threshold_inverse(y, self.P_, self.bias)

    def apply_threshold_inverse(self, y, P_, bias):
        assert self.threshold_inverse_fn is not None
        return (
            1 / P_.view(*([1 for _ in range(self.axis_channels)] + [len(P_)] + [1 for _ in range(self.axis_channels, y.ndim - 1)])) *
            self.threshold_inverse_fn(y) - bias
        )


class SigmoidLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=sigmoid_threshold, threshold_inverse_fn=sigmoid_threshold_inverse, threshold_name='sigmoid', *args, **kwargs)


class ReLULayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=relu_threshold, threshold_inverse_fn=relu_threshold_inverse, threshold_name='relu', *args, **kwargs)


class ArctanLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=arctan_threshold, threshold_inverse_fn=arctan_threshold_inverse, threshold_name='arctan', *args, **kwargs)


class TanhLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=tanh_threshold, threshold_inverse_fn=tanh_threshold_inverse, threshold_name='tanh', *args, **kwargs)


class ErfLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=erf_threshold, threshold_inverse_fn=erf_threshold_inverse, threshold_name='erf', *args, **kwargs)


class ClampLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=lambda x: clamp_threshold(x, 0, 1), threshold_name='clamp', *args, **kwargs)


class IdentityLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=lambda x: x, threshold_inverse_fn=lambda x: x, threshold_name='identity', *args, **kwargs)


class SoftplusThresholdLayer(ThresholdLayer):
    def __init__(self, beta: int = 1, threshold: int = 20, *args, **kwargs):
        super().__init__(threshold_fn=lambda x: F.softplus(x, beta, threshold), threshold_inverse_fn=lambda x: softplus_threshold_inverse(x, beta), *args, **kwargs)
        self.beta = beta
        self.threshold = threshold


class TanhSymetricLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=tanh_threshold_symetric, threshold_inverse_fn=tanh_threshold_symetric_inverse, threshold_name='tanh_symetric', *args, **kwargs)


class SigmoidSymetricLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=sigmoid_threshold_symetric, threshold_inverse_fn=sigmoid_threshold_symetric_inverse, threshold_name='sigmoid_symetric', *args, **kwargs)




class ThresholdEnum(Enum):
    sigmoid = auto()
    arctan = auto()
    tanh = auto()
    erf = auto()
    clamp = auto()
    identity = auto()
    softplus = auto()
    tanh_symetric = auto()
    sigmoid_symetric = auto()


dispatcher = {
    ThresholdEnum.sigmoid: SigmoidLayer,
    ThresholdEnum.sigmoid_symetric: SigmoidSymetricLayer,
    ThresholdEnum.arctan: ArctanLayer,
    ThresholdEnum.tanh: TanhLayer,
    ThresholdEnum.tanh_symetric: TanhSymetricLayer,
    ThresholdEnum.erf: ErfLayer,
    ThresholdEnum.clamp: ClampLayer,
    ThresholdEnum.identity: IdentityLayer,
    ThresholdEnum.softplus: SoftplusThresholdLayer,
    "sigmoid": SigmoidLayer,
    "sigmoid_symetric": SigmoidSymetricLayer,
    "arctan": ArctanLayer,
    "tanh": TanhLayer,
    "tanh_symetric": TanhSymetricLayer,
    "erf": ErfLayer,
    "clamp": ClampLayer,
    "identity": IdentityLayer,
    "softplus": SoftplusThresholdLayer,
    "relu": ReLULayer,
}
