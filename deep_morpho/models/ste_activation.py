from abc import ABC

import torch
import torch.nn as nn
from torch.autograd import Function


def sign(x: torch.Tensor) -> torch.Tensor:
    """torch.sign with f(0) = +1"""
    return (x >= 0).float() * 2 - 1
    # return (x > -0.01).float() * 2 - 1


class STELayer(nn.Module, ABC):
    STE_FN: Function

    def forward(self, *args, **kwargs):
        return self.STE_FN.apply(*args, **kwargs)

    def real_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _STEClippedIdentity(Function):
    """Defined in Binarized Neural Networks, Courbariaux et al. 2016, https://arxiv.org/abs/1602.02830
    Function version of the nn.Module STEClippedIdentity"""
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_)
        return sign(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ > 1] = 0
        grad_input[input_ < -1] = 0
        return grad_input


class STEClippedIdentity(STELayer):
    """nn.Module version of the activation _STEClippedIdentity"""
    STE_FN = _STEClippedIdentity


class _STEQuantizedK(Function):
    """Defined in DoReFa-Net, Zhou et al. 2016, https://arxiv.org/abs/1606.06160
    Function version of the nn.Module STEQuantizedK
    Input must be in [0, 1]"""
    @staticmethod
    def forward(ctx, input_: torch.Tensor, k: int = 1) -> torch.Tensor:
        assert input_.min() >= 0 and input_.max() <= 1, "input must be in [0, 1]"
        ctx.k = k
        cst = 2 ** k - 1
        return torch.round(input_ * cst) / cst

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clone(), None


class STEQuantizedK(STELayer):
    """nn.Module version of the activation _STEQuantizedK"""
    STE_FN = _STEQuantizedK

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.STE_FN.apply(x, self.k)


class _STEBernoulli(Function):
    """Defined in DoReFa-Net, Zhou et al. 2016, https://arxiv.org/abs/1606.06160
    Function version of the nn.Module STEBernoulli
    """
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        return torch.bernoulli(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clone()


class STEBernoulli(STELayer):
    """nn.Module version of the activation STEBernoulli"""
    STE_FN = _STEBernoulli


class _STEXNor(Function):
    """Defined in XNOR-Net, Rastegari et al. 2016, https://arxiv.org/abs/1603.05279
    Function version of the nn.Module STEXNor
    """
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        """input_ is a tensor of shape (out_channels, in_channels, height, width)"""
        assert input_.dim() == 4, "input must be a tensor of shape (batch_size, channels, height, width)"
        return sign(input_) * input_.abs().mean((0, 2, 3))[..., None, None]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clone()


class STEXNor(STELayer):
    """nn.Module version of the activation _STEXNor"""
    STE_FN = _STEXNor


class _STEDoReFaBinary(Function):
    """Defined in DoReFa-Net, Zhou et al. 2016, https://arxiv.org/abs/1606.06160
    Function version of the nn.Module STEDoReFaBinary
    """
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        return sign(input_) * input_.abs().mean()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clone()


class STEDoReFaBinary(STELayer):
    """nn.Module version of the activation _STEDoReFaBinary"""
    STE_FN = _STEDoReFaBinary
