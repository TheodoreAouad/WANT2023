from typing import TYPE_CHECKING, Tuple
from functools import partial

if TYPE_CHECKING:
    from .bise import BiSE

import torch
import torch.nn as nn
import numpy as np

from .bise_module_container import BiseModuleContainer
from .threshold_layer import dispatcher
from general.nn.experiments.experiment_methods import ExperimentMethods


class WeightsBise(nn.Module, ExperimentMethods):
    """Base class to deal with weights in BiSE like neurons. We suppose that the weights = f(param).
    """

    def __init__(self, bise_module: "BiSE", *args, **kwargs):
        super().__init__()
        self._bise_module: BiseModuleContainer = BiseModuleContainer(bise_module)
        self.param: torch.Tensor = self.init_param(*args, **kwargs)

    def forward(self,) -> torch.Tensor:
        return self.from_param_to_weights(self.param)

    def forward_inverse(self, weights: torch.Tensor,) -> torch.Tensor:
        return self.from_weights_to_param(weights)

    def from_param_to_weights(self, param: torch.Tensor,) -> torch.Tensor:
        raise NotImplementedError

    def from_weights_to_param(self, weights: torch.Tensor,) -> torch.Tensor:
        raise NotImplementedError

    @property
    def bise_module(self):
        return self._bise_module.bise_module

    @property
    def conv(self):
        return self.bise_module.conv

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def shape(self):
        return self.conv.weight.shape

    def init_param(self, *args, **kwargs) -> torch.Tensor:
        return nn.Parameter(torch.FloatTensor(size=self.shape))

    def set_param_from_weights(self, new_weights: torch.Tensor) -> torch.Tensor:
        assert new_weights.shape == self.bise_module.weight.shape, f"Weights must be of same shape {self.bise_module.weight.shape}"
        new_param = self.from_weights_to_param(new_weights)
        self.set_param(new_param)
        return new_param

    def set_param(self, new_param: torch.Tensor) -> torch.Tensor:
        self.param.data = new_param
        return new_param

    @property
    def grad(self):
        return self.param.grad

    @property
    def device(self):
        return self.param.device


class WeightsThresholdedBise(WeightsBise):

    def __init__(self, threshold_mode: str, P_=1, constant_P: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_mode = threshold_mode
        self.threshold_layer = dispatcher[threshold_mode](P_=P_, constant_P=constant_P, axis_channels=0, n_channels=self.out_channels,)

    def from_param_to_weights(self, param: torch.Tensor) -> torch.Tensor:
        return self.threshold_layer.forward(param)

    def from_weights_to_param(self, weights: torch.Tensor) -> torch.Tensor:
        # assert (weights >= 0).all(), weights
        return self.threshold_layer.forward_inverse(weights)


class WeightsNormalizedBiSE(WeightsThresholdedBise):

    def __init__(self, factor: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor

    def from_param_to_weights(self, param: torch.Tensor) -> torch.Tensor:
        weights = super().from_param_to_weights(param)
        # return self.factor * weights / weights.sum()
        return self.factor * weights / weights.sum((1, 2, 3)).view(-1, 1, 1, 1)


class WeightsRaw(WeightsThresholdedBise):

    def __init__(self, *args, **kwargs):
        super().__init__(threshold_mode='identity', *args, **kwargs)


# DEPRECATED
class WeightsEllipse(WeightsBise):
    def init_param(self, *args, **kwargs) -> torch.Tensor:
        return nn.Parameter(torch.FloatTensor(size=(
            self.shape[:2] +  # out_channels, in_channels
            (  # ellipse parameters
                self.dim +  # mu
                self.dim ** 2 +  # sigma
                1   # exponent parameter
            ,)
        )))
        # return nn.Parameter(torch.FloatTensor(size=([1, 1, 7])))

    def get_mu(self, param: torch.Tensor) -> torch.Tensor:
        return param[:, :, :self.dim]

    def get_sigma_inv(self, param: torch.Tensor) -> torch.Tensor:
        return param[:, :, self.dim:self.dim + self.dim**2].reshape(*self.shape[:2] + (self.dim, self.dim))

    def get_a(self, param: torch.Tensor) -> torch.Tensor:
        return param[:, :, -1]

    def get_mu_grad(self, param: torch.Tensor) -> torch.Tensor:
        if self.param.grad is not None:
            return param.grad[:, :, :self.dim]

    def get_sigma_inv_grad(self, param: torch.Tensor) -> torch.Tensor:
        if self.param.grad is not None:
            return param.grad[:, :, self.dim:self.dim + self.dim**2].reshape(*self.shape[:2] + (self.dim, self.dim))

    def get_a_grad(self, param: torch.Tensor) -> torch.Tensor:
        if self.param.grad is not None:
            return param.grad[:, :, -1]

    @property
    def mu(self) -> torch.Tensor:
        return self.get_mu(self.param)

    @property
    def sigma_inv(self) -> torch.Tensor:
        return self.get_sigma_inv(self.param)

    @property
    def a_(self) -> torch.Tensor:
        return self.get_a(self.param)

    @property
    def mu_grad(self) -> torch.Tensor:
        return self.get_mu_grad(self.param)

    @property
    def sigma_inv_grad(self) -> torch.Tensor:
        return self.get_sigma_inv_grad(self.param)

    @property
    def a_grad(self) -> torch.Tensor:
        return self.get_a_grad(self.param)

    @property
    def dim(self) -> int:
        return len(self.shape[2:])

    def from_param_to_weights(self, param: torch.Tensor) -> torch.Tensor:
        mu, sigma_inv, a_ = self.get_mu(param), self.get_sigma_inv(param), self.get_a(param)
        return self.get_weights_from_ellipse(mu, sigma_inv, a_)

    @staticmethod
    def ellipse_level(x: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor, a_: torch.Tensor) -> torch.Tensor:
        mu = mu[:, None]
        A1 = (x - mu).T @ sigma_inv
        A2 = (x - mu).T
        res = torch.bmm(A1[:, None, :], A2[..., None])
        return torch.exp(-res ** a_)

    @staticmethod
    def ellipse_matrix(shape: Tuple, mu: torch.Tensor, sigma_inv: torch.Tensor, a_: torch.Tensor, device='cpu') -> torch.Tensor:
        # ellipse_fn = partial(self.ellipse_level, mu=mu, sigma_inv=sigma_inv, a_=a_)

        XX, YY = torch.meshgrid(torch.arange(shape[0], device=device), torch.arange(shape[1], device=device))
        coords = torch.vstack([XX.flatten(), YY.flatten()])

        return WeightsEllipse.ellipse_level(coords, mu=mu, sigma_inv=sigma_inv, a_=a_).reshape(*shape)

    def get_one_ellipse_matrix(self, mu: torch.Tensor, sigma_inv: torch.Tensor, a_: torch.Tensor) -> torch.Tensor:
        return self.ellipse_matrix(shape=self.shape[:2], mu=mu, sigma_inv=sigma_inv, a_=a_, device=self.device)
        # shape_W = self.shape[2:]
        # ellipse_fn = partial(self.ellipse_level, mu=mu, sigma_inv=sigma_inv, a_=a_)

        # XX, YY = torch.meshgrid(torch.arange(shape_W[0], device=self.device), torch.arange(shape_W[1], device=self.device))
        # coords = torch.vstack([XX.flatten(), YY.flatten()])

        # return ellipse_fn(coords).reshape(*shape_W)


    def get_weights_from_ellipse(self, mu: torch.Tensor, sigma_inv: torch.Tensor, a_: torch.Tensor) -> torch.Tensor:
        res = torch.FloatTensor(size=self.shape).to(self.device)

        for chan1 in range(self.shape[0]):
            for chan2 in range(self.shape[1]):
                res[chan1, chan2] = self.get_one_ellipse_matrix(mu[chan1, chan2], sigma_inv[chan1, chan2], a_[chan1, chan2])

        return res


class WeightsEllipseRoot(WeightsEllipse):
    """Computes the sigma_inv of the ellipse using its square root. In practice, we learn the square root.
    """

    def get_sigma_inv(self, param: torch.Tensor) -> torch.Tensor:
        A = param[:, :, self.dim:self.dim + self.dim**2].view((self.shape[0] * self.shape[1],) + (self.dim, self.dim))
        # A = param[:, :, self.dim:self.dim + self.dim**2].reshape(*self.shape[:2] + (self.dim, self.dim))
        At = A.transpose(-2, -1)

        sigma_inv = torch.bmm(At, A)
        return sigma_inv.reshape(self.shape[:2] + (self.dim, self.dim))

    @property
    def root_sigma_inv(self):
        return super().get_sigma_inv(self.param)
