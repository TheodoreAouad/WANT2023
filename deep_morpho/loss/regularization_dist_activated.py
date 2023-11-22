from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp

from ..models import BiSEBase
from ..binarization.bise_closest_selem import BiseClosestMinDistOnCst
from ..binarization.projection_constant_set import ProjectionConstantSet


# TODO: handle erosion
# TODO: add case 1 of bise proj
class DifferentiableProjectionActivated:
    """ Applies projection, and returns a loss with a gradient.
    Weights and bias are such that the conv is of the form $I * weights - bias$.
    1) We compute the projection of the weights and bias on the set of activated parameters for each selem
    2) We get the selem with the smallest loss
    3) We deduce the expressions of T and K, necessary to compute the analytical form
    4) We compute the analytical form of the loss, allowing to compute its gradient

    Args:
        bise_module (BiSEBase): the bise module
        weights (torch.Tensor): shape (n_params)
        bias (torch.Tensor): shape (1)

    Returns:
        torch.Tensor: the loss

    Example:
    >>> regu = DifferentiableProjectionActivated(bise_module=bise_module)
    >>> loss = regu()
    >>> loss.backward()

    Example:
    >>> weights = torch.rand(3, 3, requires_grad=True)
    >>> bias = torch.rand(1, requires_grad=True)

    """

    def __init__(self, bise_module: BiSEBase = None, weights: torch.Tensor = None, bias: torch.Tensor = None, do_sort_weights: bool = True):
        assert bise_module is not None or (weights is not None and bias is not None), "Either bise_module or weights and bias must be provided."

        self.bise_module = bise_module
        self._weights = weights
        self._bias = bias
        self.do_sort_weights = do_sort_weights

    @property
    def weights(self):
        if self._weights is not None:
            return self._weights
        return self.bise_module.weights

    @property
    def bias(self):
        if self._bias is not None:
            return self._bias
        return -self.bise_module.bias

    def find_best_S(self) -> Tuple[np.ndarray, str]:
        if self.do_sort_weights:
            weight_values = self.weights.unique()
        else:
            weight_values = self.weights

        best_loss = np.infty
        best_S = None
        best_operation = None
        best_proj = None

        for value in weight_values:
            for operation in ["dilation", "erosion"]:
                S = self.weights >= value
                proj = self.solve_proj(operation=operation, S=S)  # TODO: ADD EROSION
                # print(proj.value)
                if proj.value < best_loss:
                    best_loss = proj.value
                    best_S = S
                    best_operation = operation
                    best_proj = proj
                if proj.value == 0:
                    return best_S, best_operation, best_proj

        return best_S, best_operation, best_proj

    @property
    def weight(self):
        return self.weights

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        S, operation, proj = self.find_best_S()
        if proj.value == 0:
            return 0

        self.proj = proj
        self.operation = operation
        self.S = S
        self.T = (self.weights <= torch.tensor(proj.b, device=self.weight.device)) & S
        self.K = (self.weights <= torch.tensor(proj.lambda0, device=self.weight.device)) & ~S

        return self.analytical_loss(weight=self.weights, bias=self.bias, S=S, T=self.T, K=self.K)

    def solve_proj(self, operation: str, S: np.ndarray) -> cp.Problem:
        """ Solves the optimization problem for a given structuring element and weight/bias.
        Args:
            weight (torch.Tensor): shape (n_params)
            bias (torch.Tensor): shape (1)
            operation (str): either "dilation" or "erosion"
            S (np.ndarray): the structuring element. bool (n_params)

        Returns:
            cp.Problem: the optimization projection
        """
        W = self.weight.cpu().detach().numpy()
        bias = self.bias.cpu().detach().numpy()
        S = S.cpu().detach().numpy()

        if operation == "erosion":
            bias = W.sum() - bias

        # W = W.flatten()
        # S = S.flatten()

        Wvar = cp.Variable(W.shape)
        bvar = cp.Variable(1)

        constraints = self.dilation_constraints(Wvar, bvar, S)
        # if operation == "dilation":
        # elif operation == "erosion":
        #     constraints = self.erosion_constraints(Wvar, bvar, S)
        # else:
        #     raise ValueError("operation must be dilation or erosion")

        objective = cp.Minimize(1/2 * cp.sum_squares(Wvar - W) + 1/2 * cp.sum_squares(bvar - bias))
        proj = cp.Problem(objective, constraints)
        proj.solve()

        proj.b = bvar.value
        proj.bvar = bvar
        proj.Wvar = Wvar
        proj.lambda0 = constraints[0].dual_value

        return proj

    def analytical_loss(self, weight, bias, S: torch.Tensor, T: torch.Tensor, K: torch.Tensor):
        self.analytical = {}
        Wini, bini = weight, bias

        card_sbar = (~S).sum()
        card_T = T.sum()
        card_K = K.sum()
        card_Kbar = card_sbar - card_K
        wsum_T = Wini[T].sum()
        wsum_Kbar = Wini[~S & ~K].sum()

        denom = (1 / (card_Kbar * (card_T  + 1) + 1))

        b = denom * (wsum_Kbar + card_Kbar * (wsum_T + bini))
        self.analytical["b"] = b

        lambdat = torch.zeros_like(Wini)
        lambdat[T] = b - Wini[T]
        self.analytical["lambdat"] = lambdat

        lambda0 = denom * ((card_T + 1) * wsum_Kbar - wsum_T - bini)
        self.analytical["lambda0"] = lambda0

        lambdak = torch.zeros_like(Wini)
        lambdak[K] = lambda0 - Wini[K]
        self.analytical["lambdak"] = lambdak

        W = torch.zeros_like(Wini)
        W[T] = b
        W[S & ~T] = Wini[S & ~T]
        W[~K & ~S] = Wini[~K & ~S] - lambda0
        self.analytical["W"] = W

        return 1/2 * (((W - Wini) ** 2).sum() + (b - bini) ** 2)

    def dilation_constraints(self, Wvar: cp.Variable, bvar: cp.Variable, S: np.ndarray):
        constraint0 = [cp.sum(Wvar[~S]) <= bvar]
        constraintsT = [bvar <= Wvar[S]]
        constraintsK = [Wvar >= 0]
        return constraint0 + constraintsT + constraintsK

    def erosion_constraints(self, Wvar: cp.Variable, bvar: cp.Variable, S: np.ndarray):
        constraint0 = [cp.sum(Wvar[S]) >= bvar]
        constraintsT = [cp.sum(Wvar) - Wvar[S] <= bvar]
        constraintsK = [Wvar >= 0]
        return constraint0 + constraintsT + constraintsK


class RegularizationProjActivated(nn.Module):
    r""" Adds a regularization loss to encourage the bimonn to be morphological.
    For a structuring element $S$ and a morphological operator $\psi$, $A(S, \psi)$ is the space of activated weights 
    and biases. We compute $$ \min_S d(A(S), (W, B)) $$.
    """
    def __init__(self, model: nn.Module = None, bise_modules: List[BiSEBase] = None):
        super().__init__()
        self.model = model
        self.bise_modules = bise_modules
        self.projectors = None

    def forward(self, *args, pl_module=None, **kwargs) -> torch.Tensor:
        if self.model is None:
            self.set_model(pl_module.model)

        loss = 0
        for bise_module in self.bise_modules:
            for weight, bias in zip(bise_module.weights, bise_module.bias):
                loss += DifferentiableProjectionActivated(weights=weight.reshape(-1), bias=-bias)()
        return loss

    def set_model(self, model: nn.Module):
        self.model = model
        if self.bise_modules is None:
            self.bise_modules = [
                m for m in self.model.modules() if isinstance(m, BiSEBase)
            ]

        return self
