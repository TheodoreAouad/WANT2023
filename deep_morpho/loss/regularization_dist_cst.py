from typing import List, Dict

import torch
import torch.nn as nn

from ..models import BiSEBase
from ..binarization.bise_closest_selem import BiseClosestMinDistOnCst
from ..binarization.projection_constant_set import ProjectionConstantSet


class RegularizationProjConstant(nn.Module):
    r""" Adds a regularization loss to encourage the bimonn to be morphological.
    For a structuring element $S$, $A(S) = \{\theta \cdot S | \theta > 0\}$. We compute
    $$ \min_S d(A(S), (W, B)) $$.
    """
    def __init__(self, model: nn.Module = None, bise_modules: List[BiSEBase] = None, mode: str = "exact"):
        super().__init__()
        self.model = model
        self.bise_modules = bise_modules
        self.mode = mode

    def forward(self, *args, pl_module=None, **kwargs) -> torch.Tensor:
        if self.model is None:
            self.set_model(pl_module.model)

        loss = 0
        for handler in self.bise_closest_handlers:
            _, _, dist = handler(return_np_array=False, verbose=False)
            loss += dist.sum()
        return loss

    def set_model(self, model: nn.Module):
        self.model = model
        if self.bise_modules is None:
            self.bise_modules = [m for m in self.model.modules() if isinstance(m, BiSEBase)]

        self.bise_closest_handlers = [BiseClosestMinDistOnCst(bise_module=bise, mode=self.mode) for bise in self.bise_modules]
        return self


# class RegularizationProjConstantApproxUniform(nn.Module):
#     r""" Based on Ternary Weight Networks (https://arxiv.org/pdf/1605.04711.pdf).
#     For a structuring element $S$, $A(S) = \{\theta \cdot S | \theta > 0\}$. We compute
#     an approximation of $$ \min_S d(A(S), (W, B)) $$. If $W$ are uniform between [0, alpha], then
#     the best threshold is alpha / 3.
#     """
#     def __init__(self, model: nn.Module = None, bise_modules: List[BiSEBase] = None):
#         super().__init__()
#         self.model = model
#         self.bise_modules = bise_modules
