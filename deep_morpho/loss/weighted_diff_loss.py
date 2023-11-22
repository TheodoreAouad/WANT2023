from typing import Callable

import torch.nn as nn


class LossWeightedDiff(nn.Module):
    """ Puts more weight on the differences.
    """

    def __init__(self, loss: Callable, weight: float, **kwargs):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, ypred, ytrue):
        if self.border[0] != 0:
            ypred = ypred[..., self.border[0]:-self.border[0], :]
            ytrue = ytrue[..., self.border[0]:-self.border[0], :]

        if self.border[1] != 0:
            ypred = ypred[..., self.border[1]:-self.border[1]]
            ytrue = ytrue[..., self.border[1]:-self.border[1]]

        return self.loss(ypred, ytrue)
