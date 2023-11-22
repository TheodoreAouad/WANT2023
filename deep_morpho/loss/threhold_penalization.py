from typing import List

from ..models import BiSE


class ThresholdPenalization:

    def __init__(self, bise_layers: List[BiSE], coef: float = .5, degree: int = 2, detach_weights: bool = True,
                 epsilon: float = .5):
        self.bise_layers = bise_layers
        self.coef = coef
        self.loss_fn = getattr(self, f"polynome_{degree}")
        self.detach_weights = detach_weights
        self.epsilon = epsilon

    def __call__(self):
        loss = 0
        for dilation in self.bise_layers:
            sum_weights = (dilation.weight > .5).sum()
            if self.detach_weights:
                sum_weights = sum_weights.detach()
            loss += self.coef * self.loss_fn(dilation.bias, 1 - self.epsilon, sum_weights - self.epsilon)
        return loss

    @staticmethod
    def polynome_2(bias, x, y):
        return (x + bias).abs() * (y + bias).abs()

    @staticmethod
    def polynome_4(bias, x, y):
        return (x + bias) ** 2 * (y + bias) ** 2
