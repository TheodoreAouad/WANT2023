import torch.nn as nn


class BoundRegularization(nn.Module):

    def __init__(self, model, lambda_=1, lower_bound=0, upper_bound=1, **kwargs):
        super().__init__()
        self.model = model
        self.lambda_ = lambda_
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_params_from_model(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        loss = 0
        for weights in self.get_params_from_model():
            loss += self.penalize_weight(weights)
        return self.lambda_ * loss

    def penalize_weight(self, weights):
        return (
            self.lower_bound_pen(weights[weights < self.lower_bound]).sum() +
            self.upper_bound_pen(weights[weights > self.upper_bound]).sum()
        )

    def lower_bound_pen(self, x):
        raise NotImplementedError

    def upper_bound_pen(self, x):
        raise NotImplementedError


class BimonnBoundRegularization(BoundRegularization):
    def get_params_from_model(self):
        return [bisel_layer.weight for bisel_layer in self.model.layers]


class QuadraticBoundRegularization(BimonnBoundRegularization):

    def lower_bound_pen(self, x):
        return x ** 2

    def upper_bound_pen(self, x):
        return (x - 1) ** 2


class LinearBoundRegularization(BimonnBoundRegularization):

    def lower_bound_pen(self, x):
        return -x

    def upper_bound_pen(self, x):
        return x - 1
