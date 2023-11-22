import torch.nn as nn


class MLP(nn.Module):
    """ Multi Layer Perceptron class.
    """

    def __init__(self, list_neurons, activation, bias=True):
        super().__init__()

        self.list_neurons = list_neurons
        self.activation = activation
        self.bias = bias

        self.mlp = self._builder(list_neurons, bias)

    @staticmethod
    def _builder(list_neurons, bias):
        dense_layers = []
        for idx, neurons in enumerate(list_neurons[:-1], 1):
            dense_layers.append(nn.Linear(neurons, list_neurons[idx], bias=bias))
        return nn.ModuleList(dense_layers)

    def forward(self, x, *args, **kwargs):
        output = x

        # If output is not the correct dimension, reshape
        if output.ndim > 2:
            output = output.view(output.shape[0], -1)

        for layer in self.mlp[:-1]:
            output = self.activation(layer(output))
        output = self.mlp[-1](output)
        return output
