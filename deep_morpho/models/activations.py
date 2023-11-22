import torch.nn as nn


class NormalizedTanh(nn.Tanh):
    def forward(self, input):
        return super().forward(input) / 2 + 0.5
