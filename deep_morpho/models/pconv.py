import torch
import torch.nn as nn


class PConv2d(nn.Conv2d):

    def __init__(self, P_: float, *args, **kwargs):
        super().__init__(bias=0, *args, **kwargs)
        self.P_ = nn.Parameter(torch.tensor([P_]).float())

    def forward(self, x: torch.Tensor):
        return self.forward(x**(self.P_ + 1)) / self.forward(x**(self.P_))
