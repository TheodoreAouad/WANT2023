from abc import ABC
import torch.nn as nn

from general.nn.experiments.experiment_methods import ExperimentMethods


class BinaryNN(nn.Module, ExperimentMethods, ABC):

    def __init__(self):
        super().__init__()
        self.binary_mode = False

    def binary(self, mode: bool = True, *args, **kwargs):
        r"""Sets the module in binary mode.

        Args:
            mode (bool): whether to set binary mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.binary_mode = mode
        for module in self.children():
            if isinstance(module, BinaryNN):
                module.binary(mode, *args, **kwargs)
        return self

    def forward_save(self, x):
        return {'output': self.forward(x)}

    def numel_binary(self):
        res = self._specific_numel_binary()
        for module in self.children():
            if isinstance(module, BinaryNN):
                res += module.numel_binary()
        return res

    def numel_float(self):
        return sum([param.numel() for param in self.parameters() if param.requires_grad])

    def _specific_numel_binary(self):
        """Specifies the number of binarizable parameters that are not contained in the children."""
        return 0

    @property
    def device(self):
        for param in self.parameters():
            return param.device


class BinarySequential(nn.Sequential, BinaryNN):
    def __init__(self, *args, binary_mode: bool = False, **kwargs):
        nn.Sequential.__init__(self, *args, **kwargs)
        self.binary_mode = binary_mode

    def __gititem__(self, idx):
        res = super().__getitem__(idx)
        res.binary_mode = self.binary_mode
        return res
