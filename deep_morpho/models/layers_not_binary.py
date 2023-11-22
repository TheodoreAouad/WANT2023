from typing import Union, Tuple, Dict

import numpy as np

from .lui import LUI, SyLUI
from .dense_lui import DenseLUI
from .bise import BiSE, SyBiSE
from ..initializer.bisel_initializer import BiselInitializer, BiselInitIdentical
from ..initializer.bise_initializer import InitBiseHeuristicWeights
from .bisel import BiSELBase


class NotBinaryNN:
    def forward(self, x):
        return self._forward(x)

    def numel_binary(self):
        return 0

    def binary(self, *args, **kwargs):
        return self

    @property
    def closest_selem_dist(self):
        return np.array([])


class LuiNotBinary(NotBinaryNN, LUI):
    ...


class SyLuiNotBinary(NotBinaryNN, SyLUI):
    ...


class DenseLuiNotBinary(NotBinaryNN, DenseLUI):
    ...



# class LuiNotBinary(LUI):
#     def forward(self, x):
#         return self._forward(x)

#     def numel_binary(self):
#         return 0

#     def binary(self, *args, **kwargs):
#         return self

#     def closest_selem_dist(self):
#         return np.array([])


# class SyLuiNotBinary(SyLUI):
#     def forward(self, x):
#         return self._forward(x)

#     def numel_binary(self):
#         return 0

#     def binary(self, *args, **kwargs):
#         return self


# class DenseLuiNotBinary(DenseLUI):
#     def forward(self, x):
#         return self._forward(x)

#     def numel_binary(self):
#         return 0

#     def binary(self, *args, **kwargs):
#         return self


class BiSELNotBinary(BiSELBase):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(
            bise_module=BiSE,
            lui_module=LuiNotBinary,
            *args,
            **kwargs
        )


class SyBiSELNotBinary(BiSELBase):
    def __init__(
        self,
        *args,
        **bise_kwargs
    ):
        super().__init__(
            bise_module=SyBiSE,
            lui_module=SyLuiNotBinary,
            *args,
            **bise_kwargs
        )
