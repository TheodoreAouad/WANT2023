import numpy as np

from .skeleton_morp_viz import SkeletonMorpViz
from .elt_generator_ste_conv import (
    EltGeneratorBiseSteConv, EltGeneratorLuiSteConv, EltGeneratorArrowSteConv
)
from .elt_generator_init import EltGeneratorInitCircle


class SteConvVizualiser(SkeletonMorpViz):

    def __init__(self, model, mode: str = "weight", **kwargs):
        self.model = model

        if mode.lower() in ["bin", "binarized", "binary", "b"]:
            mode = "binary"

        if mode.lower() in ["weights", "weight", "w"]:
            mode = "weight"

        assert mode in ["weight", "binary", ]

        kwargs.update({
            "elt_generator_bise": EltGeneratorBiseSteConv(model, mode=mode),
            "elt_generator_lui": EltGeneratorLuiSteConv(),
            "elt_generator_connections": EltGeneratorArrowSteConv(),
        })

        super().__init__(
            in_channels=model.in_channels, out_channels=model.out_channels, **kwargs
        )
        self.elt_generator_init = EltGeneratorInitCircle()

    @property
    def max_selem_shape(self):
        return np.array(self.model.kernel_size).max()
