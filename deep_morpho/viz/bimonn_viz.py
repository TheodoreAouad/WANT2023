from typing import Dict

import numpy as np

from .skeleton_morp_viz import SkeletonMorpViz
from .elt_generator_bimonn import (
    EltGeneratorBise, EltGeneratorLui, EltGeneratorConnectLuiBise, EltGeneratorBiseBinary,
    EltGeneratorConnectLuiBiseClosest
)
from .elt_generator_bimonn_forward_save import (
    EltGeneratorBiseForwardSave, EltGeneratorLuiForwardSave, EltGeneratorConnectLuiBiseForwardSave,
    EltGeneratorInitForwardSave, EltGeneratorConnectLuiBiseClosestForwardSave
)
from .elt_generator_bimonn_histogram import (
    EltGeneratorBiseHistogram, EltGeneratorLuiHistogram, EltGeneratorInitHistogram
)
from .elt_generator_init import EltGeneratorInitCircle


class BimonnVizualiser(SkeletonMorpViz):

    def __init__(self, model, mode: str = "weights", update_binaries: bool = False, **kwargs):
        self.model = model
        assert mode in ["weights", "learned", "closest"]

        if mode == "weights":
            kwargs.update({
                "elt_generator_bise": EltGeneratorBise(model),
                "elt_generator_lui": EltGeneratorLui(model),
                "elt_generator_connections": EltGeneratorConnectLuiBise(model=model, binary_mode=False),
            })

        elif mode == "learned":
            kwargs.update({
                "elt_generator_bise": EltGeneratorBiseBinary(model, learned=True, update_binaries=update_binaries),
                "elt_generator_lui": EltGeneratorLui(model, update_binaries=update_binaries),
                "elt_generator_connections": EltGeneratorConnectLuiBise(model=model, binary_mode=True),
            })

        elif mode == "closest":
            kwargs.update({
                "elt_generator_bise": EltGeneratorBiseBinary(model, learned=False, update_binaries=update_binaries),
                "elt_generator_lui": EltGeneratorLui(model, learned=False, update_binaries=update_binaries),
                "elt_generator_connections": EltGeneratorConnectLuiBiseClosest(model=model),
            })

        super().__init__(
            in_channels=model.in_channels, out_channels=model.out_channels, **kwargs
        )
        self.elt_generator_init = EltGeneratorInitCircle()

    @property
    def max_selem_shape(self):
        return np.array(self.model.kernel_size).max(1)


class BimonnForwardVizualiser(SkeletonMorpViz):

    def __init__(self, model, inpt, mode: str = "float", lui_horizontal_factor: float = 1.7, update_binaries: bool = False, **kwargs):
        self.model = model
        self.update_binaries = update_binaries

        assert mode in ["float", "binary"]
        self.mode = mode
        if mode == "float":
            self.model.binary(False)
            kwargs["elt_generator_connections"] = EltGeneratorConnectLuiBiseForwardSave(model=model)
        else:
            self.model.binary(True, update_binaries=update_binaries)
            kwargs["elt_generator_connections"] = EltGeneratorConnectLuiBiseClosestForwardSave(model=model)

        self.all_outputs = model.forward_save(inpt)

        kwargs.update({
            "elt_generator_bise": EltGeneratorBiseForwardSave(self.all_outputs),
            "elt_generator_lui": EltGeneratorLuiForwardSave(self.all_outputs),
            "elt_generator_init": EltGeneratorInitForwardSave(self.all_outputs)
        })

        super().__init__(in_channels=model.in_channels, out_channels=model.out_channels, lui_horizontal_factor=lui_horizontal_factor, **kwargs)

    @property
    def input(self):
        return self.all_outputs["input"]

    @property
    def max_selem_shape(self):
        return [self.input.shape[-1] for _ in range(len(self.model))]


class BimonnHistogramVizualiser(SkeletonMorpViz):

    def __init__(
        self, model, inpt, dpi=100, mode: str = "float", lui_horizontal_factor: float = 1.7,
        hist_kwargs: Dict = {}, bise_hist_kwargs: Dict = None, lui_hist_kwargs: Dict = None, init_hist_kwargs: Dict = None,
        update_binaries: bool = False,
        **kwargs):
        self.model = model
        self.dpi = dpi
        self.update_binaries = update_binaries

        assert mode in ["float", "binary"]
        self.mode = mode
        if mode == "float":
            self.model.binary(False)
            kwargs["elt_generator_connections"] = EltGeneratorConnectLuiBiseForwardSave(model=model)
        else:
            self.model.binary(True, update_binaries=update_binaries)
            kwargs["elt_generator_connections"] = EltGeneratorConnectLuiBiseClosestForwardSave(model=model)

        self.all_outputs = model.forward_save(inpt)

        self.hist_kwargs = hist_kwargs
        self.bise_hist_kwargs = bise_hist_kwargs if bise_hist_kwargs is not None else hist_kwargs
        self.lui_hist_kwargs = lui_hist_kwargs if lui_hist_kwargs is not None else hist_kwargs
        self.init_hist_kwargs = init_hist_kwargs if init_hist_kwargs is not None else hist_kwargs

        if model.atomic_element[0] == "sybisel":
            range_hist = (-1, 1)
        elif model.atomic_element[0] == "bisel":
            range_hist = (0, 1)

        bins = 10
        self.bise_hist_kwargs["range"] = self.bise_hist_kwargs.get("range", range_hist)
        self.lui_hist_kwargs["range"] = self.lui_hist_kwargs.get("range", range_hist)
        self.init_hist_kwargs["range"] = self.init_hist_kwargs.get("range", range_hist)

        self.bise_hist_kwargs["bins"] = self.bise_hist_kwargs.get("bins", bins)
        self.lui_hist_kwargs["bins"] = self.lui_hist_kwargs.get("bins", bins)
        self.init_hist_kwargs["bins"] = self.init_hist_kwargs.get("bins", bins)

        kwargs.update({
            "elt_generator_bise": EltGeneratorBiseHistogram(self.all_outputs, dpi=dpi, hist_kwargs=self.bise_hist_kwargs),
            "elt_generator_lui": EltGeneratorLuiHistogram(self.all_outputs, dpi=dpi, hist_kwargs=self.lui_hist_kwargs),
            "elt_generator_init": EltGeneratorInitHistogram(self.all_outputs, dpi=dpi, hist_kwargs=self.init_hist_kwargs),
        })

        super().__init__(
            in_channels=model.in_channels, out_channels=model.out_channels, lui_horizontal_factor=lui_horizontal_factor, **kwargs
        )

    @property
    def input(self):
        return self.all_outputs["input"]

    @property
    def max_selem_shape(self):
        return [max(EltGeneratorBiseHistogram.default_figsize * self.dpi) for _ in range(len(self.model))]
