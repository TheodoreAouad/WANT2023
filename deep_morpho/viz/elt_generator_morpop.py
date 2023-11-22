import numpy as np


from .element_generator import EltGenerator
from .element_arrow_no import ElementArrowNo
from general.nn.viz import (
    ElementGrouper, ElementImage, ElementSymbolDilation, ElementArrow,
    ElementSymbolErosion, ElementSymbolIntersection, ElementSymbolUnion
)


class EltGeneratorErodila(EltGenerator):

    def __init__(self, model, selem_imshow_kwargs={"cmap": "gray", "interpolation": "nearest"}, op_imshow_kwargs={"color": "k"}):
        super().__init__()
        self.model = model
        self.selem_imshow_kwargs = selem_imshow_kwargs
        self.op_imshow_kwargs = op_imshow_kwargs

    def generate(self, layer_idx, chin, chout, xy_coords_mean, height):
        selem = self.model.selems[layer_idx][chout][chin]
        op_name = self.model.operation_names[layer_idx][chout][chin]
        elt = ElementGrouper()

        elt.add_element(ElementImage(
            selem,
            xy_coords_mean=xy_coords_mean,
            imshow_kwargs=self.selem_imshow_kwargs,
        ), key="selem")

        if op_name == "dilation":
            constructor = ElementSymbolDilation
        if op_name == "erosion":
            constructor = ElementSymbolErosion

        elt.add_element(constructor(
            xy_coords_mean=xy_coords_mean + np.array([0, selem.shape[1] / 2 + 1]),
            radius=2,
            imshow_kwargs=self.op_imshow_kwargs
        ), key="operation")

        return elt




class EltGeneratorAggregation(EltGenerator):

    def __init__(self, model, imshow_kwargs={"color": "k"}):
        super().__init__()
        self.model = model
        self.imshow_kwargs = imshow_kwargs

    def generate(self, layer_idx, chout, xy_coords_mean, height):
        operation = self.model.operation_names[layer_idx][chout][-1]

        if operation == "union":
            constructor = ElementSymbolUnion
        if operation == "intersection":
            constructor = ElementSymbolIntersection

        return constructor(
            xy_coords_mean=xy_coords_mean,
            width=height / 2, height=height / 2,
            imshow_kwargs=self.imshow_kwargs
        )



class EltGeneratorConnectorMorpOp(EltGenerator):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def generate(self, group, layer_idx, chout, chin):
        selem_chans = self.model.selem_args[layer_idx][chout][-1]
        selem_chans = range(self.model.in_channels[layer_idx]) if selem_chans == "all" else selem_chans

        bise_elt = group[f"bise_layer_{layer_idx}_chout_{chout}_chin_{chin}"]["selem"]
        lui_elt = group[f"lui_layer_{layer_idx}_chout_{chout}"]

        if chin not in selem_chans:
            return None

        if self.model.do_complementation[layer_idx][chout][chin]:
            return ElementArrowNo.link_elements(bise_elt, lui_elt, height_circle=self.model.max_selem_shape[layer_idx]*0.7)


        return ElementArrow.link_elements(
            bise_elt,
            lui_elt
        )
