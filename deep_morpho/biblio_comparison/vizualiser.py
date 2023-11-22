import numpy as np

from general.nn.viz import ElementImage, ElementArrow, ElementGrouper
from ..viz.element_generator import EltGenerator
from ..viz.skeleton_morp_viz import SkeletonMorpViz
from ..viz.elt_generator_init import EltGeneratorInitCircle
from ..viz.elt_generator_bimonn import EltGeneratorConnectLuiBise
from ..viz.element_lui import ElementLuiCoefs


class EltGeneratorLayerWeights(EltGenerator):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def generate(self, layer_idx, chin, chout, xy_coords_mean, height, **kwargs):
        layer = self.model.layers[layer_idx]
        return ElementImage(layer.weight.cpu().detach().numpy(), xy_coords_mean=xy_coords_mean, size=height, **kwargs)


class EltGeneratorSequentialLui(EltGenerator):
    def generate(self, layer_idx, chout, xy_coords_mean, height, **kwargs):
        return ElementGrouper(elements={
            'coefs': ElementLuiCoefs(model=None, shape=np.array([height, height]), xy_coords_mean=xy_coords_mean, **kwargs)
        })


class EltGeneratorConnectSequential(EltGenerator):
    def generate(self, group, layer_idx, chout, chin):
        bise_elt = group[f"bise_layer_{layer_idx}_chout_{chout}_chin_{chin}"]
        lui_elt = group[f"lui_layer_{layer_idx}_chout_{chout}"]["coefs"]

        width = 1

        return ElementArrow.link_elements(bise_elt, lui_elt, width=width)


class SequentialWeightVizualiser(SkeletonMorpViz):

    def __init__(self, model, **kwargs):
        self.model = model


        kwargs.update({
            "elt_generator_bise": EltGeneratorLayerWeights(model),
            "elt_generator_lui": EltGeneratorSequentialLui(),
            "elt_generator_connections": EltGeneratorConnectSequential(),
        })


        super().__init__(
            in_channels=[1 for _ in range(len(model))], out_channels=[1 for _ in range(len(model))], **kwargs
            # in_channels=model.in_channels, out_channels=model.out_channels, **kwargs
        )
        self.elt_generator_init = EltGeneratorInitCircle()

    @property
    def max_selem_shape(self):
        return np.array(self.model.kernel_size).max(1)
