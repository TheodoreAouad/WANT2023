from general.nn.viz import ElementArrow, ElementImage
from .element_generator import EltGenerator
from .element_lui import ElementLuiSum


class EltGeneratorBiseSteConv(EltGenerator):

    def __init__(self, model, mode="binary"):
        super().__init__()
        self.model = model
        self.mode = mode

    def generate(self, layer_idx, chin, chout, xy_coords_mean, height, **kwargs):
        conv_layer = self.model.layers[layer_idx]
        conv_layer.clip_weights()

        if self.mode == "weight":
            weight = conv_layer.weight[chout, chin].detach().cpu().numpy()
            return ElementImage(weight, xy_coords_mean=xy_coords_mean, size=height, imshow_kwargs={"cmap": "coolwarm", "vmin": -1, "vmax": 1}, **kwargs)

        elif self.mode == "binary":
            weight = conv_layer.binarized_weight[chout, chin].detach().cpu().numpy()
            return ElementImage(weight, xy_coords_mean=xy_coords_mean, size=height, imshow_kwargs={"cmap": "gray", "vmin": -1, "vmax": 1}, **kwargs)


class EltGeneratorLuiSteConv(EltGenerator):
    def __init__(self, **imshow_kwargs):
        super().__init__()
        self.imshow_kwargs = imshow_kwargs

    def generate(self, layer_idx, chout, xy_coords_mean, height):
        return ElementLuiSum(xy_coords_mean=xy_coords_mean, radius=height/2, imshow_kwargs=self.imshow_kwargs)


class EltGeneratorArrowSteConv(EltGenerator):
    def generate(self, group, layer_idx, chout, chin):
        bise_elt = group[f"bise_layer_{layer_idx}_chout_{chout}_chin_{chin}"]
        lui_elt = group[f"lui_layer_{layer_idx}_chout_{chout}"]

        return ElementArrow.link_elements(bise_elt, lui_elt)
