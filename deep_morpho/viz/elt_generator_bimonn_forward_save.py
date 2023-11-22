from general.nn.viz import ElementArrow, ElementImage
from .element_generator import EltGenerator



class EltGeneratorBiseForwardSave(EltGenerator):

    def __init__(self, all_outputs):
        super().__init__()
        self.all_outputs = all_outputs

    def generate(self, layer_idx, chin, chout, xy_coords_mean, height, **kwargs):
        img = self.all_outputs[layer_idx][chin, chout][0].cpu().detach().numpy()
        return ElementImage(img, imshow_kwargs={"cmap": "gray", "vmin": 0, "vmax": 1}, xy_coords_mean=xy_coords_mean, size=height, **kwargs)


class EltGeneratorLuiForwardSave(EltGenerator):
    def __init__(self, all_outputs):
        super().__init__()
        self.all_outputs = all_outputs

    def generate(self, layer_idx, chout, xy_coords_mean, height, **kwargs):
        img = self.all_outputs[layer_idx][chout][0].cpu().detach().numpy()
        return ElementImage(img, imshow_kwargs={"cmap": "gray", "vmin": 0, "vmax": 1}, xy_coords_mean=xy_coords_mean, size=height, **kwargs)


class EltGeneratorConnectLuiBiseForwardSaveBase(EltGenerator):

    def __init__(self, model, max_width_coef=1):
        super().__init__()
        self.model = model
        self.max_width_coef = max_width_coef

    def generate(self, group, layer_idx, chout, chin):
        bise_elt = group[f"bise_layer_{layer_idx}_chout_{chout}_chin_{chin}"]
        lui_elt = group[f"lui_layer_{layer_idx}_chout_{chout}"]

        # width = self.infer_width(lui_elt, chin)

        # coefs = self.model.layers[layer_idx].luis[chout].positive_weight[0].detach().cpu().numpy()
        # coefs = coefs / coefs.max() * self.max_width_coef
        # width = coefs[chin]
        width = self.infer_width(self.model.layers[layer_idx].luis, chin=chin, chout=chout)

        # activation_P = bise_elt.model.activation_P[chout]
        # if activation_P > 0 or width == 0:
        return ElementArrow.link_elements(bise_elt, lui_elt, width=width)
        # return ElementArrowNo.link_elements(bise_elt, lui_elt, height_circle=max(self.model.kernel_size[layer_idx])*0.7, width=width)

    def infer_width(self, model, chin, chout):
        raise NotImplementedError


class EltGeneratorConnectLuiBiseForwardSave(EltGeneratorConnectLuiBiseForwardSaveBase):

    def infer_width(self, model, chin, chout):
        coefs = model.coefs[chout].detach().cpu().numpy()
        coefs = coefs / coefs.max() * self.max_width_coef
        return coefs[chin]


class EltGeneratorConnectLuiBiseClosestForwardSave(EltGeneratorConnectLuiBiseForwardSaveBase):

    def infer_width(self, model, chin, chout):
        return float(model.closest_set[chout, chin])



class EltGeneratorInitForwardSave(EltGenerator):

    def __init__(self, all_outputs):
        super().__init__()
        self.all_outputs = all_outputs

    def generate(self, chan, xy_coords_mean, height, **kwargs):
        img = self.all_outputs["input"][0, chan].cpu().detach().numpy()
        return ElementImage(img, imshow_kwargs={"cmap": "gray", "vmin": 0, "vmax": 1}, xy_coords_mean=xy_coords_mean, size=height, **kwargs)
