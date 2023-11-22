import numpy as np

from .element_bise import ElementBiseSelemChan, ElementBiseWeightsChan
from general.nn.viz import ElementArrow
from .element_arrow_no import ElementArrowNo
from .element_generator import EltGenerator
from .element_lui import ElementLui, ElementLuiClosest



class EltGeneratorBise(EltGenerator):

    def __init__(self, bimonn_model):
        super().__init__()
        self.bimonn_model = bimonn_model

    def generate(self, layer_idx, chin, chout, xy_coords_mean, height, **kwargs):
        bisel_layer = self.bimonn_model.layers[layer_idx]
        return ElementBiseWeightsChan(
            model=bisel_layer.bises, chout=bisel_layer.convert_chin_chout_bise_chan(chin=chin, chout=chout),
            xy_coords_mean=xy_coords_mean, size=height, **kwargs
        )


class EltGeneratorBiseBinary(EltGenerator):
    """ Generator for BiSE element binary. Choose either learned = True for learned or learned = False
    for closest selem.
    """
    def __init__(self, bimonn_model, learned: bool = True, update_binaries: bool = False):
        super().__init__()
        self.update_binaries = update_binaries
        self.bimonn_model = bimonn_model
        self.learned = learned

    def generate(self, layer_idx, chin, chout, xy_coords_mean, height, **kwargs):
        bisel_layer = self.bimonn_model.layers[layer_idx]
        chout = bisel_layer.convert_chin_chout_bise_chan(chin=chin, chout=chout)
        if self.update_binaries:
            if self.learned:
                bisel_layer.bises.update_learned_selems(chans=[chout])
            else:
                bisel_layer.bises.update_closest_selems(chans=[chout])

        return ElementBiseSelemChan(model=bisel_layer.bises, learned=self.learned, chout=chout, xy_coords_mean=xy_coords_mean, size=height, **kwargs)



class EltGeneratorLui(EltGenerator):

    def __init__(self, bimonn_model, learned: bool = True, imshow_kwargs={"color": "k"}, update_binaries: bool = False):
        super().__init__()
        self.bimonn_model = bimonn_model
        self.imshow_kwargs = imshow_kwargs
        self.learned = learned
        self.update_binaries = update_binaries

    def generate(self, layer_idx, chout, xy_coords_mean, height):
        lui_layer = self.bimonn_model.layers[layer_idx].luis
        # lui_layer.update_learned_sets()

        if self.update_binaries:
            lui_layer.update_binary_sets(chans=[chout])

        if self.learned:
            constructor = ElementLui
        else:
            constructor = ElementLuiClosest

        return constructor(
            lui_layer,
            chout=chout,
            xy_coords_mean=xy_coords_mean,
            shape=np.array([height, height]),
            imshow_kwargs=self.imshow_kwargs,
        )


class EltGeneratorConnectLuiBiseBase(EltGenerator):

    def __init__(self, model, max_width_coef=.3):
        super().__init__()
        self.model = model
        self.max_width_coef = max_width_coef

    def generate(self, group, layer_idx, chout, chin):
        bisel_layer = self.model.layers[layer_idx]
        bise_elt = group[f"bise_layer_{layer_idx}_chout_{chout}_chin_{chin}"]
        lui_elt = group[f"lui_layer_{layer_idx}_chout_{chout}"]["coefs"]

        chout_bise = bisel_layer.convert_chin_chout_bise_chan(chin=chin, chout=chout)

        width = self.infer_width(lui_elt, chin=chin, chout=chout,) * self.max_width_coef

        activation_P = bise_elt.model.activation_P[chout_bise]
        if activation_P > 0 or width == 0:
            return ElementArrow.link_elements(bise_elt, lui_elt, width=width)
        return ElementArrowNo.link_elements(bise_elt, lui_elt, width=width)

    def infer_width(self, lui_elt, chin, chout):
        raise NotImplementedError


class EltGeneratorConnectLuiBise(EltGeneratorConnectLuiBiseBase):

    def __init__(self, binary_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_mode = binary_mode

    def infer_width(self, lui_elt, chin, chout):
        model= lui_elt.model

        if self.binary_mode and model.is_activated[chout]:
            return float(model.learned_set[chout, chin])
            # return float(model.learned_set[0, chin])


        # coefs = model.coefs[0].detach().cpu().numpy()
        coefs = model.coefs[chout].detach().cpu().numpy()
        coefs = coefs / coefs.max()
        return coefs[chin]


class EltGeneratorConnectLuiBiseClosest(EltGeneratorConnectLuiBiseBase):

    def infer_width(self, lui_elt, chin, chout):
        model = lui_elt.model
        # return float(model.closest_set[0, chin])
        return float(model.closest_set[chout, chin])
