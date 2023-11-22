import numpy as np

from general.nn.viz import ElementHistogram
from .element_generator import EltGenerator


class EltGeneratorHistogramBase(EltGenerator):
    default_figsize = ElementHistogram.default_figsize

    def __init__(self, all_outputs, dpi=100, hist_kwargs={}):
        super().__init__()
        self.dpi = dpi
        self.all_outputs = all_outputs
        self.hist_kwargs = hist_kwargs

        # self.hist_kwargs['vmin'] = self.hist_kwargs.get("vmin", -1)
        # self.hist_kwargs['vmax'] = self.hist_kwargs.get("vmax", 1)
        self.hist_kwargs['range'] = self.hist_kwargs.get("range", (-1.05, 1.05))
        self.hist_kwargs["bins"] = self.hist_kwargs.get("bins", 20)
        self.hist_kwargs['density'] = self.hist_kwargs.get('density', True)
        self.hist_kwargs['stacked'] = self.hist_kwargs.get('stacked', False)


    def get_tensor_data(self, layer_idx=None, chin=None, chout=None, chan=None):
        raise NotImplementedError

    def generate(self, layer_idx=None, chin=None, chout=None, chan=None, xy_coords_mean=None, height=None, **kwargs):
        data = self.get_tensor_data(layer_idx, chin, chout, chan).cpu().detach().view(-1).numpy()
        return ElementHistogram(
            data,
            dpi=self.dpi, hist_kwargs=self.hist_kwargs,
            imshow_kwargs={"cmap": "gray", "vmin": 0, "vmax": 1}, xy_coords_mean=xy_coords_mean, size=height, borders=False,
            **kwargs
        )


class EltGeneratorBiseHistogram(EltGeneratorHistogramBase):
    def get_tensor_data(self, layer_idx=None, chin=None, chout=None, chan=None):
        return self.all_outputs[layer_idx][chin, chout]


class EltGeneratorLuiHistogram(EltGeneratorHistogramBase):
    def get_tensor_data(self, layer_idx=None, chin=None, chout=None, chan=None):
        return self.all_outputs[layer_idx][chout]


class EltGeneratorInitHistogram(EltGeneratorHistogramBase):
    def get_tensor_data(self, layer_idx=None, chin=None, chout=None, chan=None):
        return self.all_outputs["input"][:, chan]
