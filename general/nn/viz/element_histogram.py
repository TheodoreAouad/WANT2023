import numpy as np

from .element_image import ElementImage
from .plot_histogram import get_hist_as_array


class ElementHistogram(ElementImage):
    default_figsize = np.array([4, 6])

    def __init__(self, data, *args, dpi=128, figure_kwargs={}, hist_kwargs={}, **kwargs):
        self.data = data
        self.dpi = dpi
        self.figure_kwargs = figure_kwargs
        self.hist_kwargs = hist_kwargs

        self.hist_array = get_hist_as_array(data, dpi=dpi, figure_kwargs=figure_kwargs, **hist_kwargs)
        super().__init__(image=self.hist_array, *args, **kwargs)
