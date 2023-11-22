from typing import Tuple, Union

import numpy as np
from matplotlib.patches import Rectangle

from .element import Element


class ElementImage(Element):

    def __init__(self, image: np.ndarray, borders=True, size=None, imshow_kwargs={}, *args, **kwargs):
        self._image = image
        self._size = size if size is not None else self.image.shape[:2]
        if isinstance(self._size, (float, int)):
            self._size = (self._size, self._size)
        super().__init__(*args, **kwargs)
        self.borders = borders
        self.imshow_kwargs = imshow_kwargs


    @property
    def shape(self):
        return np.array(self._size)
        # return np.array(self.image.shape)[:2]

    @property
    def image(self):
        return self._image

    @property
    def img(self):
        return self.image

    def add_to_canva(self, canva: "Canva", coords=None, coords_type="barycentre", imshow_kwargs=None):
        if imshow_kwargs is None:
            imshow_kwargs = self.imshow_kwargs

        if coords is not None:
            if coords_type == "barycentre":
                self.set_xy_coords_mean(coords)
            elif coords_type == "botleft":
                self.set_xy_coords_botleft(coords)

        canva.ax.imshow(self.image, extent=(
            self.xy_coords_botleft[0], self.xy_coords_botleft[0] + self.shape[0],
            self.xy_coords_botleft[1], self.xy_coords_botleft[1] + self.shape[1],
        ), **imshow_kwargs)

        if self.borders:
            canva.ax.add_patch(Rectangle(self.xy_coords_botleft, self.shape[0], self.shape[1], color='k', fill=False))
