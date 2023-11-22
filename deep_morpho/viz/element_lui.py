import numpy as np
from matplotlib.patches import Polygon

from general.nn.viz import Element, ElementGrouper, ElementSymbolIntersection, ElementSymbolUnion, ElementNO, ElementSymbolDilation
from ..models import LUI, BiSEBase


OPERATION_FACTOR = .3

LUI_INVERT_CODE = {v: k for (k, v) in BiSEBase.operation_code.items()}
# LUI_INVERT_CODE = {v: k for (k, v) in LUI.operation_code.items()}


class ElementLuiCoefs(Element):

    def __init__(self, model, imshow_kwargs={}, fill=True, fill_color='w', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.imshow_kwargs = imshow_kwargs
        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')
        self.imshow_kwargs['fill'] = False  # handle fill independantly
        self.fill = fill
        self.fill_color = fill_color

    def add_to_canva(self, canva: "Canva"):
        if self.fill:
            fill_kwargs = self.imshow_kwargs.copy()
            fill_kwargs['fill'] = True
            fill_kwargs['color'] = self.fill_color
            canva.ax.add_patch(Polygon(np.stack([
                self.xy_coords_botleft, self.xy_coords_topleft, self.xy_coords_midright
            ]), closed=True, **fill_kwargs))
        canva.ax.add_patch(Polygon(np.stack([
            self.xy_coords_botleft, self.xy_coords_topleft, self.xy_coords_midright
        ]), closed=True, **self.imshow_kwargs))


class ElementLui(ElementGrouper):
    operation_element_dicts = {'erosion': ElementSymbolIntersection, 'dilation': ElementSymbolUnion}

    def __init__(self, model, chout, shape, imshow_kwargs={}, v1=None, v2=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.v1 = v1
        self.v2 = v2
        self.chout = chout
        self.imshow_kwargs = imshow_kwargs
        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')

        self.element_lui_operation = None
        self.element_no = None

        self.element_lui_coefs = ElementLuiCoefs(model, imshow_kwargs, shape=shape, *args, **kwargs)
        self.add_element(self.element_lui_coefs, key="coefs")


        shape = self.element_lui_coefs.shape * OPERATION_FACTOR
        if self.model.is_activated[self.chout]:
            operation = LUI_INVERT_CODE[self.model.learned_operation[self.chout]]
            self.element_lui_operation = self.operation_element_dicts[operation](
                width=shape[0], height=shape[1],
                xy_coords_mean=(self.element_lui_coefs.xy_coords_mean +
                                np.array([-self.element_lui_coefs.shape[0] / 3, self.element_lui_coefs.shape[-1] / 2 + 2]))
            )
            self.add_element(self.element_lui_operation, key="operation")

        if self.model.activation_P[self.chout] < 0:
            self.element_no = ElementNO(
                width=shape[0],
                xy_coords_mean=(self.element_lui_coefs.xy_coords_mean +
                                np.array([self.element_lui_coefs.shape[0] / 3, self.element_lui_coefs.shape[-1] / 2 + 2]))
            )
            self.add_element(self.element_no, key="no")


class ElementLuiClosest(ElementGrouper):
    operation_element_dicts = {'erosion': ElementSymbolIntersection, 'dilation': ElementSymbolUnion}
    # operation_element_dicts = {'intersection': ElementSymbolIntersection, 'union': ElementSymbolUnion}

    def __init__(self, model, chout, shape, imshow_kwargs={}, v1=None, v2=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.v1 = v1
        self.v2 = v2
        self.chout = chout
        self.imshow_kwargs = imshow_kwargs
        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')

        self.element_lui_operation = None

        self.element_lui_coefs = ElementLuiCoefs(model, imshow_kwargs, shape=shape, *args, **kwargs)
        self.add_element(self.element_lui_coefs, key="coefs")

        operation = LUI_INVERT_CODE[self.model.closest_operation[self.chout]]
        shape = self.element_lui_coefs.shape * OPERATION_FACTOR
        self.element_lui_operation = self.operation_element_dicts[operation](
            width=shape[0], height=shape[1],
            xy_coords_mean=(self.element_lui_coefs.xy_coords_mean +
                            np.array([-self.element_lui_coefs.shape[0] / 3, self.element_lui_coefs.shape[-1] / 2 + 2]))
        )
        self.add_element(self.element_lui_operation, key="operation")

        if self.model.activation_P[self.chout] < 0:
            self.element_no = ElementNO(
                width=shape[0],
                xy_coords_mean=(self.element_lui_coefs.xy_coords_mean +
                                np.array([self.element_lui_coefs.shape[0] / 3, self.element_lui_coefs.shape[-1] / 2 + 2]))
            )
            self.add_element(self.element_no, key="no")


class ElementLuiSum(ElementSymbolDilation):
    pass
