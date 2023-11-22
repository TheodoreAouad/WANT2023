import numpy as np

from general.nn.viz import ElementGrouper, ElementImage, ElementSymbolDilation, ElementSymbolErosion, ElementCircle
from ..models import BiSE


MAX_WIDTH_COEF = 1
BISE_INVERT_CODE = {v: k for (k, v) in BiSE.operation_code.items()}


class ElementBiseWeightsChan(ElementGrouper):
    operation_elements_dict = {'dilation': ElementSymbolDilation, "erosion": ElementSymbolErosion}

    def __init__(self, model, chout=0, size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.chout = chout
        self.size = size if size is not None else self.model.weight.shape[-2:]

        weights = self.model.weight[self.chout, 0].detach().cpu().numpy()
        if -self.model.bias[self.chout] <= weights.sum() / 2:
            operation = 'dilation'
        else:
            operation = 'erosion'

        self.weights_element = ElementImage(weights, imshow_kwargs={"interpolation": "nearest", "vmin": 0}, size=self.size, **kwargs)

        radius_operation = max(2, self.size / 4)
        self.operation_element = self.operation_elements_dict[operation](
            radius=radius_operation, xy_coords_mean=self.weights_element.xy_coords_mean + np.array([0, self.size / 2 + radius_operation / 2])
        )
        self.add_element(self.operation_element, key="operation")
        self.add_element(self.weights_element, key="weights")




class ElementBiseSelemChan(ElementGrouper):
    operation_elements_dict = {'dilation': ElementSymbolDilation, "erosion": ElementSymbolErosion}

    def __init__(self, model, chout=0, learned=True, v1=0, v2=1, size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.chout = chout
        self.learned = learned
        self.v1 = v1
        self.v2 = v2
        # self.kernel_shape = self.model.weight.shape[-2:]
        self.size = size if size is not None else self.model.weight.shape[-2:]

        if self.learned and not model.is_activated[chout]:
            # self.selem_element = ElementCircle(radius=self.kernel_shape[-1] / 2, **kwargs)
            self.selem_element = ElementCircle(radius=self.size / 2, **kwargs)
            self.operation_element = None
        else:
            if self.learned:
                # selem = model.learned_selem[..., chout]
                selem = model.learned_selem[chout, 0]
                operation = BISE_INVERT_CODE[model.learned_operation[chout]]
            else:
                # selem = model.closest_selem[..., chout]
                selem = model.closest_selem[chout, 0]
                operation = BISE_INVERT_CODE[model.closest_operation[chout]]

            radius_operation = max(2, self.size / 4)
            # radius_operation = max(2, self.kernel_shape[-1] / 4)

            self.selem_element = ElementImage(selem, imshow_kwargs={"interpolation": "nearest", "vmin": 0, "vmax": 1, "cmap": "gray"}, size=self.size, **kwargs)
            self.operation_element = self.operation_elements_dict[operation](
                radius=radius_operation, xy_coords_mean=self.selem_element.xy_coords_mean + np.array([0, self.size / 2 + radius_operation / 2])
                # radius=radius_operation, xy_coords_mean=self.selem_element.xy_coords_mean + np.array([0, self.kernel_shape[-1] / 2 + radius_operation / 2])
            )
            self.add_element(self.operation_element, key="operation")

        self.add_element(self.selem_element, key="selem")
