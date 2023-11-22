from .element_generator import EltGenerator
from general.nn.viz import ElementCircle


class EltGeneratorInitCircle(EltGenerator):

    def __init__(self, input_radius_factor=1, imshow_kwargs={"fill": True}):
        super().__init__()
        self.input_radius_factor = input_radius_factor
        self.imshow_kwargs = imshow_kwargs


    def generate(self, chan, xy_coords_mean, height):
        return ElementCircle(
            xy_coords_mean=xy_coords_mean,
            # radius=self.input_radius_factor * self.box_height / (2 * n_elts),
            radius=height / 2 * self.input_radius_factor,
            imshow_kwargs={"fill": True},
        )
