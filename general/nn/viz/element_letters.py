from .element import Element
from .plot_letters import plot_NO_on_ax


class ElementNO(Element):

    def __init__(self, xy_coords_mean, width=1, space_prop=.1, draw_circle=True, imshow_kwargs={}, **kwargs):
        super().__init__((width, width), xy_coords_mean=xy_coords_mean, **kwargs)
        self.width = width
        self.draw_circle = draw_circle
        self.space_prop = space_prop
        self.imshow_kwargs = imshow_kwargs
        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')

    def add_to_canva(self, canva: "Canva"):
        return plot_NO_on_ax(
            canva.ax, self.xy_coords_mean, width=self.width, draw_circle=self.draw_circle, space_prop=self.space_prop, **self.imshow_kwargs
        )
