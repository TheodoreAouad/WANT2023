from general.nn.viz import ElementArrow, ElementNO, ElementGrouper, Element


class ElementArrowNo(ElementGrouper):

    def __init__(self, arrow_element, no_element):
        super().__init__()

        self.arrow_element = arrow_element
        self.no_element = no_element

        self.add_element(self.arrow_element, key="arrow")
        self.add_element(self.no_element, key="no")


    @staticmethod
    def link_elements(
        elt1: Element,
        elt2: Element,
        key=None,
        height_circle=None,
        height_circle_rate=.15,
        link1="adapt",
        link2="adapt",
        length_includes_head=True, width=.1, **kwargs
    ):
        arrow_elt = ElementArrow.link_elements(
            elt1, elt2, key=key, link1=link1, link2=link2, 
            length_includes_head=length_includes_head, width=width, **kwargs
        )
        if height_circle is None:
            height_circle = height_circle_rate * max(arrow_elt.shape)
        xy_coords_mean = arrow_elt.barycentre
        no_elt = ElementNO(xy_coords_mean=xy_coords_mean, width=height_circle)

        return ElementArrowNo(arrow_elt, no_elt)
