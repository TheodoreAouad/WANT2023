from .skeleton_morp_viz import SkeletonMorpViz
from .elt_generator_morpop import EltGeneratorAggregation, EltGeneratorErodila, EltGeneratorConnectorMorpOp
from .elt_generator_init import EltGeneratorInitCircle


class MorpOperationsVizualiser(SkeletonMorpViz):

    def __init__(self, model, binary_mode=False, **kwargs):
        self.model = model
        super().__init__(
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            elt_generator_init=None,
            elt_generator_bise=EltGeneratorErodila(model),
            elt_generator_lui=EltGeneratorAggregation(model),  # binary true because it does not hide any info for LUI to be binary
            elt_generator_connections=EltGeneratorConnectorMorpOp(model),
            **kwargs
        )
        self.elt_generator_init = EltGeneratorInitCircle(
            # radius=self.box_height / (2 * model.in_channels[0]),
        )

    @property
    def max_selem_shape(self):
        return self.model.max_selem_shape
