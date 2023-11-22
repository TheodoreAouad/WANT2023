from general.nn.loss import DiceLoss

from .masked_loss import LossMaskedBorder
from .diceloss_normalized import NormalizedDiceLoss


class MaskedDiceLoss(LossMaskedBorder):

    def __init__(self, border, *args, **kwargs):
        super().__init__(loss=DiceLoss(*args, **kwargs), border=border)


class MaskedNormalizedDiceLoss(LossMaskedBorder):

    def __init__(self, border, *args, **kwargs):
        super().__init__(loss=NormalizedDiceLoss(*args, **kwargs), border=border)
