from torch.nn import BCELoss

from .masked_loss import LossMaskedBorder
from .bce_normalized import BCENormalizedLoss



class MaskedBCELoss(LossMaskedBorder):

    def __init__(self, border, *args, **kwargs):
        super().__init__(loss=BCELoss(*args, **kwargs), border=border)


class MaskedBCENormalizedLoss(LossMaskedBorder):

    def __init__(self, border, *args, **kwargs):
        super().__init__(loss=BCENormalizedLoss(*args, **kwargs), border=border)
