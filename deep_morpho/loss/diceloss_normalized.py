from general.nn.loss import DiceLoss


class NormalizedDiceLoss(DiceLoss):
    """ Normalizes the input to be between 0 and 1. Must give the vmin and vmax of the input.
    """

    def __init__(self, vmin: float = 0, vmax: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, ypred, ytrue):
        ypred = (ypred - self.vmin) / (self.vmax - self.vmin)
        ytrue = (ytrue - self.vmin) / (self.vmax - self.vmin)
        return super().forward(ypred, ytrue)
