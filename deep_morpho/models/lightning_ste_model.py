from .generic_lightning_model import GenericLightningModel
from .ste_model import BNNConv


class LightningSTEConv(GenericLightningModel):
    pass


class LightningBNNConv(LightningSTEConv):
    model_class = BNNConv
