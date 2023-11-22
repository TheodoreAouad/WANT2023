from .lightning_bimonn import LightningBiMoNN
from .bimonn_identity import BimonnIdentity


class LightningBimonnIdentity(LightningBiMoNN):
    model_class = BimonnIdentity
