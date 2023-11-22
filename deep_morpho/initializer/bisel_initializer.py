from typing import Dict, List, Tuple
from copy import deepcopy

from .bise_initializer import BiseInitializer


class BiselInitializer:
    """The Bisel initializer works in two steps. First, it generates one initializer for bise and one for lui. Then, it proposes
    another step of post initialization, to rework the initialization (ex: in case we need to have dependent init between lui and bise).
    """
    def __init__(
        self,
        bise_initializer: BiseInitializer = BiseInitializer(),
        lui_initializer: BiseInitializer = BiseInitializer(),
        *args, **kwargs
    ):
        self.bise_initializer = bise_initializer
        self.lui_initializer = lui_initializer

    def get_bise_initializers(self, module) -> Tuple[BiseInitializer, Dict]:
        return deepcopy(self.bise_initializer)

    def get_lui_initializers(self, module) -> Tuple[BiseInitializer, Dict]:
        return deepcopy(self.lui_initializer)

    def post_initialize(self, module):
        pass


class BiselInitIdentical(BiselInitializer):
    def __init__(self, initializer: BiseInitializer, *args, **kwargs):
        super().__init__(
            bise_initializer=initializer,
            lui_initializer=initializer,
        )
