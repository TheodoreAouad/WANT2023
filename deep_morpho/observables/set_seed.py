import os

import torch
import numpy as np
import random

from deep_morpho.utils import set_seed
from general.nn.observables import Observable


class SetSeed(Observable):
    def __init__(self, seed=None, *args, **kwargs):
        super().__init__()
        self.seed = seed
        # os.environ["PL_GLOBAL_SEED"] = str(seed)

    def on_train_start(self, *args, **kwargs):
        self.seed = set_seed(self.seed)


class RandomObservable(Observable):
    def on_batch_start(self, trainer, *args, **kwargs):
        self.freq_idx += 1
        if self.freq_idx % self.freq == 0:
            trainer.logger.experiment.add_scalar('random/numpy', np.random.rand(), trainer.global_step)
            trainer.logger.experiment.add_scalar('random/random', random.random(), trainer.global_step)
            trainer.logger.experiment.add_scalar('random/torch', torch.rand(1), trainer.global_step)


class CheckSeed(Observable):
    def on_batch_start(self, trainer, *args, **kwargs):
        trainer.logger.experiment.add_scalar('random/numpy', np.random.rand(), trainer.global_step)
        trainer.logger.experiment.add_scalar('random/random', random.random(), trainer.global_step)
        trainer.logger.experiment.add_scalar('random/torch', torch.rand(1), trainer.global_step)
