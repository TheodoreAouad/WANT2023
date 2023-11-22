import pathlib
from os.path import join

import torch.optim as optim
from pytorch_lightning.callbacks import Callback

from general.utils import save_json


class CheckLearningRate(Callback):

    def __init__(self, freq=1) -> None:
        super().__init__()
        self.freq = freq
        self.freq_idx = 0
        self.last_lr = None

    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
        if self.freq_idx % self.freq == 0:
            cur_lr = self.get_lr_from_optimizer(pl_module.optimizers())
            trainer.logger.experiment.add_scalar('learning_rate', cur_lr, trainer.global_step)
            self.last_lr = cur_lr

        self.freq_idx += 1

    def get_lr_from_optimizer(self, optimizer):
        if isinstance(optimizer, (optim.Adam, optim.SGD)):
            return optimizer.param_groups[0]['lr']
        
        raise NotImplementedError("optimizer not recognized.")

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        save_json({"lr": self.last_lr}, join(final_dir, "lr.json"))
