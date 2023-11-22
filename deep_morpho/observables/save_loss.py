from os.path import join
import pathlib
from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl

from general.utils import save_json
from general.nn.observables import Observable


class SaveLoss(Observable):

    def __init__(self, freq: int = 100, *args, **kwargs):
        super().__init__(freq=freq, *args, **kwargs)
        self.last_loss = None

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "SdTEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        self.freq_idx += 1
        if self.freq_idx % self.freq == 0:
            loss_dict = {k: v for k, v in outputs.items() if 'loss' in k}
            loss_dict.update({
                f"{k}_grad": v.grad.item() for k, v in loss_dict.items() if v.grad is not None
            })
            trainer.logger.experiment.add_scalars(
                "loss/train", loss_dict, trainer.global_step
            )
            trainer.logged_metrics.update(
                **{f"loss/train/{k}": v for k, v in loss_dict.items()}
            )

            self.last_loss = loss_dict
        # for k, v in outputs.items():
        #     if 'loss' in k:
        #         trainer.log(f"loss/train/{k}", v)

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json({
            k: str(v) if isinstance(v, float) else str(v.item()) for (k, v) in self.last_loss.items()
        }, join(final_dir, "last_loss.json"))

        return self.last_loss
