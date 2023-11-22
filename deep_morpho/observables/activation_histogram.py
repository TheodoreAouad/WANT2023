from abc import ABC, abstractmethod
from typing import Any, Dict

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch

from .observable_layers import Observable


class ActivationHistogramBimonnBase(Observable, ABC):

    def __init__(self, *args, freq: Dict = {"train": 100, "val": 30}, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = {"train": 0, "val": 0,}
        self.first_batch = True

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def register_histogram(self, trainer, pl_module, batch, state, init_folder=""):
        handles = []
        step = trainer.global_step
        for idx, layer in enumerate(pl_module.model.layers):
            # Create function generator such that it does not give the same layer each time
            def get_hook(idx_layer, name, suffix):
                def forward_hook(module_, module_input, module_output):
                    trainer.logger.experiment.add_histogram(
                        f"activation_hist_{name}{suffix}/{state}/layer_{idx_layer}_{module_.__class__.__name__}",
                        module_output,
                        step
                    )
                return forward_hook

            handles.append(layer.register_forward_hook(get_hook(idx, self.name, suffix=init_folder)))

        with torch.no_grad():
            self.forward_hist(pl_module, batch[0].to(pl_module.model.device))
            # pl_module.model.forward()

        for handle in handles:
            handle.remove()

    @abstractmethod
    def forward_hist(self, pl_module, input_):
        pass


    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if self.first_batch:
            self.register_histogram(trainer=trainer, pl_module=pl_module, batch=batch, state="init", init_folder="_init")
        self.first_batch = False

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
    ):
        if self.freq_idx["train"] % self.freq["train"] != 0:
            self.freq_idx["train"] += 1
            return
        self.freq_idx["train"] += 1

        self.register_histogram(trainer=trainer, pl_module=pl_module, batch=batch, state="train")


    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
    ):
        if self.freq_idx["val"] % self.freq["val"] != 0:
            self.freq_idx["val"] += 1
            return
        self.freq_idx["val"] += 1

        self.register_histogram(trainer=trainer, pl_module=pl_module, batch=batch, state="val")


class ActivationHistogramBimonn(ActivationHistogramBimonnBase):
    def forward_hist(self, pl_module, input_):
        return pl_module.model.forward(input_)

    @property
    def name(self):
        return "float"


class ActivationHistogramBinaryBimonn(ActivationHistogramBimonnBase):
    def __init__(self, update_binaries=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_binaries = update_binaries

    def forward_hist(self, pl_module, input_):
        model = pl_module.model
        init_mode = model.binary_mode

        model.binary(mode=True, update_binaries=self.update_binaries)
        output = model.forward(input_)
        model.binary(mode=init_mode, update_binaries=self.update_binaries)

        return output

    @property
    def name(self):
        return "binary"
