import pathlib
from os.path import join
import torch

import matplotlib.pyplot as plt

from general.nn.observables import Observable
from deep_morpho.viz.bimonn_viz import BimonnVizualiser
from deep_morpho.viz.ste_conv_viz import SteConvVizualiser


class PlotBimonn(Observable):

    def __init__(self, freq: int = 300, figsize=None, dpi=None, do_plot={"weights": True, "learned": True, "closest": True,}, update_binaries: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = 0
        self.figsize = figsize
        self.dpi = dpi
        self.do_plot = do_plot
        self.update_binaries = update_binaries

        self.last_figs = {}

    def on_train_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ) -> None:
        with torch.no_grad():
            if self.freq_idx % self.freq == 0:
                self.save_figs(trainer, pl_module)
        self.freq_idx += 1

    def on_train_end(self, trainer, pl_module):
        self.save_figs(trainer, pl_module)

    def save_figs(self, trainer, pl_module):
        for key, do_key in self.do_plot.items():
            if do_key:
                vizualiser = BimonnVizualiser(pl_module.model, mode=key, update_binaries=self.update_binaries)
                fig = vizualiser.get_fig(figsize=self.figsize, dpi=self.dpi)
                trainer.logger.experiment.add_figure(f"model/{key}", fig, trainer.global_step)
                if key in self.last_figs.keys():
                    plt.close(self.last_figs[key])
                self.last_figs[key] = fig

    def save(self, save_path: str):
        if len(self.last_figs) == 0:
            return

        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        for key, fig in self.last_figs.items():
            fig.savefig(join(final_dir, f"model_{key}.png"))
            plt.close(fig)

        return self.last_figs


class PlotSTE(Observable):

    def __init__(self, freq: int = 300, figsize=None, dpi=None, do_plot={"weight": True, "binary": True}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = 0
        self.figsize = figsize
        self.dpi = dpi
        self.do_plot = do_plot

        self.last_figs = {}

    def on_train_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ) -> None:
        with torch.no_grad():
            if self.freq_idx % self.freq == 0:
                self.save_figs(trainer, pl_module)
        self.freq_idx += 1

    def on_train_end(self, trainer, pl_module):
        self.save_figs(trainer, pl_module)

    def save_figs(self, trainer, pl_module):
        for key, do_key in self.do_plot.items():
            if do_key:
                vizualiser = SteConvVizualiser(pl_module.model, mode=key,)
                fig = vizualiser.get_fig(figsize=self.figsize, dpi=self.dpi)
                trainer.logger.experiment.add_figure(f"model/{key}", fig, trainer.global_step)
                if key in self.last_figs.keys():
                    plt.close(self.last_figs[key])
                self.last_figs[key] = fig

    def save(self, save_path: str):
        if len(self.last_figs) == 0:
            return

        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        for key, fig in self.last_figs.items():
            fig.savefig(join(final_dir, f"model_{key}.png"))
            plt.close(fig)

        return self.last_figs
