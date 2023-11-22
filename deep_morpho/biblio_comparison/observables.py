from os.path import join
import itertools
import pathlib

import torch
import matplotlib.pyplot as plt

from deep_morpho.observables import ObservableLayers, Observable
from general.utils import max_min_norm, save_json
from .vizualiser import SequentialWeightVizualiser


class PlotWeights(ObservableLayers):

    def __init__(self, *args, freq: int = 100, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)
        self.last_weights = []

    def on_train_batch_end_layers(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
    ):
        weights = layer.weight
        trainer.logger.experiment.add_figure(
            f"weights/layer_{layer_idx}",
            self.get_figure_weights(weights, title=f'param={layer.P.item():.2f}'),
            trainer.global_step
        )

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        for layer_idx, layer in enumerate(pl_module.model.layers):
            self.last_weights.append(layer.weight)
            trainer.logger.experiment.add_figure(
                f"weights/layer_{layer_idx}",
                self.get_figure_weights(layer.weight, title=f'param={layer.P.item():.2f}'),
                trainer.global_step
            )

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        for layer_idx, weight in enumerate(self.last_weights):
            fig = self.get_figure_weights(weight)
            fig.savefig(join(final_dir, f"layer_{layer_idx}.png"))
            plt.close(fig)

        return self.last_weights

    @staticmethod
    def get_figure_weights(weights, title=''):
        weights = weights.cpu().detach()
        weights_normed = max_min_norm(weights)
        figure = plt.figure(figsize=(8, 8))
        plt.title(title)
        plt.imshow(weights_normed, interpolation='nearest', cmap="viridis")
        # plt.colorbar()
        # plt.clim(0, 1)

        # Use white text if squares are dark; otherwise black.
        threshold = weights_normed.max() / 2.

        for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
            color = "white" if weights_normed[i, j] < threshold else "black"
            plt.text(j, i, round(weights[i, j].item(), 2), horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure


class PlotParameters(ObservableLayers):

    def __init__(self, *args, freq: int = 1, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)
        self.last_params = {}

    def on_train_batch_end_layers(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
    ):
        trainer.logger.log_metrics({f"params/P/layer_{layer_idx}": layer.P}, trainer.global_step)
        self.last_params[layer_idx] = layer.P.item()

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json(self.last_params, join(final_dir, "params.json"))


class PlotBiblioModel(Observable):

    def __init__(self, freq: int = 300, figsize=None, dpi=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = 0
        self.figsize = figsize
        self.dpi = dpi

        self.last_fig = {}

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
        vizualiser = SequentialWeightVizualiser(pl_module.model)
        fig = vizualiser.get_fig(figsize=self.figsize, dpi=self.dpi)
        trainer.logger.experiment.add_figure("model", fig, trainer.global_step)

        self.last_fig = fig

    def save(self, save_path: str):
        if self.last_fig is None:
            return

        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        self.last_fig.savefig(join(final_dir, "model.png"))
        plt.close(self.last_fig)

        return self.last_fig
