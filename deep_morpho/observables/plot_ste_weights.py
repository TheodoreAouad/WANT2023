import itertools

import matplotlib.pyplot as plt
import numpy as np

from general.nn.observables import Observable
from general.utils import max_min_norm, save_json


class PlotSteWeights(Observable):
    def __init__(self, *args, freq: int = 100, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)
        self.last_weights = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.freq_idx % self.freq != 0:
            self.freq_idx += 1
            return
        self.freq_idx += 1

        for layer_idx, layer in enumerate(pl_module.model.layers):
            weights = layer.weight
            binarized_weight = layer.binarized_weight
            for chin in range(weights.shape[1]):
                for chout in range(weights.shape[0]):
                    trainer.logger.experiment.add_figure(
                        f"weights/layer_{layer_idx}_chin_{chin}_chout_{chout}",
                        self.get_figure_weights(weights[chout, chin]),
                        trainer.global_step
                    )
                    trainer.logger.experiment.add_figure(
                        f"weights_binary/layer_{layer_idx}_chin_{chin}_chout_{chout}",
                        self.get_figure_binary_weights(binarized_weight[chout, chin]),
                        trainer.global_step
                    )

    @staticmethod
    def get_figure_binary_weights(weights):
        weights = weights.cpu().detach()

        figure = plt.figure(figsize=(8, 8))
        plt.imshow(weights, interpolation='nearest', cmap="gray", vmin=-1, vmax=1)
        plt.colorbar()

        return figure

    @staticmethod
    def get_figure_weights(weights):
        weights = np.clip(weights.cpu().detach(), -1, 1)
        figure = plt.figure(figsize=(8, 8))

        plt.imshow(weights, interpolation='nearest', cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()

        for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
            color = "white" if abs(weights[i, j]) > .7 else "black"
            plt.text(j, i, round(weights[i, j].item(), 2), horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure
