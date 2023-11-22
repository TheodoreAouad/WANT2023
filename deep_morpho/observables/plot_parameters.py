import pathlib
from os.path import join
import itertools
from typing import Any, Dict


import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch.nn as nn


from .observable_layers import ObservableLayersChans, Observable
from general.utils import max_min_norm, save_json

from ..models import BiSE, BiSEL, BiSELBase


class PlotWeightsBiSE(ObservableLayersChans):

    def __init__(self, *args, freq: int = 100, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)
        self.last_weights = []

    def on_train_batch_end_layers_chans(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
        chan_input: int,
        chan_output: int,
    ):
        weights_param = layer.get_weight_param_bise(chin=chan_input, chout=chan_output)
        weights = layer.get_weight_bise(chin=chan_input, chout=chan_output)
        bias_bise = layer.get_bias_bise(chin=chan_input, chout=chan_output)
        activation_P_bise = layer.get_activation_P_bise(chin=chan_input, chout=chan_output)

        trainer.logger.experiment.add_figure(
            f"weights/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            self.get_figure_raw_weights(weights, bias_bise, activation_P_bise),
            trainer.global_step
        )
        trainer.logger.experiment.add_figure(
            f"weights_raw/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            self.get_figure_raw_weights(weights_param, bias_bise, activation_P_bise),
            trainer.global_step
        )



    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        for layer_idx, layer in enumerate(pl_module.model.layers):
            to_add = {"bias_bise": layer.bias_bise, "activation_P_bise": layer.activation_P_bise}

            # to_add["weights_param"] = layer.bises.weight_param

            to_add["weights"] = layer.weight
            self.last_weights.append(to_add)

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(join(final_dir, "png")).mkdir(exist_ok=True, parents=True)
        pathlib.Path(join(final_dir, "npy")).mkdir(exist_ok=True, parents=True)
        for layer_idx, layer_dict in enumerate(self.last_weights):
            # for key, weight in layer_dict.items():
                # if key not in [
                #     "weights", "weights_param"
                # ]:
                    # continue
            weight = layer_dict["weights"]
            for chan_output in range(weight.shape[0]):
                for chan_input in range(weight.shape[1]):
                    fig = self.get_figure_raw_weights(
                        weight[chan_output, chan_input],
                        bias=layer_dict['bias_bise'][chan_output, chan_input],
                        activation_P=layer_dict['activation_P_bise'][chan_output, chan_input]
                    )

                    fig.savefig(join(final_dir, "png", f"weight_layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
                    np.save(join(final_dir, "npy", f"weight_layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.npy"), weight[chan_output, chan_input].cpu().detach())
                    plt.close(fig)

        return self.last_weights

    @staticmethod
    def get_figure_raw_weights(weights, bias, activation_P):
        weights = weights.cpu().detach()
        weights_normed = max_min_norm(weights)
        figure = plt.figure(figsize=(8, 8))
        plt.title(f"bias={bias.item():.3f}  act_P={activation_P.item():.3f}  sum={weights.sum():.3f}")
        plt.imshow(weights_normed, interpolation='nearest',)
        plt.colorbar()
        # plt.clim(0, 1)

        # Use white text if squares are dark; otherwise black.
        threshold = weights_normed.max() / 2.

        for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
            color = "white" if weights_normed[i, j] < threshold else "black"
            plt.text(j, i, round(weights[i, j].item(), 2), horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure


class PlotParametersBiSE(ObservableLayersChans):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_params = {}

    def on_train_batch_end_layers_chans(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
        chan_input: int,
        chan_output: int,
    ):
        metrics = {}
        last_params = {}
        weights = layer.get_weight_bise(chin=chan_input, chout=chan_output)
        bias_bise = layer.get_bias_bise(chin=chan_input, chout=chan_output)
        activation_P_bise = layer.get_activation_P_bise(chin=chan_input, chout=chan_output)

        weights_sum = weights.sum() / 2
        lb_bias = weights.min()
        ub_bias = 2 * weights_sum

        metrics[f'params/activation_P/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}'] = activation_P_bise
        metrics[f'params/bias_bise/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}'] = bias_bise

        metrics[f'monitor/bise_B-Wsum/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}'] = -bias_bise - weights_sum
        # We want lb < -bias < ub
        metrics[f'monitor/bias_lb/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}'] = -bias_bise - lb_bias
        metrics[f'monitor/bias_ub/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}'] = ub_bias + bias_bise

        trainer.logger.log_metrics(metrics, trainer.global_step)
        self.last_params[layer_idx] = last_params

        trainer.logger.experiment.add_scalars(
            f"comparative/activation_P/layer_{layer_idx}_chout_{chan_output}",
            {f"chin_{chan_input}": activation_P_bise},
            trainer.global_step
        )
        trainer.logger.experiment.add_scalars(
            f"comparative/bias_bise/layer_{layer_idx}_chout_{chan_output}",
            {f"chin_{chan_input}": bias_bise},
            trainer.global_step
        )

        trainer.logger.experiment.add_scalars(
            f"comparative/bise_B-Wsum/layer_{layer_idx}_chout_{chan_output}",
            {f"chin_{chan_input}": -bias_bise - weights_sum},
            trainer.global_step
        )
        trainer.logger.experiment.add_scalars(
            f"comparative/bias_lb/layer_{layer_idx}_chout_{chan_output}",
            {f"chin_{chan_input}": -bias_bise - lb_bias},
            trainer.global_step
        )
        trainer.logger.experiment.add_scalars(
            f"comparative/bias_ub/layer_{layer_idx}_chout_{chan_output}",
            {f"chin_{chan_input}": ub_bias + bias_bise},
            trainer.global_step
        )

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json({k1: {k2: str(v2) for k2, v2 in v1.items()} for k1, v1 in self.last_params.items()}, join(final_dir, "parameters.json"))
        return self.last_params


class ActivationPHistogramBimonn(Observable):
    def __init__(self, *args, freq: Dict = {"train": 100, "val": 30}, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = {"train": 0, "val": 0,}


    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
    ):
        if self.freq['train'] is not None and self.freq_idx["train"] % self.freq["train"] != 0:
            self.freq_idx["train"] += 1
            return
        self.freq_idx["train"] += 1

        for idx, layer in enumerate(pl_module.model.layers):
            # Create function generator such that it does not give the same layer each time
            if isinstance(layer, BiSELBase):
                trainer.logger.experiment.add_histogram(
                    f"param_P_hist/layer_{idx}_bise",
                    layer.activation_P_bise,
                    trainer.global_step
                )
                trainer.logger.experiment.add_histogram(
                    f"param_P_hist/layer_{idx}_lui",
                    layer.activation_P_lui,
                    trainer.global_step
                )
