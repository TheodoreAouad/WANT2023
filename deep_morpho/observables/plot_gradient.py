import matplotlib.pyplot as plt
import itertools
from typing import Any
import pathlib
from os.path import join

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch.nn as nn
from tqdm import tqdm

from deep_morpho.models.bisel import BiSEL
from .observable_layers import ObservableLayersChans
from general.utils import max_min_norm
from ..models import BiSE



class GradientWatcher(ObservableLayersChans):

    @staticmethod
    def get_figure_gradient(gradient, title=''):
        gradient = gradient.cpu().detach()
        gradient_normed = max_min_norm(gradient)
        figure = plt.figure(figsize=(8, 8))
        plt.title(title, wrap=True)
        plt.imshow(gradient_normed, interpolation='nearest', cmap=plt.cm.gray)
        plt.colorbar()
        # plt.clim(gradient.min(), gradient.max())

        # Use white text if squares are dark; otherwise black.
        threshold = gradient_normed.max() / 2.

        for i, j in itertools.product(range(gradient.shape[0]), range(gradient.shape[1])):
            color = "white" if gradient_normed[i, j] < threshold else "black"
            # plt.text(j, i, round(gradient[i, j].item(), 2), horizontalalignment="center", color=color)
            plt.text(j, i, f"{gradient[i, j].item():.2e}", horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure


class PlotGradientBise(GradientWatcher):

    def __init__(self, *args, freq: int = 100, plot_figure=True, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)
        self.plot_figure = plot_figure

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
        grad_weights = layer.get_weight_grad_bise(chin=chan_input, chout=chan_output)
        if grad_weights is None:
            return

        if self.plot_figure:
            trainer.logger.experiment.add_figure(
                f"weights_gradient/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
                self.get_figure_gradient(grad_weights),
                trainer.global_step
            )
        trainer.logger.experiment.add_histogram(
            f"weights_gradient_hist/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            grad_weights,
            trainer.global_step
        )

    def on_train_batch_end_layers_chans_always(
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
        grad_bise_bias = layer.get_bias_grad_bise(chin=chan_input, chout=chan_output)
        if grad_bise_bias is not None:
            trainer.logger.experiment.add_scalars(
                f"weights/bisel/bias_gradient/layer_{layer_idx}_chout_{chan_output}",
                {f"chin_{chan_input}": grad_bise_bias},
                trainer.global_step
            )

        grad_bise_weights = layer.get_weight_grad_bise(chin=chan_input, chout=chan_output)
        if grad_bise_weights is not None:
            trainer.logger.experiment.add_scalars(
                f"weights/bisel/weights_gradient_mean/layer_{layer_idx}_chout_{chan_output}",
                {f"chin_{chan_input}": grad_bise_weights.mean()},
                trainer.global_step
            )

        for chan in [chan_input]:
            grad_lui_weight = layer.get_weight_grad_lui(chin=chan_input, chout=chan_output)
            if grad_lui_weight is not None:
                trainer.logger.experiment.add_scalars(
                    f"weights/lui/weights_gradient/layer_{layer_idx}_chout_{chan_output}",
                    {f"chin_{chan}": grad_lui_weight},
                    trainer.global_step
                )

    def on_train_batch_end_layers_chan_output_always(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
        chan_output: int,
    ):
        grad_lui_bias = layer.get_bias_grad_lui(chout=chan_output)
        if grad_lui_bias is None:
            return

        trainer.logger.experiment.add_scalars(
            f"weights/lui/bias_gradient/layer_{layer_idx}",
            {f"chout_{chan_output}": grad_lui_bias},
            trainer.global_step
        )



class ExplosiveWeightGradientWatcher(GradientWatcher):

    def __init__(self, threshold=.1, max_nb_save=3, progress_bar=True, *args, freq: int = 1, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)
        self.threshold = threshold
        self.max_nb_save = max_nb_save
        self.max_grads = [0 for _ in range(max_nb_save)]
        # self.cur_nb_save = 0
        self.progress_bar = progress_bar

    def on_train_batch_end_with_preds_layers_chans_always(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
        layer: "nn.Module",
        layer_idx: int,
        chan_input: int,
        chan_output: int,
    ):
        grad_weights = layer.get_weight_grad_bise(chin=chan_input, chout=chan_output)
        if grad_weights is None:
            return

        weights = layer.get_weight_bise(chin=chan_input, chout=chan_output)
        grad_bias = layer.get_bias_grad_bise(chin=chan_input, chout=chan_output)

        # if grad_weights.mean().abs() < self.threshold:
        #     return

        mean_grad = grad_weights.mean().abs()
        if mean_grad < self.max_grads[0]:
            return

        self.max_grads[0] = mean_grad
        self.max_grads.sort()

        trainer.logger.experiment.add_figure(
            f"explosive_weights_gradient/gradient/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            self.get_figure_gradient(grad_weights, title=f'bias_grad={grad_bias.item():.2f}'),
            trainer.global_step
        )
        trainer.logger.experiment.add_histogram(
            f"explosive_weights_gradient_hist/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            grad_weights,
            trainer.global_step
        )

        trainer.logger.experiment.add_figure(
            f"explosive_weights_gradient/weights/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            self.get_figure_raw_weights(weights),
            trainer.global_step
        )

        # if self.cur_nb_save >= self.max_nb_save:
        #     return
        # self.cur_nb_save += 1

        savepath = join(trainer.log_dir, "observables", self.__class__.__name__, f"epoch{trainer.current_epoch}_batch{batch_idx}_chin{chan_input}_chout{chan_output}")

        pathlib.Path(join(savepath, "model_checkpoint")).mkdir(exist_ok=True, parents=True)
        torch.save(pl_module.model.state_dict(), join(savepath, "model_checkpoint", "model.pt"))

        pathlib.Path(join(savepath, "batch")).mkdir(exist_ok=True, parents=True)

        imgs, targets = batch
        iterator = range(len(imgs))
        if self.progress_bar:
            iterator = tqdm(iterator)
        for img_idx in iterator:
            img, target, pred = imgs[img_idx], targets[img_idx], preds[img_idx]
            fig = self.plot_three(
                *[k.cpu().detach().numpy() for k in [img, pred, target]],
                title=f'weights_grad={grad_weights.mean().item():.2f}  bias_grad={grad_bias.item():.2f} epoch={trainer.current_epoch} batch={batch_idx} img={img_idx}'
            )
            fig.savefig(join(savepath, "batch", f"img{img_idx}.png"))
            plt.close(fig)


    @staticmethod
    def plot_three(img, pred, target, title=''):
        ncols = max(img.shape[0], pred.shape[0])
        fig, axs = plt.subplots(3, ncols, figsize=(4 * ncols, 4 * 3), squeeze=False)
        fig.suptitle(title, wrap=True)

        for chan in range(img.shape[0]):
            axs[0, chan].imshow(img[chan], cmap='gray', vmin=0, vmax=1)
            axs[0, chan].set_title(f'input_{chan}')

        for chan in range(pred.shape[0]):
            axs[1, chan].imshow(pred[chan], cmap='gray', vmin=0, vmax=1)
            axs[1, chan].set_title(f'pred_{chan} vmin={pred[chan].min():.2} vmax={pred[chan].max():.2}')

        for chan in range(target.shape[0]):
            axs[2, chan].imshow(target[chan], cmap='gray', vmin=0, vmax=1)
            axs[2, chan].set_title(f'target_{chan}')

        return fig

    @staticmethod
    def get_figure_raw_weights(weights):
        weights = weights.cpu().detach()
        weights_normed = max_min_norm(weights)
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(weights_normed, interpolation='nearest', cmap=plt.cm.gray)
        plt.colorbar()
        # plt.clim(0, 1)

        # Use white text if squares are dark; otherwise black.
        threshold = weights_normed.max() / 2.

        for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
            color = "white" if weights_normed[i, j] < threshold else "black"
            plt.text(j, i, round(weights[i, j].item(), 2), horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure
