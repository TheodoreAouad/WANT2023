import pathlib
import itertools
from os.path import join

import matplotlib.pyplot as plt
import torch
import numpy as np

from .observable_layers import ObservableLayersChans
from ..models.weights_layer import WeightsEllipse
from general.utils import max_min_norm, save_json


# DEPRECATED
class PlotParametersBiseEllipse(ObservableLayersChans):

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
        weights_handler = layer.bises.weights_handler
        if not isinstance(weights_handler, WeightsEllipse):
            return

        metrics = {}
        last_params = {}

        metrics[f'params/ellipse_a/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}'] = weights_handler.a_[chan_output]

        trainer.logger.log_metrics(metrics, trainer.global_step)
        self.last_params[layer_idx] = last_params

        trainer.logger.experiment.add_scalars(
            f"comparative/ellipse_a/layer_{layer_idx}_chout_{chan_output}",
            {f"chin_{chan_input}": weights_handler.a_[chan_output]},
            trainer.global_step
        )

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json({k1: {k2: str(v2) for k2, v2 in v1.items()} for k1, v1 in self.last_params.items()}, join(final_dir, "parameters.json"))
        return self.last_params


class PlotSigmaBiseEllipse(ObservableLayersChans):
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
        weights_handler = layer.bises[chan_input].weights_handler
        if not isinstance(weights_handler, WeightsEllipse):
            return

        sigma_inv = weights_handler.sigma_inv[chan_output, 0]

        trainer.logger.experiment.add_figure(
            f"ellipse_sigma_inv/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            self.get_figure(sigma_inv),
            trainer.global_step
        )

    @staticmethod
    def get_figure(weights,):
        weights = weights.cpu().detach()
        weights_normed = max_min_norm(weights)
        figure = plt.figure(figsize=(4, 4))
        plt.title(f"sigma_inv")
        plt.imshow(weights_normed, interpolation='nearest',)
        plt.axis('off')
        # plt.colorbar()

        # Use white text if squares are dark; otherwise black.
        threshold = weights_normed.max() / 2.

        for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
            color = "white" if weights_normed[i, j] < threshold else "black"
            plt.text(j, i, round(weights[i, j].item(), 4), horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        for layer_idx, layer in enumerate(pl_module.model.layers):
            sigma_inv = torch.cat([bise.weights_handler.sigma_inv for bise in layer.bises], axis=1)
            self.last_weights.append({"sigma_inv": sigma_inv})

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(join(final_dir, "png")).mkdir(exist_ok=True, parents=True)
        pathlib.Path(join(final_dir, "npy")).mkdir(exist_ok=True, parents=True)
        for layer_idx, layer_dict in enumerate(self.last_weights):
            for key, weight in layer_dict.items():
                for chan_output in range(weight.shape[0]):
                    for chan_input in range(weight.shape[1]):
                        fig = self.get_figure(
                            weight[chan_output, chan_input]
                        )

                        fig.savefig(join(final_dir, "png", f"{key}_layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
                        np.save(join(final_dir, "npy", f"{key}_layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.npy"), weight[chan_output, chan_input].cpu().detach())
                        plt.close(fig)

        return self.last_weights


class PlotWeightsBiseEllipse(ObservableLayersChans):

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
        weights_norm = layer.weight[chan_output, chan_input]

        mu_ellipse = layer.bises[chan_input].weights_handler.mu[chan_output, 0]

        trainer.logger.experiment.add_figure(
            f"weights/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            self.get_figure_raw_weights(weights_norm, layer.bias_bise[chan_output, chan_input], layer.activation_P_bise[chan_output, chan_input], mu_ellipse),
            trainer.global_step
        )


    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        for layer_idx, layer in enumerate(pl_module.model.layers):
            to_add = {
                "bias_bise": layer.bias_bise, "activation_P_bise": layer.activation_P_bise, "weights": layer.weight,
                "mu_ellipse": torch.cat([bise.weights_handler.mu for bise in layer.bises], axis=1)
            }

            to_add["weights"] = layer.weight

            self.last_weights.append(to_add)

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(join(final_dir, "png")).mkdir(exist_ok=True, parents=True)
        pathlib.Path(join(final_dir, "npy")).mkdir(exist_ok=True, parents=True)
        for layer_idx, layer_dict in enumerate(self.last_weights):
            weight = layer_dict['weights']
            for chan_output in range(weight.shape[0]):
                for chan_input in range(weight.shape[1]):
                    fig = self.get_figure_raw_weights(
                        weight[chan_output, chan_input],
                        bias=layer_dict['bias_bise'][chan_output, chan_input],
                        activation_P=layer_dict['activation_P_bise'][chan_output, chan_input],
                        mu=layer_dict['mu_ellipse'][chan_output, chan_input],
                    )

                    fig.savefig(join(final_dir, "png", f"weights_layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
                    np.save(join(final_dir, "npy", f"weights_layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.npy"), weight[chan_output, chan_input].cpu().detach())
                    plt.close(fig)

        return self.last_weights

    @staticmethod
    def get_figure_raw_weights(weights, bias, activation_P, mu):
        weights = weights.cpu().detach()
        mu = mu.cpu().detach().numpy()
        weights_normed = max_min_norm(weights)
        figure = plt.figure(figsize=(8, 8))
        plt.title(f"bias={bias.item():.3f}  act_P={activation_P.item():.3f}  sum={weights.sum():.3f}   mu=({mu[0]:.2}, {mu[1]:.2})")
        plt.imshow(weights_normed, interpolation='nearest',)
        plt.scatter(mu[1], mu[0], c='red')
        plt.colorbar()
        # plt.clim(0, 1)

        # Use white text if squares are dark; otherwise black.
        threshold = weights_normed.max() / 2.

        for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
            color = "white" if weights_normed[i, j] < threshold else "black"
            plt.text(j, i, round(weights[i, j].item(), 2), horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure



class PlotGradientBiseEllipse(ObservableLayersChans):

    def __init__(self, *args, freq: int = 1, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)

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
        weights_handler = layer.bises[chan_input].weights_handler
        if not isinstance(weights_handler, WeightsEllipse):
            return


        if layer.bises[chan_input].weight.grad is not None:
            grad_mu = weights_handler.mu_grad
            grad_sigma_inv = weights_handler.sigma_inv_grad
            grad_a = weights_handler.a_grad

            trainer.logger.experiment.add_histogram(
                f"weights_gradient_hist/sigma_inv/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
                grad_sigma_inv,
                trainer.global_step
            )
            for grad_tensor, grad_title in zip([grad_mu, grad_sigma_inv, grad_a], ['mu', 'sigma_inv', 'a']):
                trainer.logger.experiment.add_scalars(
                    f"weights/bisel/ellipse_gradient_mean/{grad_title}/layer_{layer_idx}_chout_{chan_output}",
                    {f"chin_{chan_input}": grad_tensor.mean()},
                    trainer.global_step
                )


        if layer.bises[chan_input].bias_handler.grad is not None:
            grad_bise_bias = layer.bises[chan_input].bias_handler.grad[chan_output]
            trainer.logger.experiment.add_scalars(
                f"weights/bisel/bias_gradient/layer_{layer_idx}_chout_{chan_output}",
                {f"chin_{chan_input}": grad_bise_bias},
                trainer.global_step
            )

        # TODO: do the same for LUI
        # for chan in [chan_input, chan_input + layer.in_channels]:
        for chan in [chan_input]:
            if layer.luis[chan_output].weight.grad is not None:
                grad_lui_weight = layer.luis[chan_output].weight.grad[0, chan]
                trainer.logger.experiment.add_scalars(
                    f"weights/lui/weights_gradient/layer_{layer_idx}_chout_{chan_output}",
                    {f"chin_{chan}": grad_lui_weight},
                    trainer.global_step
                )
