from abc import ABC, abstractmethod
from functools import partial
import pathlib
from os.path import join
from typing import Dict, Tuple

import torch
import matplotlib.pyplot as plt
import numpy as np

from general.nn.observables.observable import Observable
from deep_morpho.datasets import GrayToChannelDatasetBase
from .plot_pred import PlotPredsClassif, PlotPredsClassifChannel, PlotPreds
from general.utils import save_json



# class BinaryModeMetric(Observable):

#     def __init__(self, metrics, freq=100, plot_freq={}, do_plot_figure: bool = True, update_binaries: bool = False):
#         self.metrics = metrics
#         self.freq = freq
#         self.freq_idx = 0
#         self.last_value = {}
#         self.do_plot_figure = do_plot_figure
#         self.update_binaries = update_binaries


#     def on_train_batch_end_with_preds(
#         self,
#         trainer: "pl.Trainer",
#         pl_module: "pl.LightningModule",
#         outputs: "STEP_OUTPUT",
#         batch: "Any",
#         batch_idx: int,
#         preds: "Any",
#     ):
#         if self.freq_idx % self.freq != 0:
#             self.freq_idx += 1
#             return
#         self.freq_idx += 1

#         with torch.no_grad():
#             pl_module.model.binary(update_binaries=self.update_binaries)

#             inputs, targets = batch
#             inputs = (inputs > 0).float()  # handle both {0, 1} and {-1, 1}
#             targets = (targets > 0).float()  # handle both {0, 1} and {-1, 1}

#             preds = pl_module.model(inputs)
#             for metric_name in self.metrics:
#                 metric = self.metrics[metric_name](targets, preds)
#                 self.last_value[metric_name] = metric
#                 # pl_module.log(f"mean_metrics_{batch_or_epoch}/{metric_name}/{state}", metric)

#                 trainer.logger.experiment.add_scalars(
#                     f"comparative/binary_mode/{metric_name}", {'train': metric}, trainer.global_step
#                 )

#                 trainer.logger.log_metrics(
#                     {f"binary_mode/{metric_name}_{'train'}": metric}, trainer.global_step
#                 )
#                 trainer.logged_metrics.update(
#                     {f"binary_mode/{metric_name}_{'train'}": metric}
#                 )

#             img, pred, target = inputs[0], preds[0], targets[0]
#             if self.do_plot_figure:
#                 fig = self.plot_three(*[k.cpu().detach().numpy() for k in [img, pred, target]], title='train')
#                 trainer.logger.experiment.add_figure("preds/train/binary_mode/input_pred_target", fig, trainer.global_step)

#             pl_module.model.binary(False)


#     @staticmethod
#     def plot_three(img, pred, target, title=''):
#         ncols = max(img.shape[0], pred.shape[0])
#         fig, axs = plt.subplots(3, ncols, figsize=(4 * ncols, 4 * 3), squeeze=False)
#         fig.suptitle(title)

#         for chan in range(img.shape[0]):
#             axs[0, chan].imshow(img[chan], cmap='gray', vmin=0, vmax=1)
#             axs[0, chan].set_title(f'input_{chan}')

#         for chan in range(pred.shape[0]):
#             axs[1, chan].imshow(pred[chan], cmap='gray', vmin=0, vmax=1)
#             axs[1, chan].set_title(f'pred_{chan} vmin={pred[chan].min():.2} vmax={pred[chan].max():.2}')

#         for chan in range(target.shape[0]):
#             axs[2, chan].imshow(target[chan], cmap='gray', vmin=0, vmax=1)
#             axs[2, chan].set_title(f'target_{chan}')

#         return fig

#     def save(self, save_path: str):
#         final_dir = join(save_path, self.__class__.__name__)
#         pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
#         save_json({k: str(v) for k, v in self.last_value.items()}, join(final_dir, "metrics.json"))
#         return self.last_value


class BinaryModeMetricBase(Observable, ABC):

    def __init__(
        self,
        metrics,
        freq: Dict = {"train": 100, "val": 10, "test": 1},
        plot_freq: Dict = {"train": 100, "val": 10, "test": 100},
        figsize_atom=(4, 4),
        update_binaries: bool = False,
        n_imgs=10,
    ):
        self.metrics = metrics
        self.update_binaries = update_binaries
        self.metrics_sum = {state: {k: 0 for k in metrics.keys()} for state in ['train', 'val', 'test']}
        self.n_inputs = {state: 0 for state in ['train', 'val', 'test']}

        self.freq = freq
        self.freq_idx = {"train": 0, "val": 0, "test": 0}
        self.plot_freq = plot_freq
        self.plot_freq_idx = {"train": 0, "val": 0, "test": 0}
        self.test_step = 0

        self.last_value = {state: {} for state in ["train", "val", "test"]}

        self.val_step = 0
        self.figsize_atom = figsize_atom
        self.n_imgs = n_imgs

    def metric_mean(self, state, key):
        return self.metrics_sum[state][key] / max(1, self.n_inputs[state])

    def _update_metric_with_loss(self, pl_module):
        self.metrics.update({"loss": lambda x, y: pl_module.compute_loss_value(y, x)})
        for state in ["train", "val", "test"]:
            self.metrics_sum[state].update({"loss": 0})

    def on_train_start(self, trainer, pl_module):
        self._update_metric_with_loss(pl_module)

    def on_test_start(self, trainer, pl_module):
        if 'loss' not in self.metrics.keys():
            self._update_metric_with_loss(pl_module)

    def on_train_epoch_start(self, *args, **kwargs):
        for key in self.metrics_sum["train"]:
            self.metrics_sum["train"][key] = 0
            self.metrics_sum["val"][key] = 0
        self.n_inputs["train"] = 0
        self.n_inputs["val"] = 0

    def on_test_epoch_start(self, *args, **kwargs):
        for key in self.metrics_sum["test"]:
            self.metrics_sum["test"][key] = 0
        self.n_inputs["test"] = 0

    def on_train_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ):
        self._compute_metric_and_plot(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            preds=preds,
            state='train'
        )


    def on_validation_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ):
        self._compute_metric_and_plot(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            preds=preds,
            state='val'
        )

    def on_test_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ):
        self._compute_metric_and_plot(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            preds=preds,
            state='test'
        )

    def do_metric_update(self, state):
        return self.freq[state] is not None and self.freq_idx[state] % self.freq[state] == 0
    
    def do_plot_update(self, state):
        return self.plot_freq[state] is not None and self.plot_freq_idx[state] % self.plot_freq[state] == 0

    def _compute_metric_and_plot(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
        state: str,
    ):
        if self.freq[state] is None and self.plot_freq[state] is None:
            return

        if not self.do_metric_update(state) and not self.do_plot_update(state):
            self.plot_freq_idx[state] += 1
            self.freq_idx[state] += 1
            return

        if state == "test":
            step = self.test_step
            self.test_step += 1
        else:
            step = trainer.global_step

        with torch.no_grad():
            pl_module.model.binary(update_binaries=self.update_binaries)

            inputs, targets = batch
            inputs = (inputs > 0).float()  # handle both {0, 1} and {-1, 1}
            targets = (targets > 0).float()  # handle both {0, 1} and {-1, 1}

            preds = pl_module.model(inputs)
            self.n_inputs[state] += targets.shape[0]

            if self.freq_idx[state] % self.freq[state] == 0:
                for metric_name in self.metrics:
                    metric = self.metrics[metric_name](targets, preds)
                    self.metrics_sum[state][metric_name] += metric * targets.shape[0]

                    # pl_module.log(f"mean_metrics_{batch_or_epoch}/{metric_name}/{state}", metric)
                    pl_module.log(f"binary_mode/{metric_name}_{state}", metric)

                    trainer.logger.experiment.add_scalars(
                        f"comparative/binary_mode/{metric_name}", {state: metric}, step
                    )

                    trainer.logger.log_metrics(
                        {f"binary_mode/{metric_name}_{state}": metric}, step
                    )
                    trainer.logged_metrics.update(
                        {f"binary_mode/{metric_name}_{state}": metric}
                    )

            # if self.plot_freq[state] is not None and self.plot_freq_idx[state] % self.plot_freq[state] == 0:
            if self.do_plot_update(state):
                fig = self.plot_step(state=state, inputs=inputs, preds=preds, targets=targets, pl_module=pl_module,)
                trainer.logger.experiment.add_figure(f"preds/{state}/binary_mode/input_pred_target", fig, step)

            self.plot_freq_idx[state] += 1
            self.freq_idx[state] += 1

            pl_module.model.binary(False)

    @abstractmethod
    def plot_step(self, state, inputs, preds, targets, pl_module):
        pass

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        dict_str = {}
        for k1, v1 in self.last_value.items():
            dict_str[k1] = {}
            for k2, v2 in v1.items():
                dict_str[k1][k2] = str(v2)
        save_json(dict_str, join(final_dir, "metrics.json"))
        return self.last_value

    def on_train_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None
    ):
        for metric_name in self.metrics.keys():
            metric = self.metric_mean("train", metric_name)
            trainer.logger.log_metrics(
                {f"binary_mode/metrics_epoch_mean/{metric_name}_train": metric}, step=trainer.current_epoch
            )
            pl_module.log(f"binary_mode/metrics_epoch_mean/per_batch_step/{metric_name}_train", metric)
            self.last_value["train"][metric_name] = metric

    def on_validation_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None
    ):
        for metric_name in self.metrics.keys():
            metric = self.metric_mean("val", metric_name)
            trainer.logger.log_metrics(
                {f"binary_mode/metrics_epoch_mean/{metric_name}_val": metric}, step=trainer.current_epoch
            )
            pl_module.log(f"binary_mode/metrics_epoch_mean/per_batch_step/{metric_name}_val", metric)
            self.last_value["val"][metric_name] = metric

    def on_test_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None
    ):
        for metric_name in self.metrics.keys():
            metric = self.metric_mean("test", metric_name)
            trainer.logger.log_metrics(
                {f"binary_mode/metrics_epoch_mean/{metric_name}_test": metric}, step=trainer.current_epoch
            )
            pl_module.log(f"binary_mode/metrics_epoch_mean/per_batch_step/{metric_name}_test", metric)
            self.last_value["test"][metric_name] = metric

    def save_hparams(self) -> Dict:
        res = {}
        for state in ["train", "val", "test"]:
            for metric_name in self.metrics.keys():
                res[f"binary_{metric_name}_{state}"] = self.last_value[state][metric_name]
        return res


class BinaryModeMetricMorpho(BinaryModeMetricBase):
    def plot_step(self, state, inputs, preds, targets, pl_module):
        inputs, preds, targets = [k.cpu().detach().numpy()[0] for k in [inputs, preds, targets]]
        return PlotPreds.plot_three(
            inputs, preds, targets,
            title=f"binary_mode/{state}",
            fig_kwargs={"vmin": 0, "vmax": 1}
        )


class BinaryModeMetricClassif(BinaryModeMetricBase):

    def plot_step(self, state, inputs, preds, targets, pl_module):
        imgs = [k.cpu().detach().numpy()[0] for k in inputs]
        fig = self.plot_pred(
            imgs,
            *[k.cpu().detach().numpy() for k in [preds, targets]],
            figsize_atom=self.figsize_atom,
            n_imgs=self.n_imgs,
            title=state,
            xlims=(-1, 1) if hasattr(pl_module.model, "atomic_element") and pl_module.model.atomic_element==["sybisel"] else (0, 1),
        )
        return fig


    @staticmethod
    def plot_pred(*args, **kwargs):
        return PlotPredsClassif.plot_pred(*args, **kwargs)


class BinaryModeMetricClassifChannel(BinaryModeMetricBase):
    def __init__(self, dataset: GrayToChannelDatasetBase, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset

    @staticmethod
    def plot_pred(*args, **kwargs):
        return PlotPredsClassifChannel.plot_pred(*args, **kwargs)

    def plot_step(self, state, inputs, preds, targets, pl_module):
        imgs = np.stack([self.dataset.from_channels_to_gray_numpy(img) for img in inputs], axis=0)
        fig = self.plot_pred(
            imgs,
            *[k.cpu().detach().numpy() for k in [preds, targets]],
            figsize_atom=self.figsize_atom,
            n_imgs=self.n_imgs,
            title=state,
            xlims=(-1, 1) if hasattr(pl_module.model, "atomic_element") and pl_module.model.atomic_element==["sybisel"] else (0, 1),
            tick_label=self.dataset.classes,
        )
        return fig
