from typing import Dict
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from .plot_pred import PlotPreds
from .baseline_metric import InputAsPredMetric
from ..datasets.mnist_dataset import MnistGrayScaleDataset
from .binary_mode_metric import BinaryModeMetricBase
from general.nn.observables import CalculateAndLogMetrics


# TODO: rewrite
class BinaryModeMetricGrayScale(BinaryModeMetricBase):
    def on_train_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ):
        if self.freq_idx % self.freq != 0:
            self.freq_idx += 1
            return
        self.freq_idx += 1

        with torch.no_grad():
            pl_module.model.binary(update_binaries=self.update_binaries)

            new_inputs = (batch[0] > 0).float()  # handle both {0, 1} and {-1, 1}
            new_targets = batch[1] > 0

            new_inputs.indexes = batch[0].indexes
            new_inputs.gray_values = batch[0].gray_values
            new_inputs.original = batch[0].original

            new_targets.original = batch[1].original

            preds = pl_module.model(new_inputs)

            inputs, preds, targets, original_inputs, original_targets = MnistGrayScaleDataset.get_relevent_tensors_batch((new_inputs, new_targets), preds)

            for metric_name in self.metrics:

                for key, tar in zip(["_rec", "_ori"], [targets, original_targets]):
                    key = f'{metric_name}{key}'
                    metric = self.metrics[metric_name](tar, preds)

                    self.last_value[key] = metric
                    # pl_module.log(f"mean_metrics_{batch_or_epoch}/{metric_name}/{state}", metric)

                    trainer.logger.experiment.add_scalars(
                        f"comparative/binary_mode/{key}", {'train': metric}, trainer.global_step
                    )

                    trainer.logger.log_metrics(
                        {f"binary_mode/{key}_{'train'}": metric}, trainer.global_step
                    )
                    trainer.logged_metrics.update(
                        {f"binary_mode/{key}_{'train'}": metric}
                    )


            img, pred, target, original_img, original_target = inputs[0], preds[0], targets[0], original_inputs[0], original_targets[0]
            fig = self.plot_five(*[k.cpu().detach().numpy()[0] for k in [img, pred, target, original_img, original_target]], title='train')
            trainer.logger.experiment.add_figure("preds/train/binary_mode/input_pred_target", fig, trainer.global_step)

            pl_module.model.binary(False)


    @staticmethod
    def plot_five(img, pred, target, original_img, original_target, title='', fig_kwargs={}):
        fig, axs = plt.subplots(2, 3, figsize=(4 * 3, 4 * 2), squeeze=False)
        fig.suptitle(title)

        nb_values = len(np.unique(img))
        MSE_input = np.mean((img - original_img)**2)
        MSE_target = np.mean((target - original_target)**2)

        axs[0, 0].imshow(img, **fig_kwargs)
        axs[0, 0].set_title(f'input - {nb_values} values')

        axs[0, 1].imshow(pred, **fig_kwargs)
        axs[0, 1].set_title(f'rec pred vmin={pred.min():.2} vmax={pred.max():.2}')

        axs[0, 2].imshow(target, **fig_kwargs)
        axs[0, 2].set_title(f'target - {nb_values} values')

        axs[1, 0].imshow(original_img, **fig_kwargs)
        axs[1, 0].set_title(f'ori input - MSE={MSE_input:.2}')

        axs[1, 2].imshow(original_target, **fig_kwargs)
        axs[1, 2].set_title(f'ori target - MSE={MSE_target:.2}')

        return fig


class InputAsPredMetricGrayScale(InputAsPredMetric):
    def on_train_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        if self.freq_idx % self.freq != 0:
            self.freq_idx += 1
            return
        self.freq_idx += 1

        inputs, preds, targets, original_inputs, original_targets = MnistGrayScaleDataset.get_relevent_tensors_batch(batch, preds)
        self._calculate_and_log_metrics(trainer, pl_module, targets, inputs.squeeze(), state='train', suffix='_rec')
        self._calculate_and_log_metrics(trainer, pl_module, original_targets, inputs.squeeze(), state='train', suffix='_ori')


class PlotPredsGrayscale(PlotPreds):
    def __init__(self, freq: Dict = {"train": 100, "val": 10}, fig_kwargs={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.idx = {"train": 0, "val": 0}
        self.saved_fig = {"train": None, "val": None}
        self.fig_kwargs = fig_kwargs


    def on_validation_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: "int",
        preds: "Any",
    ) -> None:
        if self.idx['val'] % self.freq['val'] != 0:
            self.idx['val'] += 1
            return

        idx = random.choice(range(len(batch[0].original)))
        img, pred, target, original_img, original_target = MnistGrayScaleDataset.get_relevent_tensors_idx(idx, batch, preds)
        fig = self.plot_five(
            *[k.cpu().detach().numpy() for k in [img, pred, target, original_img, original_target]],
            title=f'val | epoch {trainer.current_epoch}',
            fig_kwargs=self.fig_kwargs,
        )
        trainer.logger.experiment.add_figure("preds/val/input_pred_target", fig, self.idx['val'])
        self.saved_fig['val'] = fig

        self.idx['val'] += 1

    def on_train_batch_end_with_preds(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: 'STEP_OUTPUT',
        batch: 'Any',
        batch_idx: int,
        preds: 'Any',
    ) -> None:
        if self.idx['train'] % self.freq["train"] != 0:
            self.idx['train'] += 1
            return

        with torch.no_grad():
            idx = 0
            img, pred, target, original_img, original_target = MnistGrayScaleDataset.get_relevent_tensors_idx(idx, batch, preds)
            fig = self.plot_five(
                *[k.cpu().detach().numpy() for k in [img, pred, target, original_img, original_target]],
                title='train',
                fig_kwargs=self.fig_kwargs,
            )
            trainer.logger.experiment.add_figure("preds/train/input_pred_target", fig, trainer.global_step)
            self.saved_fig['train'] = fig

        self.idx['train'] += 1


    @staticmethod
    def plot_five(img, pred, target, original_img, original_target, title='', fig_kwargs={}):
        fig, axs = plt.subplots(2, 3, figsize=(4 * 3, 4 * 2), squeeze=False)
        fig.suptitle(title)

        nb_values = len(np.unique(img))
        MSE_input = np.mean((img - original_img)**2)
        MSE_target = np.mean((target - original_target)**2)

        axs[0, 0].imshow(img, **fig_kwargs)
        axs[0, 0].set_title(f'input - {nb_values} values')

        axs[0, 1].imshow(pred, **fig_kwargs)
        axs[0, 1].set_title(f'rec pred vmin={pred.min():.2} vmax={pred.max():.2}')

        axs[0, 2].imshow(target, **fig_kwargs)
        axs[0, 2].set_title(f'target - {nb_values} values')

        axs[1, 0].imshow(original_img, **fig_kwargs)
        axs[1, 0].set_title(f'ori input - MSE={MSE_input:.2}')

        axs[1, 2].imshow(original_target, **fig_kwargs)
        axs[1, 2].set_title(f'ori target - MSE={MSE_target:.2}')

        return fig


class CalculateAndLogMetricGrayScale(CalculateAndLogMetrics):
    def __init__(self, metrics, keep_preds_for_epoch=False, *args, **kwargs):
        super().__init__(metrics, keep_preds_for_epoch, *args, **kwargs)

    def on_train_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        self.freq_idx['train'] += 1
        if self.freq_idx['train'] % self.freq['train'] == 0:
            inputs, preds, targets, original_inputs, original_targets = MnistGrayScaleDataset.get_relevent_tensors_batch(batch, preds)
            self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='train', suffix="_rec")
            self._calculate_and_log_metrics(trainer, pl_module, original_targets, preds, state='train', suffix="_ori")

    def on_validation_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        self.freq_idx['val'] += 1
        if self.freq_idx['val'] % self.freq['val'] == 0:
            inputs, preds, targets, original_inputs, original_targets = MnistGrayScaleDataset.get_relevent_tensors_batch(batch, preds)
            self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='val', suffix="_rec")
            self._calculate_and_log_metrics(trainer, pl_module, original_targets, preds, state='val', suffix="_ori")

    def on_test_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        self.freq_idx['test'] += 1
        if self.freq_idx['test'] % self.freq['test'] == 0:
            inputs, preds, targets, original_inputs, original_targets = MnistGrayScaleDataset.get_relevent_tensors_batch(batch, preds)
            self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='test', suffix="_rec")
            self._calculate_and_log_metrics(trainer, pl_module, original_targets, preds, state='test', suffix="_ori")
