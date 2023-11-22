from typing import Any, Dict
import pathlib
from os.path import join
import random

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt

from deep_morpho.datasets import GrayToChannelDatasetBase
from general.nn.observables import Observable


class PlotPredsDefault(Observable):

    def __init__(
        self,
        freq_batch: Dict = {"train": 100, "val": 10, "test": 10},
        figsize_atom=(4, 4),
        n_imgs=10,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.freq_batch = freq_batch
        self.batch_idx = {"train": 0, "val": 0, "test": 0}
        self.saved_fig = {"train": None, "val": None}
        self.figsize_atom = figsize_atom
        self.n_imgs = n_imgs

    def on_validation_batch_end_with_preds(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        preds: Any
    ) -> None:
        if self.freq_batch["val"] is not None and self.batch_idx["val"] % self.freq_batch["val"] == 0:
            self.plot_pred_state(
                trainer=trainer, pl_module=pl_module, batch=batch, preds=preds, state="batch_val",
                title=f'val | epoch {trainer.current_epoch}', step=self.batch_idx["val"])
        self.batch_idx["val"] += 1

    def on_train_batch_end_with_preds(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: 'STEP_OUTPUT',
        batch: 'Any',
        batch_idx: int,
        preds: 'Any',
    ) -> None:
        if self.freq_batch["train"] is not None and self.batch_idx["train"] % self.freq_batch["train"] == 0:
            self.plot_pred_state(trainer=trainer, pl_module=pl_module, batch=batch, preds=preds, state="batch_train", title="train",
                step=trainer.global_step)
        self.batch_idx["train"] += 1

    def on_test_batch_end_with_preds(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: 'STEP_OUTPUT',
        batch: 'Any',
        batch_idx: int,
        preds: 'Any',
    ) -> None:
        if self.freq_batch["test"] is not None and self.batch_idx["test"] % self.freq_batch["test"] == 0:
            self.plot_pred_state(trainer=trainer, pl_module=pl_module, batch=batch, preds=preds, state="batch_test", title="test",
                step=self.batch_idx["test"])
        self.batch_idx["test"] += 1

    def plot_pred_state(self, trainer, pl_module, batch, preds, state, title, step):
        with torch.no_grad():
            imgs = [k.cpu().detach().numpy().transpose(1, 2, 0) for k in batch[0]]
            fig = self.plot_pred(
                imgs,
                *[k.cpu().detach().numpy() for k in [preds, batch[1]]],
                figsize_atom=self.figsize_atom,
                n_imgs=self.n_imgs,
                title=title,
            )
            trainer.logger.experiment.add_figure(f"preds/{state}/input_pred_target", fig, step)
            self.saved_fig[state] = fig


    @staticmethod
    def plot_pred(imgs, preds, targets, figsize_atom, n_imgs, title='',):
        n_imgs = min(n_imgs, len(imgs))
        W, L = figsize_atom
        fig, axs = plt.subplots(n_imgs, 1, figsize=(W, L * n_imgs))

        for ax_idx in range(n_imgs):
            img = imgs[ax_idx]
            img = (img - img.min()) / (img.max() - img.min())
            axs[ax_idx].imshow(img)

        fig.suptitle(title)

        return fig

    def save(self, save_path: str):
        for state in ['train', 'val']:
            if self.saved_fig[state] is not None:
                final_dir = join(save_path, self.__class__.__name__)
                pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
                self.saved_fig[state].savefig(join(final_dir, f"input_pred_target_{state}.png"))
                plt.close(self.saved_fig[state])

        return self.saved_fig




class PlotPreds(Observable):

    def __init__(self, freq: Dict = {"train": 100, "val": 10}, fig_kwargs={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.idx = {"train": 0, "val": 0}
        self.saved_fig = {"train": None, "val": None}
        self.fig_kwargs = fig_kwargs
        self.fig_kwargs['vmin'] = self.fig_kwargs.get('vmin', 0)
        self.fig_kwargs['vmax'] = self.fig_kwargs.get('vmax', 1)

    def on_validation_batch_end_with_preds(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        preds: Any
    ) -> None:
        if self.idx['val'] % self.freq['val'] == 0:
            idx = random.choice(range(len(batch[0])))
            img, target = batch[0][idx], batch[1][idx]
            pred = preds[idx]
            fig = self.plot_three(*[k.cpu().detach().numpy() for k in [img, pred, target]], title=f'val | epoch {trainer.current_epoch}')
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
        if self.idx['train'] % self.freq["train"] == 0:
            with torch.no_grad():
                # idx = random.choice(range(len(batch[0])))
                idx = 0
                img, target = batch[0][idx], batch[1][idx]
                pred = preds[idx]
                fig = self.plot_three(*[k.cpu().detach().numpy() for k in [img, pred, target]], title='train', fig_kwargs=self.fig_kwargs)
                trainer.logger.experiment.add_figure("preds/train/input_pred_target", fig, trainer.global_step)
                self.saved_fig['train'] = fig

        self.idx['train'] += 1

    @staticmethod
    def plot_three(img, pred, target, title='', fig_kwargs={"vmin": 0, "vmax": 1}):
        ncols = max(img.shape[0], pred.shape[0])
        fig, axs = plt.subplots(3, ncols, figsize=(4 * ncols, 4 * 3), squeeze=False)
        fig.suptitle(title)

        for chan in range(img.shape[0]):
            axs[0, chan].imshow(img[chan], cmap='gray', **fig_kwargs)
            axs[0, chan].set_title(f'input_{chan}')

        for chan in range(pred.shape[0]):
            axs[1, chan].imshow(pred[chan], cmap='gray', **fig_kwargs)
            axs[1, chan].set_title(f'pred_{chan} vmin={pred[chan].min():.2} vmax={pred[chan].max():.2}')

        for chan in range(target.shape[0]):
            axs[2, chan].imshow(target[chan], cmap='gray', **fig_kwargs)
            axs[2, chan].set_title(f'target_{chan}')

        return fig

    def save(self, save_path: str):
        for state in ['train', 'val']:
            if self.saved_fig[state] is not None:
                final_dir = join(save_path, self.__class__.__name__)
                pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
                self.saved_fig[state].savefig(join(final_dir, f"input_pred_target_{state}.png"))
                plt.close(self.saved_fig[state])

        return self.saved_fig


class PlotPredsClassif(Observable):

    def __init__(self, freq: Dict = {"train": 100, "val": 10}, figsize_atom=(4, 4), n_imgs=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.idx = {"train": 0, "val": 0}
        self.saved_fig = {"train": None, "val": None}
        self.figsize_atom = figsize_atom
        self.n_imgs = n_imgs

    def on_validation_batch_end_with_preds(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        preds: Any
    ) -> None:
        self.plot_pred_state(
            trainer=trainer, pl_module=pl_module, batch=batch, preds=preds, state="val",
            title=f'val | epoch {trainer.current_epoch}', step=self.idx["val"],)

    def on_train_batch_end_with_preds(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
            outputs: 'STEP_OUTPUT',
            batch: 'Any',
            batch_idx: int,
            preds: 'Any',
    ) -> None:
        self.plot_pred_state(trainer=trainer, pl_module=pl_module, batch=batch, preds=preds, state="train", title="train",
            step=trainer.global_step,)

    def plot_pred_state(self, trainer, pl_module, batch, preds, state, title, step):
        if self.idx[state] % self.freq[state] == 0:
            with torch.no_grad():
                imgs = [k.cpu().detach().numpy()[0] for k in batch[0]]
                fig = self.plot_pred(
                    imgs,
                    *[k.cpu().detach().numpy() for k in [preds, batch[1]]],
                    figsize_atom=self.figsize_atom,
                    n_imgs=self.n_imgs,
                    title=title,
                    xlims=(-1, 1) if hasattr(pl_module.model, "atomic_element") and pl_module.model.atomic_element==["sybisel"] else (0, 1),
                )
                trainer.logger.experiment.add_figure(f"preds/{state}/input_pred_target", fig, step)
                self.saved_fig[state] = fig

        self.idx[state] += 1


    @staticmethod
    def plot_pred(imgs, preds, targets, figsize_atom, n_imgs, title='', xlims=(0, 1), tick_label=None):
        n_imgs = min(n_imgs, len(imgs))
        W, L = figsize_atom
        fig, axs = plt.subplots(n_imgs, 2, figsize=(2 * W, L * n_imgs))
        n_classes = len(preds[0])

        if tick_label is None:
            tick_label = list(range(n_classes))

        for ax_idx in range(n_imgs):
            img, pred, target = imgs[ax_idx], preds[ax_idx], targets[ax_idx]
            axs[ax_idx, 0].imshow(img)
            axs[ax_idx, 0].set_title(tick_label[target.argmax()])

            pred_label = pred.argmax()

            colors = ["red" for _ in range(n_classes)]
            colors[pred_label] = "green"

            axs[ax_idx, 1].barh(range(n_classes), pred, tick_label=tick_label, color=colors)
            axs[ax_idx, 1].set_xlim(*xlims)
            axs[ax_idx, 1].set_title(f'pred: {tick_label[pred.argmax()]}')

            for idx, value in enumerate(pred):
                axs[ax_idx, 1].text(value.item(), idx, f'{value.item():.2f}')

        fig.suptitle(title)

        return fig

    def save(self, save_path: str):
        for state in ['train', 'val']:
            if self.saved_fig[state] is not None:
                final_dir = join(save_path, self.__class__.__name__)
                pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
                self.saved_fig[state].savefig(join(final_dir, f"input_pred_target_{state}.png"))
                plt.close(self.saved_fig[state])

        return self.saved_fig


class PlotPredsClassifChannel(PlotPredsClassif):
    def __init__(self, dataset: GrayToChannelDatasetBase, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset

    def plot_pred_state(self, trainer, pl_module, batch, preds, state, title, step):
        if self.idx[state] % self.freq[state] == 0:
            with torch.no_grad():
                imgs = np.stack([self.dataset.from_channels_to_gray_numpy(img) for img in batch[0]], axis=0)
                fig = self.plot_pred(
                    imgs,
                    *[k.cpu().detach().numpy() for k in [preds, batch[1]]],
                    figsize_atom=self.figsize_atom,
                    n_imgs=self.n_imgs,
                    title=title,
                    xlims=(-1, 1) if hasattr(pl_module.model, "atomic_element") and pl_module.model.atomic_element==["sybisel"] else (0, 1),
                    tick_label=self.dataset.classes
                )
                trainer.logger.experiment.add_figure(f"preds/{state}/input_pred_target", fig, step)
                self.saved_fig[state] = fig

        self.idx[state] += 1

    def save(self, save_path: str):
        super().save(save_path)
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        torch.save(self.dataset.levelset_values, join(final_dir, "levelset_values.pt"))
