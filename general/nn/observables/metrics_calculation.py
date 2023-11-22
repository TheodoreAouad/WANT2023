import pathlib
from os.path import join
from typing import Dict

import torch
from .observable import Observable
from ...utils import save_json


class CalculateAndLogMetrics(Observable):
    """
    class used to calculate and track metrics in the tensorboard
    """
    def __init__(self, metrics, keep_preds_for_epoch=True, freq={"train": 1, "val": 1, "test": 1}):
        self.freq = freq
        self.freq_idx = {"train": 1, "val": 1, "test": 1}
        self.metrics = {k: v for k, v in metrics.items()}
        self.metrics_sum = {state: {k: 0 for k in metrics.keys()} for state in ['train', 'val', 'test']}
        self.n_inputs = {state: 0 for state in ['train', 'val', 'test']}
        self.last_value = {state: {} for state in ["train", "val", "test"]}
        self.keep_preds_for_epoch = keep_preds_for_epoch

        if self.keep_preds_for_epoch:
            self.all_preds = {'train': torch.tensor([]), 'val': torch.tensor([]), 'test': torch.tensor([])}
            self.all_targets = {'train': torch.tensor([]), 'val': torch.tensor([]), 'test': torch.tensor([])}
        self.tb_steps = {metric: {} for metric in self.metrics.keys()}
        # self.tb_steps = {metric: {"train": 0, "val": 0, "test": 0} for metric in self.metrics.keys()}

        self._hp_metrics = dict(
            **{f"metrics_{batch_or_epoch}/{metric_name}_{state}": -1
               for batch_or_epoch in ['batch', 'epoch']
               for metric_name in self.metrics.keys()
               for state in ['train', 'val']
               }
        )

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
        self.n_inputs["train"] = 0

    def on_validation_epoch_start(self, *args, **kwargs):
        for key in self.metrics_sum["val"]:
            self.metrics_sum["val"][key] = 0
        self.n_inputs["val"] = 0

    def on_test_epoch_start(self, *args, **kwargs):
        for key in self.metrics_sum["test"]:
            self.metrics_sum["test"][key] = 0
        self.n_inputs["test"] = 0

    def on_train_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        self.freq_idx['train'] += 1
        if self.freq_idx['train'] % self.freq['train'] == 0:
            inputs, targets = batch
            self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='train')

    def on_validation_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        self.freq_idx['val'] += 1
        if self.freq_idx['val'] % self.freq['val'] == 0:
            inputs, targets = batch
            self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='val')

    def on_test_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        self.freq_idx['test'] += 1
        if self.freq_idx['test'] % self.freq['test'] == 0:
            inputs, targets = batch
            self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='test')

    def _calculate_and_log_metrics(self, trainer, pl_module, targets, preds, state='train', batch_or_epoch='batch', suffix=""):
        key = f"{state}{suffix}"

        if batch_or_epoch == 'batch':
            self.n_inputs[state] += targets.shape[0]
        for metric_name in self.metrics:
            metric = self.metrics[metric_name](targets, preds)

            if batch_or_epoch == 'batch':
                step = trainer.global_step
                self.metrics_sum[state][metric_name] += metric * targets.shape[0]

            else:
                step = trainer.current_epoch

            pl_module.log(f"metrics_{batch_or_epoch}/{metric_name}{suffix}_{state}", metric)

            trainer.logger.experiment.add_scalars(
                f"comparative/metrics_{batch_or_epoch}/{metric_name}{suffix}", {state: metric}, step
            )

            trainer.logger.log_metrics(
                {f"metrics_{batch_or_epoch}/{metric_name}{suffix}_{state}": metric}, step
            )
            trainer.logged_metrics.update(
                {f"metrics_{batch_or_epoch}/{metric_name}{suffix}_{state}": metric}
            )

            # if batch_or_epoch == 'batch':
            #     self.tb_steps[metric_name][key] = step + 1

            # f"metrics_multi_label_{batch_or_epoch}/{metric_name}/{state}", {f'label_{name_label}': metric}, step
            # trainer.logger.experiment.add_scalars(metric_name, {f'{metric_name}_{state}': metric})

    def on_train_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None
    ):
        if self.keep_preds_for_epoch:
            self._calculate_and_log_metrics(trainer, pl_module, self.all_targets['train'], self.all_preds['train'], state='train', batch_or_epoch='epoch')
            self.all_preds['train'] = torch.tensor([])
            self.all_targets['train'] = torch.tensor([])

        for metric_name in self.metrics.keys():
            metric = self.metric_mean("train", metric_name)
            trainer.logger.log_metrics(
                {f"metrics_epoch_mean/{metric_name}_train": metric}, step=trainer.current_epoch
            )
            pl_module.log(f"metrics_epoch_mean/per_batch_step/{metric_name}_train", metric)
            self.last_value["train"][metric_name] = metric


    def on_validation_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if self.keep_preds_for_epoch:
            self._calculate_and_log_metrics(trainer, pl_module, self.all_targets['val'], self.all_preds['val'], state='val', batch_or_epoch='epoch')
            self.all_preds['val'] = torch.tensor([])
            self.all_targets['val'] = torch.tensor([])

        for metric_name in self.metrics.keys():
            metric = self.metric_mean("val", metric_name)
            trainer.logger.log_metrics(
                {f"metrics_epoch_mean/{metric_name}_val": metric}, step=trainer.current_epoch
            )
            pl_module.log(f"metrics_epoch_mean/per_batch_step/{metric_name}_val", metric, )
            self.last_value["val"][metric_name] = metric



    def on_test_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if self.keep_preds_for_epoch:
            self._calculate_and_log_metrics(trainer, pl_module, self.all_targets['test'], self.all_preds['test'], state='test', batch_or_epoch='epoch')
            self.all_preds['test'] = torch.tensor([])
            self.all_targets['test'] = torch.tensor([])

        for metric_name in self.metrics.keys():
            metric = self.metric_mean("test", metric_name)
            trainer.logger.log_metrics(
                {f"metrics_epoch_mean/{metric_name}_test": metric}, step=trainer.current_epoch
            )
            pl_module.log(f"metrics_epoch_mean/per_batch_step/{metric_name}_test", metric)
            self.last_value["test"][metric_name] = metric


    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        # save_json({k: str(v) for k, v in self.last_value.items()}, join(final_dir, "metrics.json"))
        dict_str = {}
        for k1, v1 in self.last_value.items():
            dict_str[k1] = {}
            for k2, v2 in v1.items():
                dict_str[k1][k2] = str(v2)
        save_json(dict_str, join(final_dir, "metrics.json"))
        return self.last_value

    def save_hparams(self) -> Dict:
        res = {}
        for state in ["train", "val", "test"]:
            for metric_name in self.metrics.keys():
                res[f"{metric_name}_{state}"] = self.last_value[state][metric_name]
        return res
