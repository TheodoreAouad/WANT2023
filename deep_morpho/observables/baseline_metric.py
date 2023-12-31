import pathlib
from os.path import join

import torch

from general.nn.observables.observable import Observable
from general.utils import save_json



class InputAsPredMetric(Observable):
    """
    class used to calculate and track metrics in the tensorboard
    """
    def __init__(self, metrics, freq: int = 1):
        self.metrics = {k: v for k, v in metrics.items()}
        self.tb_steps = {metric: {} for metric in self.metrics.keys()}
        self.last_value = {}
        self.freq = freq
        self.freq_idx = 1
        # self.last_value = {k: 0 for k in metrics.keys()}

    def on_train_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        if self.freq_idx % self.freq != 0:
            self.freq_idx += 1
            return
        self.freq_idx += 1

        inputs, targets = batch
        if inputs.shape[1] < targets.shape[1]:
            inputs = torch.cat([inputs for _ in range(targets.shape[1])], axis=1)
        elif inputs.shape[1] > targets.shape[1]:
            inputs = torch.cat([inputs[:, 0:1, ...] for _ in range(targets.shape[1])], axis=1)
        self._calculate_and_log_metrics(trainer, pl_module, targets, inputs, state='train')
        # self._calculate_and_log_metrics(trainer, pl_module, targets, inputs.squeeze(), state='train')

    def _calculate_and_log_metrics(self, trainer, pl_module, targets, preds, state='train', batch_or_epoch='batch', suffix: str = ""):
        key = f'{state}{suffix}'
        for metric_name in self.metrics:
            metric = self.metrics[metric_name](targets, preds)
            self.last_value[f'{metric_name}{suffix}'] = metric
            # pl_module.log(f"mean_metrics_{batch_or_epoch}/{metric_name}/{state}", metric)
            if batch_or_epoch == 'batch':
                # step = self.tb_steps[metric_name].get(key, 0)
                step = trainer.global_step
            else:
                step = trainer.current_epoch

            trainer.logger.experiment.add_scalars(
                f"comparative/metrics_{batch_or_epoch}/{metric_name}{suffix}", {f"input_as_pred": metric}, step
            )

            if batch_or_epoch == 'batch':
                self.tb_steps[metric_name][key] = step + 1

            # f"metrics_multi_label_{batch_or_epoch}/{metric_name}/{state}", {f'label_{name_label}': metric}, step
            # trainer.logger.experiment.add_scalars(metric_name, {f'{metric_name}_{state}': metric})

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json({k: str(v) for k, v in self.last_value.items()}, join(final_dir, "baseline_metrics.json"))
        return self.last_value
