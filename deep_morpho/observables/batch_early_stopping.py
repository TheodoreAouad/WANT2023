from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Tuple
from os.path import join
import pathlib

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping as LightningEarlyStopping
from pytorch_lightning.callbacks import Callback

from general.nn.observables import Observable
from general.utils import save_json, log_console

from ..models import BiSEBase


class ReasonCodeEnum(Enum):
    CHECK_FINITE = 0
    STOPPING_THRESHOLD = 1
    DIVERGENCE_THRESHOLD = 2
    STOP_IMPROVING = 3
    ALL_BISE_ACTIVATED = 4


class EarlyStoppingBase(Observable, ABC):
    def __init__(self, name: str = '', console_logger=None, verbose: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.reason_code = None
        self.stopped_batch = None
        self.stopped_epoch = None
        self.verbose = verbose
        self.console_logger = console_logger

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        return

    def on_validation_end(self, trainer, pl_module) -> None:
        return

    def on_train_batch_end(self, trainer: 'pl.Trainer', *args, **kwargs) -> None:
        return

    def get_save_dict(self):
        return {
            "stopped_batch": self.stopped_batch,
            "stopped_epoch": self.stopped_epoch,
            "stopping_reason": str(self.reason_code),
        }

    def save(self, save_path: str):
        final_dir = join(save_path, join(self.__class__.__name__, self.name))
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        to_save = self.get_save_dict()

        save_json(to_save, join(final_dir, "results.json"))
        return to_save

    def perform_early_stopping(self, trainer, pl_module):
        should_stop, reason, reason_code = self.run_early_stopping_check(trainer, pl_module)

        trainer.should_stop = trainer.should_stop or should_stop

        if should_stop:
            self.reason_code = reason_code
            self.stopped_epoch = trainer.current_epoch
            self.stopped_batch = trainer.global_step
            if self.verbose:
                log_console(f"epoch={self.stopped_epoch}, batch={self.stopped_batch}", reason)


    @abstractmethod
    def run_early_stopping_check(self, trainer, pl_module) -> Tuple[bool, str, ReasonCodeEnum]:
        should_stop: bool = None
        reason_code: ReasonCodeEnum = None
        reason: str = None
        return should_stop, reason, reason_code


class EarlyStoppingMetricBase(EarlyStoppingBase, LightningEarlyStopping, ABC):

    def on_save_checkpoint(self, *args, **kwargs):
        res = super().on_save_checkpoint(None, None, None)  # lightning necessity to avoid *args and kwargs. May change in future versions.
        res['stopped_batch'] = self.stopped_batch
        res['stopped_epoch'] = self.stopped_epoch
        return res

    def on_load_checkpoint(self, callback_state: Dict[str, Any]) -> None:
        super().on_load_checkpoint(callback_state)
        self.stopped_batch = callback_state['stopped_batch']

    def _run_early_stopping_check(self, trainer) -> None:
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        if self._should_skip_check(trainer):
            return None, None, None
        logs = trainer.logged_metrics
        # logs = trainer.callback_metrics

        if (
            trainer.fast_dev_run  # disable early_stopping with fast_dev_run
            or not self._validate_condition_metric(logs)  # short circuit if metric not present
        ):
            return

        current = logs.get(self.monitor)

        # when in dev debugging
        trainer.dev_debugger.track_early_stopping_history(self, current)

        should_stop, reason, reason_code = self._evalute_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)

        return should_stop, reason, reason_code



    def _evalute_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, str]:
        should_stop = False
        reason_str = None
        reason_code = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason_str = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
            reason_code = ReasonCodeEnum.CHECK_FINITE
        elif self.stopping_threshold is not None and (
            self.monitor_op(current, self.stopping_threshold) or current == self.stopping_threshold
        ):
            should_stop = True
            reason_str = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
            reason_code = ReasonCodeEnum.STOPPING_THRESHOLD
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason_str = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
            reason_code = ReasonCodeEnum.DIVERGENCE_THRESHOLD
        elif self.monitor_op(current - self.min_delta, self.best_score):
            should_stop = False
            reason_str = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason_str = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )
                reason_code = ReasonCodeEnum.STOP_IMPROVING

        return should_stop, reason_str, reason_code

    def run_early_stopping_check(self, trainer, pl_module) -> Tuple[bool, str, ReasonCodeEnum]:
        return self._run_early_stopping_check(trainer)

    def get_save_dict(self):
        to_save = super().get_save_dict()

        best_score = self.best_score
        if isinstance(best_score, torch.Tensor):
            best_score = best_score.item()

        to_save.update({
            "wait_count": self.wait_count,
            "patience": self.patience,
            "monitor": self.monitor,
            "best_score": best_score,
        })

        return to_save


class BatchEarlyStopping(EarlyStoppingMetricBase):
    def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module, *args, **kwargs) -> None:
        return self.perform_early_stopping(trainer, pl_module)


class EpochValEarlyStopping(EarlyStoppingMetricBase):
    def on_validation_end(self, trainer: 'pl.Trainer', pl_module, *args, **kwargs) -> None:
        return self.perform_early_stopping(trainer, pl_module)



class BatchActivatedEarlyStopping(EarlyStoppingBase):

    def __init__(self, patience=300, *args, **kwargs):
        """Callback to stop training when all BiSE are activated.

        Args:
            patience (int, optional): Number of batches to wait until we start the stopping. We avoid initial activations. Defaults to 300.
        """
        super().__init__(*args, **kwargs)
        self.patience = patience

    def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs: "STEP_OUTPUT", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.perform_early_stopping(trainer, pl_module)

    def run_early_stopping_check(self, trainer, pl_module) -> Tuple[bool, str, ReasonCodeEnum]:
        if trainer.global_step < self.patience:
            return

        # for bisel in pl_module.model.layers:
        #     for bise in bisel.bises:
        #         bise.update_learned_selems()
        #         if not bise.is_activated.all():
        #             return
        for bise_module in pl_module.model.modules():
            if not isinstance(bise_module, BiSEBase):
                continue
            if not bise_module.is_activated.all():
                return False, None, None

        return True, "All BiSE activated", ReasonCodeEnum.ALL_BISE_ACTIVATED

        # trainer.should_stop = True
        # if trainer.should_stop:
        #     self.stopped_epoch = trainer.current_epoch
        #     self.stopped_batch = trainer.global_step

    # def save(self, save_path: str):
    #     final_dir = join(save_path, join(self.__class__.__name__))
    #     pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

    #     to_save = {
    #         "stopped_batch": self.stopped_batch,
    #         "patience": self.patience,
    #     }

    #     save_json(to_save, join(final_dir, "results.json"))
    #     return to_save
