import inspect
from typing import Any, List, Optional, Callable, Dict, Union, Tuple, IO
from functools import reduce

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.core.saving import load_hparams_from_tags_csv, load_hparams_from_yaml

from ..observables.observable import Observable
from ..experiments.experiment_methods import ExperimentMethods
from ..loss import LossHandler
from general.nn.extend_signature_and_forward import extend_signature_and_forward


class ObsLightningModule(LightningModule, ExperimentMethods):

    def __init__(self, observables=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observables: Optional[List[Observable]] = observables

    def training_step(self, batch: Any, batch_idx: int):
        outputs, preds = self.obs_training_step(batch, batch_idx)

        for obs in self.observables:
            if isinstance(obs, Observable):
                obs.on_train_batch_end_with_preds(
                    self.trainer,
                    self.trainer.lightning_module,
                    outputs,
                    batch,
                    batch_idx,
                    preds
                )
        return outputs

    def validation_step(self, batch: Any, batch_idx: int):
        outputs, preds = self.obs_validation_step(batch, batch_idx)

        for obs in self.observables:
            if isinstance(obs, Observable):
                obs.on_validation_batch_end_with_preds(
                    self.trainer,
                    self.trainer.lightning_module,
                    outputs,
                    batch,
                    batch_idx,
                    preds
                )
        return outputs

    def test_step(self, batch: Any, batch_idx: int):
        outputs, preds = self.obs_test_step(batch, batch_idx)

        for obs in self.observables:
            if isinstance(obs, Observable):
                obs.on_test_batch_end_with_preds(
                    self.trainer,
                    self.trainer.lightning_module,
                    outputs,
                    batch,
                    batch_idx,
                    preds
                )

        return outputs

    def test_epoch_end(self, outputs: EPOCH_OUTPUT):
        self.obs_test_epoch_end(outputs)
        for obs in self.observables:
            obs.on_test_epoch_end(self.trainer, self.trainer.lightning_module)

    def obs_training_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def obs_validation_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def obs_test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def obs_test_epoch_end(self, outputs: EPOCH_OUTPUT):
        pass


class NetLightning(ObsLightningModule):
    def __init__(
            self,
            model: "nn.Module",
            learning_rate: float,
            # loss_handler: Union[Callable, Tuple[Callable, Dict], Dict[str, Union[Callable, Tuple[Callable, Dict]]]],
            loss: Union[Callable, LossHandler],
            optimizer: Callable,
            optimizer_args: Dict = {},
            observables: Optional[List[Observable]] = [],
            reduce_loss_fn: Callable = lambda x: reduce(lambda a, b: a + b, x),
            **kwargs
    ):
        """ Basic class for a neural network framework using pytorch lightning, with the predictions available
        for the callbacs.

        Args:
            model (nn.Module): the module we want to train
            learning_rate (float): learning rate of the optimizer
            loss (callable): the loss function we want to use
            optimizer (Callable): the constructor of the optimizer used to upgrade the parameters using their gradient
            optimizer_args (dict): the init arguments of the constructor of the optimizer, minus the learning rate
            observables (list): list of observables that we want to follow
            reduce_loss_fn (Callable): if we have multiple loss, tells how we want to aggregate all losses
        """

        super().__init__(observables, **kwargs)
        self.model = model
        self.learning_rate = learning_rate
        # self.loss = self.configure_loss(loss)

        self.loss = loss
        self.loss_forward = extend_signature_and_forward(self.loss)

        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        # self.reduce_loss_fn = reduce_loss_fn
        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    # def configure_loss(self, loss):
    #     if isinstance(loss, dict):
    #         return {k: self.configure_loss(v) for (k, v) in loss.items()}

    #     if isinstance(loss, tuple):
    #         return loss[0](model=self.model, **loss[1])

    #     return loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate, **self.optimizer_args)

    def obs_training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)

        outputs = self.compute_loss(state="/training", ypred=predictions, ytrue=y)

        return outputs, predictions

    def obs_validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)

        outputs = self.compute_loss(state="/validation", ypred=predictions, ytrue=y)

        return outputs, predictions

    def obs_test_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)

        outputs = self.compute_loss(state="/test", ypred=predictions, ytrue=y)

        return outputs, predictions

    def compute_loss_value(self, ypred, ytrue):
        """Outputs the 1d loss value for the given example.
        """
        return self.compute_loss(ypred, ytrue, do_log=False)["loss"].item()

    def compute_loss(self, ypred, ytrue, state="", do_log=True):
        """Computes total loss for each component of the loss.
        """
        values = self.loss_forward(ypred, ytrue, pl_module=self)

        if do_log:
            if isinstance(values, dict):
                for key, value in values.items():
                    if hasattr(value, "item"):
                        self.log(f"loss{state}/{key}", value.item())  # put .item() to avoid memory leak
            else:
                self.log(f"loss{state}", values.item())

        return values
        # values = {}
        # assert not ypred.isnan().any(), "NaN in prediction"
        # if isinstance(self.loss, dict):
        #     for key, loss_fn in self.loss.items():
        #         values[key] = loss_fn(ypred, ytrue)

        #     if "loss" in self.loss.keys():
        #         i = 0
        #         while f"loss_{i}" in self.loss.keys():
        #             i += 1
        #         values[f"loss_{i}"] = values["loss"]

        #     values["loss"] = self.reduce_loss_fn(values.values())

        # else:
        #     values["loss"] = self.loss(ypred, ytrue)

        # grad_values = {}
        # for key, value in values.items():
        #     if do_log:
        #         self.log(f"loss{state}/{key}", value.item())  # put .item() to avoid memory leak
        #     if key == "loss":
        #         continue

        #     if value.requires_grad:
        #         value.retain_grad()  # see graph of each loss term
        #         grad_values[f'{key}_grad'] = value.grad
        #         values[key] = values[key].detach()

        # values['grads'] = grad_values

        # return values

    @classmethod
    def load_from_checkpoint(cls, path: str, model_kwargs: Dict = {}, *args, **kwargs):
        """If model info in the checkpoint, load the right model."""
        checkpoint = torch.load(path)

        if "hyper_parameters" not in checkpoint.keys():
            return super().load_from_checkpoint(path, *args, **kwargs)

        if "model_type" not in checkpoint["hyper_parameters"].keys():
            return super().load_from_checkpoint(path, *args, **kwargs)

        model_type = checkpoint["hyper_parameters"]["model_type"].lower()

        return cls.select(model_type).load_from_checkpoint_ignore_keys(
            path, ignore_keys=[("hyper_parameters", "model_type")], model_kwargs=model_kwargs,
            *args, **kwargs
        )

    @classmethod
    def load_from_checkpoint_ignore_keys(
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        ignore_keys: List[Union[Tuple[str], str]] = [],
        model_kwargs: Dict = {},
        **kwargs,
    ):
        """The same as load_from_checkpoint of lightning, except that deletes some unwanted arguments of checkpoints.
        If a list of tuple is given, the last argument of the dict tree is deleted.
        Ex:
        >>> ignore_keys = [("hyper_parameters", "model_type")]
        then the key "model_type" is deleted from the dict checkpoint["hyper_parameters"]"""
        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

        # Delete key to ignore.
        for to_ignore in ignore_keys:
            if isinstance(to_ignore, tuple):
                v1 = checkpoint[to_ignore[0]]
                for key in to_ignore[1:-1]:
                    v1 = v1[key]
                del v1[to_ignore[-1]]
            else:
                del checkpoint[to_ignore]

        if hparams_file is not None:
            extension = hparams_file.split('.')[-1]
            if extension.lower() == 'csv':
                hparams = load_hparams_from_tags_csv(hparams_file)
            elif extension.lower() in ('yml', 'yaml'):
                hparams = load_hparams_from_yaml(hparams_file)
            else:
                raise ValueError('.csv, .yml or .yaml is required for `hparams_file`')

            hparams['on_gpu'] = False

            # overwrite hparams by the given file
            checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

        # for past checkpoint need to add the new key
        if cls.CHECKPOINT_HYPER_PARAMS_KEY not in checkpoint:
            checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = {}
        # override the hparams with values that were passed in
        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)
        checkpoint["hyper_parameters"]["model_args"].update(model_kwargs)

        model = cls._load_model_state(checkpoint, strict=strict, **kwargs)
        return model
