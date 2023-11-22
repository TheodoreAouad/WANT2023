from typing import List, Callable
from os.path import join
import os

import torch
from pytorch_lightning.loggers import TensorBoardLogger

from general.nn.observables import Observable
from deep_morpho.trainers import Trainer
from deep_morpho.datasets import DataModule
from deep_morpho.models import GenericLightningModel
from .parser import Parser
from deep_morpho.utils import set_seed
from .context import Task
from general.utils import log_console, save_yaml
from .load_model_fn import default_load_model_fn
from .load_datamodule_fn import default_load_datamodule_fn
from .load_observables_fn import default_load_observables_fn
from .enforcers import ArgsEnforcer
from general.nn.experiments.experiment_methods import ExperimentMethods



class ExperimentBase(ExperimentMethods):
    """Base class for any experiment. Can be used for experiment. Most of the experiments are children of the class.
    The children:
        - can have their own load_model_fn, load_datamodule_fn, load_observables_fn
        - can have their own get_experiment_name
        - can have their own specific setup, or run
    """
    trainer_class = Trainer

    def __init__(
        self,
        args: Parser,
        dest_dir: str,
        load_model_fn: Callable = default_load_model_fn,
        load_datamodule_fn: Callable = default_load_datamodule_fn,
        load_observables_fn: Callable = default_load_observables_fn,
        args_enforcers: List[ArgsEnforcer] = [],
    ):
        self.args = args
        # self.tb_logger = tb_logger

        self.load_model_fn = load_model_fn
        self.load_datamodule_fn = load_datamodule_fn
        self.load_observables_fn = load_observables_fn
        self.args_enforcers = self.args.get("args_enforcers", []) + args_enforcers

        name: str = self.get_experiment_name()
        self.tb_logger = TensorBoardLogger(dest_dir, name=name, default_hp_metric=False)
        self.console_logger = None

        self.device = None

        self.trainer: self.trainer_class = None
        self.observables: list = None
        self.callbacks: list = None
        self.trainloader, self.valloader, self.testloader = None, None, None
        self.input_sample = None


        self.metric_float_obs = None
        self.metric_binary_obs = None
        self.model_checkpoint_obs = None

        self._check_args()

    def enforce_args(self):
        for enforcer in self.args_enforcers:
            enforcer.enforce(self)

    @property
    def log_dir(self) -> str:
        return self.tb_logger.log_dir

    def _check_args(self) -> None:
        pass

    def get_experiment_name(self) -> str:
        return join(self.args["dataset"].replace("dataset", ""), self.args["model"])

    def load_model(self) -> GenericLightningModel:
        self.model = self.load_model_fn(self)

    def load_datamodule(self) -> DataModule:
        self.trainloader, self.valloader, self.testloader = self.load_datamodule_fn(self)

    def load_observables(self):
        (
            self.observables,
            self.callbacks,
            self.metric_float_obs,
            self.metric_binary_obs,
            self.model_checkpoint_obs,
                                        ) = self.load_observables_fn(self)

        if "observables" in self.args:
            self.observables += self.args["observables"]

    def train(self):
        self.trainer.fit(self.model, self.trainloader, self.valloader)

    def load_final_weights(self):
        if os.path.exists(self.model_checkpoint_obs.best_model_path):
            path_weight = self.model_checkpoint_obs.best_model_path
        else:
            path_weight = self.model_checkpoint_obs.last_model_path

        self.log_console(f"Loading best model from {path_weight}")
        self.model.load_state_dict(torch.load(path_weight)["state_dict"])

    def test(self):
        self.trainer.test(self.model, self.testloader)

    def save(self):
        metric_dict = {}
        for obs in self.observables:
            if isinstance(obs, Observable):
                metric_dict.update(obs.save_hparams())

        args_str = {}
        for k, v in self.args.items():
            args_str[k] = f"{v}"
        self.tb_logger.log_hyperparams(args_str, metric_dict)

        for observable in self.observables:
            observable.save(join(self.log_dir, 'observables'))

    def log_console(self, *args, **kwargs):
        return log_console(*args, **kwargs, logger=self.console_logger)

    def log_tensorboard(self):
        with Task("Logging model to Tensorboard", self.console_logger):
            self.tb_logger.experiment.add_graph(self.model, self.input_sample[0].unsqueeze(0).to(self.device))

        with Task("Logging hyperparameters to Tensorboard", self.console_logger):
            hyperparam_str = ""
            for k, v in self.args.items():
                hyperparam_str += f"**{k}**: {v}  \n"
            hyperparam_str = hyperparam_str[:-2]

            self.tb_logger.experiment.add_text("hyperparams", hyperparam_str, global_step=0)


    def log_binarizable_params(self):
        nb_binary = self.model.model.numel_binary() if hasattr(self.model.model, "numel_binary") else 0
        nb_float = (
            self.model.model.numel_float() if hasattr(self.model.model, "numel_float")
            else sum([p.numel() for p in self.model.model.parameters() if p.requires_grad])
        )
        self.args["float_params"] = nb_float
        self.args["binary_params"] = nb_binary
        self.log_console(
            f"Binarizable parameters: {nb_binary} / {nb_float} = {nb_binary/nb_float:.4f}"
        )


    def setup(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.args["gpus.trainer"] = 1 if self.device == torch.device("cuda") else 0

        self.args["seed"] = set_seed(self.args['batch_seed'])
        self.log_console(f"Seed: {self.args['seed']}")

        self.enforce_args()

        with Task("Loading Data", self.console_logger):
            self.load_datamodule()

        self.input_sample = next(iter(self.trainloader))[0]

        with Task("Loading Observables", self.console_logger):
            self.load_observables()

        with Task("Loading Model", self.console_logger):
            self.load_model()

        self.log_console(f"Model: {self.model}")
        self.log_binarizable_params()

        if "callbacks.trainer" in self.args:
            del self.args["callbacks.trainer"]

        if "logger.trainer" in self.args:
            del self.args["logger.trainer"]

        self.trainer = self.trainer_class(
            callbacks=self.observables.copy() + self.callbacks.copy(),
            logger=self.tb_logger,
            **self.args.trainer_args()
        )

    def run(self):
        self.setup()

        save_yaml(self.args, join(self.log_dir, "args.yaml"))

        self.log_tensorboard()


        with Task("Training", self.console_logger):
            self.train()

        with Task("Loading best weights", self.console_logger):
            self.load_final_weights()

        with Task("Testing", self.console_logger):
            self.test()

        with Task("Saving", self.console_logger):
            self.save()
