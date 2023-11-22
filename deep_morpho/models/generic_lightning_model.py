# import inspect
from typing import Dict, Callable, List
from functools import reduce


from general.nn.pytorch_lightning_module.obs_lightning_module import NetLightning
from general.nn.observables import Observable


class GenericLightningModel(NetLightning):

    def __init__(
        self,
        model_args: Dict,
        learning_rate: float,
        loss: Callable,
        optimizer: Callable,
        optimizer_args: Dict = {},
        observables: List[Observable] = [],
        reduce_loss_fn: Callable = lambda x: reduce(lambda a, b: a + b, x),
        **kwargs
    ):
        super().__init__(
            model=self.model_class(**model_args),
            learning_rate=learning_rate,
            loss=loss,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            observables=observables,
            reduce_loss_fn=reduce_loss_fn,
            **kwargs
        )

        self.save_hyperparameters(ignore="observables")
        self.hparams["model_type"] = self.__class__.__name__


    @classmethod
    def default_args(cls) -> Dict[str, dict]:
        """Return the default arguments of the model, in the format of argparse.ArgumentParser"""
        default_args = super().default_args()
        if hasattr(cls, "model_class"):
            default_args["model_args"] = {"default": cls.model_class.default_args()}
        return default_args

    @classmethod
    def select(cls, name: str, **kwargs):
        name = name.lower()
        if not name.startswith("lightning"):
            name = "lightning" + name
        return super().select(name, **kwargs)


    @classmethod
    def get_model_from_experiment(cls, experiment: "deep_morpho.experiment.ExperimentBase"):
        args = experiment.args

        model_args = args.model_args()

        model = cls(
            model_args=model_args,
            learning_rate=args["learning_rate"],
            loss=args["loss"],
            optimizer=args["optimizer"],
            optimizer_args=args["optimizer_args"],
            observables=experiment.observables,
        )
        model.to(experiment.device)
        return model

    @classmethod
    def is_child(cls, name: str) -> bool:
        name = name.lower()
        if not name.startswith("lightning"):
            name = "lightning" + name
        return name in cls.listing()
