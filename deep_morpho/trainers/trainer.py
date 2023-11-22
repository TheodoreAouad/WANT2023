from pytorch_lightning import Trainer as LightningTrainer

from general.nn.experiments.experiment_methods import ExperimentMethods


class Trainer(LightningTrainer, ExperimentMethods):
    pass


# TODO: move to general

