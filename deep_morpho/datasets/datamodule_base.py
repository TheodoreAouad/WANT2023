from typing import Tuple, List, Dict
from abc import ABC, abstractmethod

from torch.utils.data.dataloader import DataLoader

from general.nn.experiments.experiment_methods import ExperimentMethods


class DataModule(ExperimentMethods, ABC):
    @classmethod
    @abstractmethod
    def get_loader(cls, batch_size: int, num_workers: int, **kwargs) -> DataLoader:
        pass

    @classmethod
    @abstractmethod
    def get_train_val_test_loader_from_experiment(cls, experiment: "ExperimentBase", ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        pass

    @staticmethod
    def get_train_val_test_kwargs_pop_keys(experiment: "ExperimentBase", keys: List[str]) -> Tuple[Dict, Dict, Dict]:
        args = experiment.args

        train_kwargs = args.trainset_args()
        val_kwargs = args.valset_args()
        test_kwargs = args.testset_args()

        for kwargs in [train_kwargs, val_kwargs, test_kwargs]:
            for key in keys:
                if key in kwargs:
                    del kwargs[key]

        return train_kwargs, val_kwargs, test_kwargs
