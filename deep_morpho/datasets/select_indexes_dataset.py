from typing import Union, Optional, List, Callable, Tuple
from random import shuffle

import numpy as np
from torch.utils.data.dataloader import DataLoader

from .datamodule_base import DataModule
# from deep_morpho.experiments.parser import Parser


class SelectIndexesDataset(DataModule):

    def __init__(
        self,
        n_inputs: Union[int, str] = "all",
        first_idx: int = 0,
        indexes: Optional[List[int]] = None,
        *args, **kwargs
    ) -> None:
        self.n_inputs = n_inputs
        self.first_idx = first_idx
        self.indexes = indexes

        if self.n_inputs != "all" or self.indexes is not None:
            if self.indexes is None:
                self.n_inputs = min(n_inputs, len(self.data))
                self.indexes = list(range(first_idx, min(first_idx + n_inputs, len(self))))

            self.data = self.data[self.indexes]
            self.targets = np.array(self.targets)[self.indexes]

        self.indexes = np.array(self.indexes)
        self.n_classes = len(np.unique(self.targets))

    @classmethod
    def get_loader(
        cls,
        batch_size,
        n_inputs: int = "all",
        num_workers: int = 0,
        shuffle: bool = False,
        **kwargs
    ):
        if n_inputs == 0:
            return DataLoader([])
        return DataLoader(
            cls(n_inputs=n_inputs, **kwargs), batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, )

    @classmethod
    def get_train_val_test_loader_from_experiment(cls, experiment: "ExperimentBase", ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        args: "Parser" = experiment.args

        n_inputs_train = args[f"n_inputs{args.trainset_args_suffix}"]
        n_inputs_val = args[f"n_inputs{args.valset_args_suffix}"]
        n_inputs_test = args[f"n_inputs{args.testset_args_suffix}"]

        train_kwargs, val_kwargs, test_kwargs = cls.get_train_val_test_kwargs_pop_keys(
            experiment, keys=["n_inputs", "indexes", "train", "shuffle", "first_idx"]
        )

        all_train_idxs = list(range(min(n_inputs_train + n_inputs_val, 60_000)))
        shuffle(all_train_idxs)

        train_idxes = all_train_idxs[:n_inputs_train]
        val_idxes = all_train_idxs[n_inputs_train:n_inputs_train + n_inputs_val]

        trainloader = cls.get_loader(indexes=train_idxes, train=True, shuffle=True, batch_size=args["batch_size"], num_workers=args["num_workers"], **train_kwargs)
        valloader = cls.get_loader(indexes=val_idxes, train=True, shuffle=False, batch_size=args["batch_size"], num_workers=args["num_workers"], **val_kwargs)
        testloader = cls.get_loader(first_idx=0, n_inputs=n_inputs_test, train=False, shuffle=False, batch_size=args["batch_size"], num_workers=args["num_workers"], **test_kwargs)

        return trainloader, valloader, testloader
