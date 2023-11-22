from typing import Tuple
import os
from os.path import join
import re
import pathlib

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from deep_morpho.dataloaders.custom_loaders import DataLoaderEpochTracker
from deep_morpho.morp_operations import ParallelMorpOperations
from general.utils import load_json, log_console
from .datamodule_base import DataModule

# def get_loader(batch_size, n_inputs, random_gen_fn, random_gen_args, morp_operation, device='cpu', **kwargs):
#     return DataLoader(
#         MultiRectDatasetGenerator(random_gen_fn, random_gen_args, morp_operation=morp_operation, device=device, n_inputs=n_inputs, ),
#         batch_size=batch_size,  **kwargs
#     )

# DEBUG
# N_DISKO = 0


class DiskorectDataset(DataModule, Dataset):
    """ Random Generation behavior: generate a new input at each call of __getitem__. If num_workers=0, each epoch is independent
    and outputs a new input. If num_workers>0, each epoch is the same and outputs the same inputs.
    """
    def __init__(
            self,
            random_gen_fn,
            random_gen_args,
            morp_operation: ParallelMorpOperations,
            device: str = "cpu",
            n_inputs: int = 1000,
            seed: int = None,
            max_generation_nb: int = 0,
            do_symetric_output: bool = False,
    ):
        self.random_gen_fn = random_gen_fn
        self.random_gen_args = random_gen_args
        self.device = device
        self.n_inputs = n_inputs
        self.morp_fn = morp_operation
        self.max_generation_nb = max_generation_nb
        self.do_symetric_output = do_symetric_output
        self.data = {}
        self.seed = seed
        # self.rng = np.random.default_rng(seed)
        # print(seed)
        self.rng = None
        self.epoch = None

        # DEBUG
        # self.nb_inp = 0
        # global N_DISKO
        # self.n_disko = N_DISKO
        # N_DISKO += 1
        # print(f"Reset. N_DISKO = {N_DISKO}, self.n_disko = {self.n_disko}, self.nb_inp = {self.nb_inp}")


    def get_rng(self):
        if self.rng is None:
            seed = self.seed if self.seed is not None else np.random.randint(0, 2 ** 32 - 1)

            # Different RNG for each worker and epoch. new_seed = seed * 2 ** epoch * 3 ** worker_id, ensuring that
            # each couple (worker, epoch) has a different seed.
            if self.epoch is not None:
                seed *= 2 ** self.epoch

            info = torch.utils.data.get_worker_info()
            if info is not None:
                seed *= 3 ** info.id

            self.rng = np.random.default_rng(seed)

        return self.rng

    def __getitem__(self, idx):
        if self.max_generation_nb == 0:
            return self.generate_input_target()

        idx = idx % self.max_generation_nb

        if idx not in self.data.keys():
            self.data[idx] = self.generate_input_target()

        return self.data[idx]

    def generate_input_target(self):
        # # DEBUG
        # with open("todelete/worker_id.txt", "a") as f:
        #     info = torch.utils.data.get_worker_info()
        #     f.write(f"{info.id}\n")

        self.rng = self.get_rng()
        input_ = self.random_gen_fn(rng_float=self.rng.random, rng_int=self.rng.integers, **self.random_gen_args,)
        target = self.morp_fn(input_)

        target = torch.tensor(target).float()
        input_ = torch.tensor(input_).float()

        if input_.ndim == 2:
            input_ = input_.unsqueeze(-1)  # Must have at least one channel

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)
        target = target.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        # DEBUG
        # pathlib.Path(f"todelete/{self.n_disko}/input_{self.nb_inp}.png").parent.mkdir(parents=True, exist_ok=True)
        # plt.imsave(f"todelete/{self.n_disko}/input_{self.nb_inp}.png", input_.squeeze().cpu().numpy())
        # self.nb_inp += 1
        # print(f"todelete/{self.n_disko}/input_{self.nb_inp}.png")

        if self.do_symetric_output:
            return 2 * input_ - 1, 2 * target - 1
        return input_, target

    def __len__(self):
        return self.n_inputs

    @classmethod
    def get_loader(
        cls,
        batch_size,
        n_inputs: int = "all",
        num_workers: int = 0,
        shuffle: bool = False,
        track_epoch: bool = True,
        **kwargs
    ):
        if track_epoch:
            loader = DataLoaderEpochTracker
        else:
            loader = DataLoader

        if n_inputs == 0:
            return loader([])
        return loader(
            cls(n_inputs=n_inputs, **kwargs), batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, )

    @classmethod
    def get_train_val_test_loader_from_experiment(cls, experiment: "ExperimentBase") -> Tuple[DataLoader, DataLoader, DataLoader]:
        args = experiment.args

        n_inputs_train = args[f"n_inputs{args.trainset_args_suffix}"]
        n_inputs_val = args[f"n_inputs{args.valset_args_suffix}"]
        n_inputs_test = args[f"n_inputs{args.testset_args_suffix}"]

        train_kwargs, val_kwargs, test_kwargs = cls.get_train_val_test_kwargs_pop_keys(
            experiment, keys=["n_inputs"]
        )

        train_loader = cls.get_loader(n_inputs=n_inputs_train, shuffle=True, batch_size=args["batch_size"], num_workers=args["num_workers"], **train_kwargs)
        val_loader = cls.get_loader(n_inputs=n_inputs_val, shuffle=False, batch_size=args["batch_size"], num_workers=args["num_workers"], **val_kwargs)
        test_loader = cls.get_loader(n_inputs=n_inputs_test, shuffle=False, batch_size=args["batch_size"], num_workers=args["num_workers"], **test_kwargs)

        return train_loader, val_loader, test_loader


class MultiRectDataset(Dataset):
    def __init__(
            self,
            inputs_path: str,
            targets_path: str,
            do_load_in_ram: bool = False,
            verbose: bool = True,
            n_inputs: int = None,
            logger=None,
    ):
        self.inputs_path = inputs_path
        self.targets_path = targets_path
        self.do_load_in_ram = do_load_in_ram
        self.verbose = verbose
        self.logger = logger
        self.n_inputs = n_inputs

        self.all_inputs_name = sorted(os.listdir(inputs_path), key=lambda x: int(re.findall(r'\d+', x)[0]))
        if self.n_inputs is not None:
            self.all_inputs_name = self.all_inputs_name[:self.n_inputs]

        if self.do_load_in_ram:
            self.all_inputs = []
            self.all_targets = []

            if verbose:
                log_console('Loading data in RAM...', logger=self.logger)
            for inpt in self.get_verbose_iterator(self.all_inputs_name):
                self.all_inputs.append(np.load(join(inputs_path, inpt)))
                self.all_targets.append(np.load(join(targets_path, inpt)))


    def get_verbose_iterator(self, iterator):
        if self.verbose:
            return tqdm(iterator)
        return iterator

    def __getitem__(self, idx):
        if self.do_load_in_ram:
            input_, target = self.all_inputs[idx], self.all_targets[idx]
        else:
            img_name = self.all_inputs_name[idx]
            input_, target = np.load(join(self.inputs_path, img_name)), np.load(join(self.targets_path, img_name))
        target = torch.tensor(target).float()
        input_ = torch.tensor(input_).unsqueeze(0).float()

        return input_, target

    def __len__(self):
        return len(self.all_inputs_name)

    @staticmethod
    def get_loader(batch_size, dataset_path, do_load_in_ram, morp_operation, logger=None, n_inputs=None, **kwargs):
        inputs_path = join(dataset_path, 'images')
        metadata = load_json(join(dataset_path, 'metadata.json'))
        targets_path = metadata["seqs"][morp_operation.get_saved_key()]['path_target']
        return DataLoader(
            MultiRectDataset(inputs_path, targets_path, do_load_in_ram=do_load_in_ram, verbose=True, logger=logger, n_inputs=n_inputs),
            batch_size=batch_size, **kwargs
        )
