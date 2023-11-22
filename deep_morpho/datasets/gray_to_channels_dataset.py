from abc import ABC, abstractmethod
from typing import Tuple, Dict, Callable, List
from random import shuffle

import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader

from deep_morpho.gray_scale import level_sets_from_gray, gray_from_level_sets, undersample
from deep_morpho.tensor_with_attributes import TensorGray
from .select_indexes_dataset import SelectIndexesDataset


class LevelsetValuesHandler(ABC):

    def __init__(self, img: torch.Tensor, *args, **kwargs):
        """
        Args:
            img (torch.Tensor): shape (n_channels, *img_shape)
        """
        self.levelset_values = self.init_levelset_values(img=img, *args, **kwargs)

    @abstractmethod
    def init_levelset_values(self, img: torch.Tensor, *args, **kwargs):
        return


class LevelsetValuesManual(LevelsetValuesHandler):
    def init_levelset_values(self, values: torch.Tensor, *args, **kwargs):
        return values


class LevelsetValuesDefault(LevelsetValuesHandler):
    def init_levelset_values(self, img: torch.Tensor, *args, **kwargs):
        return torch.tensor([img.unique() for _ in range(img.shape[0])])


class LevelsetValuesEqualIndex(LevelsetValuesHandler):
    def __init__(self, n_values: int, *args, **kwargs):
        super().__init__(*args, n_values=n_values, **kwargs)

    def init_levelset_values(self, img: torch.Tensor, n_values: int, *args, **kwargs):
        levelsets = torch.zeros(img.shape[0], n_values)
        for chan in range(img.shape[0]):
            values, count = img.unique(return_counts=True)
            values, count = values[1:-1], count[1:-1]

            # Python 3.7 compatibility
            values2 = []
            for v, c in zip(values, count):
                values2 += [v for _ in range(c)]
            values = torch.tensor(sorted(values2))

            # values = torch.tensor(sorted(sum([[v for _ in range(c)] for v, c in zip(values, count)], start=[])))  # repeat values that occur multiple times

            levelsets[chan] = values[undersample(0, len(values) - 1, n_values)]
        return levelsets


class GrayToChannelDatasetBase(SelectIndexesDataset):
    def __init__(
        self,
        img: torch.Tensor,
        levelset_handler_mode: LevelsetValuesHandler = LevelsetValuesEqualIndex,
        levelset_handler_args: Dict = {"n_values": 10},
        do_symetric_output: bool = False,
        apply_one_hot_target: bool = True,
        *args, **kwargs
    ):
        self.levelset_handler_mode = levelset_handler_mode
        self.levelset_handler_args = levelset_handler_args

        self.levelset_handler_args["img"] = img
        self.levelset_handler = levelset_handler_mode(**levelset_handler_args)
        self.do_symetric_output = do_symetric_output
        self.apply_one_hot_target = apply_one_hot_target

        assert hasattr(self, "data"), "Must have data attribute."
        super().__init__(*args, **kwargs)


    @property
    def in_channels(self):
        return np.prod(self.levelset_values.shape)

    @property
    def levelset_values(self):
        return self.levelset_handler.levelset_values

    def from_gray_to_channels(self, img: torch.Tensor) -> torch.Tensor:
        return self._from_gray_to_channels(img, self.levelset_values)

    def from_channels_to_gray(self, img_channels: torch.Tensor) -> torch.Tensor:
        return self._from_channels_to_gray(img_channels, self.levelset_values)

    @staticmethod
    def _from_gray_to_channels(img: torch.Tensor, levelset_values: torch.Tensor) -> torch.Tensor:
        """ Given a gray scale image with multiple channels, outputs a binary image with level sets as channels.

        Args:
            img (torch.Tensor): shape (channels, width, length), gray scale image
            levelset_values (np.ndarray): shape (channels, n_values), level set values for each channel

        Returns:
            torch.Tensor: shape (channels * n_values, width, length), binary image with channels as level sets
        """
        all_binary_imgs = []
        values_device = levelset_values.to(img.device)

        for chan in range(img.shape[0]):
            bin_img, _ = level_sets_from_gray(img[chan], values_device[chan])
            all_binary_imgs.append(bin_img)

        return torch.cat(all_binary_imgs, axis=0)

    @staticmethod
    def _from_channels_to_gray(img_channels: torch.Tensor, levelset_values: torch.Tensor) -> torch.Tensor:
        """ Given a binary image with level sets as channels, gives the gray scale image with multiple channels.

        Args:
            img_channels (torch.Tensor):  shape (channels * n_values, width, length), binary image with
                                            channels as level sets
            levelset_values (np.ndarray): shape (channels, n_values), level set values for each channel

        Returns:
            torch.Tensor: shape (channels, width, length), gray scale image
        """

        n_channels, n_values = levelset_values.shape
        values_device = levelset_values.to(img_channels.device)
        gray_img = torch.zeros((n_channels,) + img_channels.shape[1:])

        for chan in range(n_channels):
            gray_img[chan] = gray_from_level_sets(
                img_channels[chan * n_values:(chan + 1) * n_values], values_device[chan]
            )

        return gray_img

    def __getitem__(self, index: int) -> Tuple[TensorGray, torch.Tensor]:
        input_, target = self.data[index], self.targets[index]
        return self._transform_sample(input_, target)


    def _transform_sample(self, input_: torch.Tensor, target: torch.Tensor) -> Tuple[TensorGray, torch.Tensor]:
        if isinstance(input_, torch.Tensor):
            input_ = input_.clone().detach().float()
        else:
            input_ = torch.tensor(input_).float()
        # input_ = torch.tensor(input_).float()
        original_img = input_ + 0

        if self.apply_one_hot_target:
            target_int = target
            target = torch.zeros(10)
            target[target_int] = 1

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        if self.preprocessing is not None:
            input_ = self.preprocessing(input_)

        input_ = TensorGray(self.from_gray_to_channels(input_))

        if self.do_symetric_output:
            input_ = 2 * input_ - 1
            if self.apply_one_hot_target:
                target = 2 * target - 1

        input_.original = original_img

        return input_, target

    @classmethod
    def get_loader(
        cls,
        batch_size,
        train,
        levelset_handler_mode: LevelsetValuesHandler = LevelsetValuesEqualIndex,
        levelset_handler_args: Dict = {"n_values": 10},
        preprocessing: Callable = None,
        indexes: List[int] = None,
        first_idx: int = 0,
        n_inputs: int = "all",
        do_symetric_output=False,
        num_workers: int = 0,
        shuffle: bool = False,
        **kwargs
    ):
        if n_inputs == 0:
            return DataLoader([])
        return DataLoader(
            cls(
                n_inputs=n_inputs, first_idx=first_idx, indexes=indexes,
                train=train, preprocessing=preprocessing, do_symetric_output=do_symetric_output,
                levelset_handler_mode=levelset_handler_mode, levelset_handler_args=levelset_handler_args,
            ), batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, )

    # @classmethod
    # def get_train_val_test_loader(cls, n_inputs_train, n_inputs_val, n_inputs_test, *args, **kwargs):
    #     all_train_idxs = list(range(min(n_inputs_train + n_inputs_val, 60_000)))
    #     shuffle(all_train_idxs)

    #     for key in ["indexes", "train", "shuffle"]:
    #         if key in kwargs:
    #             del kwargs[key]

    #     train_idxes = all_train_idxs[:n_inputs_train]
    #     val_idxes = all_train_idxs[n_inputs_train:n_inputs_train + n_inputs_val]

    #     trainloader = cls.get_loader(indexes=train_idxes, train=True, shuffle=True, *args, **kwargs)
    #     kwargs.update({
    #         "levelset_handler_mode": LevelsetValuesManual,
    #         "levelset_handler_args": {"values": trainloader.dataset.levelset_values},
    #     })
    #     valloader = cls.get_loader(indexes=val_idxes, train=True, shuffle=False, *args, **kwargs)
    #     testloader = cls.get_loader(first_idx=0, n_inputs=n_inputs_test, train=False, shuffle=False, *args, **kwargs)
    #     return trainloader, valloader, testloader

    @classmethod
    def get_train_val_test_loader_from_experiment(cls, experiment: "ExperimentBase") -> Tuple[DataLoader, DataLoader, DataLoader]:
        args: Parser = experiment.args

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

        trainloader = cls.get_loader(
            indexes=train_idxes, train=True, shuffle=True,
            batch_size=args["batch_size"], num_workers=args["num_workers"], **train_kwargs
        )
        for kwargs in [val_kwargs, test_kwargs]:
            kwargs.update({
                "levelset_handler_mode": LevelsetValuesManual,
                "levelset_handler_args": {"values": trainloader.dataset.levelset_values},
            })
        valloader = cls.get_loader(
            indexes=val_idxes, train=True, shuffle=False,
            batch_size=args["batch_size"], num_workers=args["num_workers"], **val_kwargs
        )
        testloader = cls.get_loader(
            first_idx=0, n_inputs=n_inputs_test, train=False, shuffle=False,
            batch_size=args["batch_size"], num_workers=args["num_workers"], **test_kwargs
        )

        return trainloader, valloader, testloader


    def from_channels_to_gray_numpy(self, img_channels: torch.Tensor) -> np.ndarray:
        return self.from_channels_to_gray(img_channels).numpy().transpose(1, 2, 0).astype(int)
