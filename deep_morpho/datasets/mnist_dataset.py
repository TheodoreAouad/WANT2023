from os.path import join
from typing import Tuple, Any, Callable, List
import warnings
from random import choice

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch

from deep_morpho.morp_operations import ParallelMorpOperations
from deep_morpho.tensor_with_attributes import TensorGray
from .mnist_base_dataset import MnistBaseDataset, MnistGrayScaleBaseDataset
from .select_indexes_dataset import SelectIndexesDataset
from .gray_to_channels_dataset import GrayToChannelDatasetBase
from .datamodule_base import DataModule

with open("deep_morpho/datasets/root_mnist_dir.txt", "r") as f:
    ROOT_MNIST_DIR = f.read()


transform_default = ToTensor()


class MNISTClassical(SelectIndexesDataset, MNIST):
    def __init__(
        self,
        root: str = ROOT_MNIST_DIR,
        preprocessing: Callable = None,
        train: bool = True,
        transform: Callable = transform_default,
        *args,
        **kwargs,
    ):
        MNIST.__init__(
            self,
            root=root,
            transform=transform,
            train=train,
        )
        self.preprocessing = preprocessing

        SelectIndexesDataset.__init__(self, *args, **kwargs)

    @property
    def processed_folder(self) -> str:
        return join(self.root, "processed")

    @property
    def raw_folder(self) -> str:
        return join(self.root, "raw")


class MnistMorphoDataset(MnistBaseDataset, MNIST):
    def __init__(
        self,
        morp_operation: ParallelMorpOperations,
        n_inputs: int = "all",
        threshold: float = 30,
        size=(50, 50),
        first_idx: int = 0,
        indexes=None,
        preprocessing=None,
        root: str = ROOT_MNIST_DIR,
        train: bool = True,
        invert_input_proba: bool = 0,
        do_symetric_output: bool = False,
        **kwargs,
    ) -> None:
        MNIST.__init__(self, root, train, **kwargs)
        MnistBaseDataset.__init__(
            self,
            morp_operation=morp_operation,
            n_inputs=n_inputs,
            threshold=threshold,
            size=size,
            indexes=indexes,
            first_idx=first_idx,
            preprocessing=preprocessing,
            invert_input_proba=invert_input_proba,
            do_symetric_output=do_symetric_output,
        )

    @property
    def processed_folder(self) -> str:
        return join(self.root, "processed")


class MnistGrayScaleDataset(MnistGrayScaleBaseDataset, MNIST):
    def __init__(
        self,
        morp_operation: ParallelMorpOperations,
        n_inputs: int = "all",
        n_gray_scale_values: str = "all",
        size=(50, 50),
        first_idx: int = 0,
        preprocessing=None,
        root: str = ROOT_MNIST_DIR,
        train: bool = True,
        do_symetric_output: bool = False,
        **kwargs,
    ) -> None:
        MNIST.__init__(self, root, train, **kwargs)
        MnistGrayScaleBaseDataset.__init__(
            self,
            morp_operation=morp_operation,
            n_inputs=n_inputs,
            n_gray_scale_values=n_gray_scale_values,
            size=size,
            first_idx=first_idx,
            preprocessing=preprocessing,
            do_symetric_output=do_symetric_output,
        )

    # @classmethod
    # def get_loader(
    #     cls, batch_size, n_inputs, morp_operation, train, first_idx=0, size=(50, 50),
    #     do_symetric_output=False, preprocessing=None, n_gray_scale_values="all",
    #     num_workers=0,
    # **kwargs):
    #     if n_inputs == 0:
    #         return DataLoader([])
    #     return DataLoader(
    #         cls(
    #             morp_operation=morp_operation, n_inputs=n_inputs, first_idx=first_idx,
    #             train=train, preprocessing=preprocessing, size=size,
    #             do_symetric_output=do_symetric_output, n_gray_scale_values=n_gray_scale_values,
    #         ), batch_size=batch_size, collate_fn=collate_fn_gray_scale, num_workers=num_workers)

    # @classmethod
    # def get_train_val_test_loader(cls, n_inputs_train, n_inputs_val, n_inputs_test, *args, **kwargs):
    #     for key in ['n_inputs', 'first_idx', 'train', 'shuffle']:
    #         if key in kwargs:
    #             del kwargs[key]

    #     # trainloader = cls.get_loader(first_idx=0, n_inputs=n_inputs_train, train=True, shuffle=True, *args, **kwargs)
    #     # valloader = cls.get_loader(first_idx=0, n_inputs=n_inputs_val, train=False, shuffle=False, *args, **kwargs)
    #     # testloader = cls.get_loader(first_idx=n_inputs_val, n_inputs=n_inputs_test, train=False, shuffle=False, *args, **kwargs)
    #     # return trainloader, valloader, testloader

    #     all_train_idxs = list(range(min(n_inputs_train + n_inputs_val, 60_000)))
    #     shuffle(all_train_idxs)

    #     train_idxes = all_train_idxs[:n_inputs_train]
    #     val_idxes = all_train_idxs[n_inputs_train:n_inputs_train + n_inputs_val]

    #     for key in ["first_idx", "n_inputs", "indexes", "train"]:
    #         if key in kwargs:
    #             del kwargs[key]

    #     trainloader = cls.get_loader(indexes=train_idxes, train=True, shuffle=True, *args, **kwargs)
    #     valloader = cls.get_loader(indexes=val_idxes, train=True, shuffle=False, *args, **kwargs)
    #     testloader = cls.get_loader(first_idx=0, n_inputs=n_inputs_test, train=False, shuffle=False, *args, **kwargs)
    #     return trainloader, valloader, testloader

    @property
    def processed_folder(self) -> str:
        return join(self.root, "processed")


class MnistClassifDataset(MNIST, DataModule):
    def __init__(
        self,
        root: str = ROOT_MNIST_DIR,
        threshold: float = 30,
        first_idx: int = 0,
        n_inputs: int = "all",
        indexes: List[int] = None,
        train: bool = True,
        invert_input_proba: float = 0,
        preprocessing=None,
        do_symetric_output: bool = False,
        size=None,
        **kwargs,
    ) -> None:
        super().__init__(root=root, train=train, **kwargs)
        self.n_inputs = n_inputs
        self.first_idx = first_idx
        self.threshold = threshold
        self.invert_input_proba = invert_input_proba
        self.n_classes = 10
        self.preprocessing = preprocessing
        self.do_symetric_output = do_symetric_output

        warnings.warn("Size not used yet on classif.")
        self.size = size  # WARNING: not used

        if n_inputs != "all":
            if indexes is None:
                n_inputs = min(n_inputs, len(self.data))
                indexes = list(range(first_idx, first_idx + n_inputs))

            self.data = self.data[indexes]
            self.targets = self.targets[indexes]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input_ = (self.data[index].numpy() >= (self.threshold))[..., None]
        target_int = int(self.targets[index])
        target = torch.zeros(10)
        target[target_int] = 1

        if torch.rand(1) < self.invert_input_proba:
            input_ = 1 - input_

        input_ = torch.tensor(input_).float()

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        if self.transform is not None:
            input_ = self.transform(input_)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.do_symetric_output:
            input_ = 2 * input_ - 1
            target = 2 * target - 1

        return input_, target

    @property
    def processed_folder(self) -> str:
        return join(self.root, "processed")


class MnistClassifChannelDataset(GrayToChannelDatasetBase, MNIST):
    def __init__(self, root: str = ROOT_MNIST_DIR, preprocessing: Callable = None, train: bool = True, *args, **kwargs):
        MNIST.__init__(
            self,
            root=root,
            transform=lambda x: torch.tensor(x),
            train=train,
            download=True,
        )
        self.preprocessing = preprocessing

        GrayToChannelDatasetBase.__init__(self, img=torch.tensor(choice(self.data).unsqueeze(0)), *args, **kwargs)

    @property
    def processed_folder(self) -> str:
        return join(self.root, "processed")

    @property
    def raw_folder(self) -> str:
        return join(self.root, "MNIST", "raw")

    def __getitem__(self, index: int) -> Tuple[TensorGray, torch.Tensor]:
        input_, target = self.data[index], self.targets[index]
        return self._transform_sample(input_[..., None], target)
