from typing import Tuple, Any, Union
# import cv2
# import numpy as np
from PIL import Image

import torch
import numpy as np

from deep_morpho.morp_operations import ParallelMorpOperations
# from deep_morpho.gray_scale import level_sets_from_gray, gray_from_level_sets
from deep_morpho.tensor_with_attributes import TensorGray
from .gray_dataset import GrayScaleDataset
from .select_indexes_dataset import SelectIndexesDataset
# from general.utils import set_borders_to
from .datamodule_base import DataModule


def resize_image(img: np.ndarray, size: Tuple) -> np.ndarray:
    img_int8 = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return np.array(Image.fromarray(img_int8).resize((size[1], size[0]), Image.Resampling.BICUBIC))


class MnistBaseDataset(SelectIndexesDataset):

    def __init__(
        self,
        morp_operation: ParallelMorpOperations,
        threshold: float = 30,
        size=(50, 50),
        preprocessing=None,
        # indexes=None,
        # first_idx: int = 0,
        # n_inputs: int = "all",
        invert_input_proba: bool = 0,
        do_symetric_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.morp_operation = morp_operation
        self.threshold = threshold
        self.preprocessing = preprocessing
        self.size = size
        self.invert_input_proba = invert_input_proba
        self.do_symetric_output = do_symetric_output


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input_ = (resize_image(self.data[index].numpy(), self.size) >= (self.threshold))[..., None]

        if torch.rand(1) < self.invert_input_proba:
            input_ = 1 - input_

        target = torch.tensor(self.morp_operation(input_)).float()
        input_ = torch.tensor(input_).float()

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)
        target = target.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        if self.preprocessing is not None:
            input_ = self.preprocessing(input_)
            target = self.preprocessing(target)

        if self.do_symetric_output:
            return 2 * input_ - 1, 2 * target - 1

        return input_.float(), target.float()


class MnistGrayScaleBaseDataset(GrayScaleDataset):

    def __init__(
        self,
        morp_operation: ParallelMorpOperations,
        n_inputs: int = "all",
        n_gray_scale_values: str = "all",
        size=(50, 50),
        first_idx: int = 0,
        preprocessing=None,
        do_symetric_output: bool = False,
        **kwargs,
    ) -> None:
        GrayScaleDataset.__init__(self, n_gray_scale_values)
        self.morp_operation = morp_operation
        self.preprocessing = preprocessing
        self.n_inputs = n_inputs
        # self.n_gray_scale_values = n_gray_scale_values
        self.size = size
        self.do_symetric_output = do_symetric_output

        if n_inputs != "all":
            self.data = self.data[first_idx:n_inputs+first_idx]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # input_ = (
        #     cv2.resize(self.data[index].numpy(), (self.size[1], self.size[0]), interpolation=cv2.INTER_CUBIC)
        # )[..., None]
        input_ = (resize_image(self.data[index].numpy(), self.size))[..., None]

        # target = torch.tensor(self.morp_operation(input_)).float()
        # input_ = torch.tensor(input_).float()
        target = TensorGray(self.morp_operation(input_)).float()
        input_ = TensorGray(input_).float()

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)
        target = target.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        if self.preprocessing is not None:
            input_ = self.preprocessing(input_)
            target = self.preprocessing(target)

        original_input = input_.detach()
        original_target = target.detach()

        input_, target = self.level_sets_from_gray(input_, target)

        if self.do_symetric_output:
            gray_values = input_.gray_values
            input_ = 2 * input_ - 1
            target = 2 * target - 1
            input_.gray_values = gray_values

        input_.original = original_input
        target.original = original_target

        return input_.float(), target.float()

    def gray_from_level_sets(self, ar: Union[np.ndarray, torch.Tensor], values: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.do_symetric_output:
            return super().gray_from_level_sets((ar > 0).float(), values)
        return super().gray_from_level_sets(ar, values)
