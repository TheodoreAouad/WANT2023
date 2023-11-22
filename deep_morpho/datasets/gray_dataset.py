from typing import Union, Tuple

import numpy as np
import torch

from deep_morpho.gray_scale import level_sets_from_gray, gray_from_level_sets
from .datamodule_base import DataModule


class GrayScaleDataset(DataModule):

    def __init__(
        self,
        n_gray_scale_values: str = "all",
        **kwargs,
    ) -> None:
        self.n_gray_scale_values = n_gray_scale_values


    def level_sets_from_gray(self, input_: torch.Tensor, target: torch.Tensor):
        input_ls, input_values = level_sets_from_gray(input_, n_values=self.n_gray_scale_values)
        target_ls, _ = level_sets_from_gray(target, input_values)
        input_ls.gray_values = input_values

        return input_ls, target_ls

    def gray_from_level_sets(self, ar: Union[np.ndarray, torch.Tensor], values: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return gray_from_level_sets(ar, values)

    @staticmethod
    def gray_batch_from_level_sets_batch(batch_tensor: torch.Tensor, values: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
        """Given an input batch of level set tensor, the corresponding values and the corresponding indexes, recovers
        the gray scale batch of tensor.
        Usually, `values` and `indexes` are attributes of the inputs (batch[0]).

        Args:
            batch_tensor (torch.Tensor): shape (sum_{batch size}{nb level sets} , 1 , W , L)
            indexes (torch.Tensor): shape (batch size + 1,)
            values (torch.Tensor): shape (sum_{batch size}{nb level sets},)

        Returns:
            torch.Tensor: shape (batch size , 1 , W , L)
        """

        final_tensor = []

        for idx in range(1, len(indexes)):
            idx1 = indexes[idx - 1]
            idx2 = indexes[idx]
            input_tensor = gray_from_level_sets(batch_tensor[idx1:idx2, 0], values=values[idx1:idx2])

            final_tensor.append(input_tensor)

        return torch.stack(final_tensor)[:, None, ...]

    @staticmethod
    def gray_from_level_sets_batch_idx(index: int, batch_tensor: torch.Tensor, values: torch.Tensor, indexes: torch.Tensor,) -> torch.Tensor:
        """Get a gray image from its index in the batch tensor. The index must be below batch_size.

        Args:
            index(int): index of the tensor inside the original batch tensor.
            batch_tensor (torch.Tensor): shape (sum_{batch size}{nb level sets} , 1 , W , L)
            indexes (torch.Tensor): shape (batch size + 1,)
            values (torch.Tensor): shape (sum_{batch size}{nb level sets},)

        Returns:
            torch.Tensor: shape (W , L)
        """
        return gray_from_level_sets(
            batch_tensor[indexes[index]:indexes[index + 1], 0],
            values=values[indexes[index]:indexes[index + 1]]
        )

    @staticmethod
    def get_relevent_tensors_idx(idx: int, batch: torch.Tensor, preds: torch.Tensor) -> Tuple[torch.Tensor]:
        """ Given the batch and the predictions, outputs all the useful tensors for a given idx.

        Args:
            idx (int): index of the original image
            batch (torch.Tensor): batch of level sets
            preds (torch.Tensor): preds of level sets

        Returns:
            Tuple[torch.Tensor]: reconstructed image, prediction, reconstructed target, original img, original target
        """
        img = GrayScaleDataset.gray_from_level_sets_batch_idx(
            index=idx,
            batch_tensor=batch[0],
            values=batch[0].gray_values,
            indexes=batch[0].indexes,
        )
        target = GrayScaleDataset.gray_from_level_sets_batch_idx(
            index=idx,
            batch_tensor=batch[1],
            values=batch[0].gray_values,
            indexes=batch[0].indexes,
        )
        pred = GrayScaleDataset.gray_from_level_sets_batch_idx(
            index=idx,
            batch_tensor=preds,
            values=batch[0].gray_values,
            indexes=batch[0].indexes,
        )

        original_img = batch[0].original[idx, 0]
        original_target = batch[1].original[idx, 0]

        return img, pred, target, original_img, original_target


    @staticmethod
    def get_relevent_tensors_batch(batch: torch.Tensor, preds: torch.Tensor) -> Tuple[torch.Tensor]:
        """ Given the batch and the predictions, outputs all the useful tensors for a given idx.

        Args:
            idx (int): index of the original image
            batch (torch.Tensor): batch of level sets
            preds (torch.Tensor): preds of level sets

        Returns:
            Tuple[torch.Tensor]: reconstructed image, prediction, reconstructed target, original img, original target
        """
        img = GrayScaleDataset.gray_batch_from_level_sets_batch(
            batch_tensor=batch[0],
            values=batch[0].gray_values,
            indexes=batch[0].indexes,
        )
        target = GrayScaleDataset.gray_batch_from_level_sets_batch(
            batch_tensor=batch[1],
            values=batch[0].gray_values,
            indexes=batch[0].indexes,
        )
        pred = GrayScaleDataset.gray_batch_from_level_sets_batch(
            batch_tensor=preds,
            values=batch[0].gray_values,
            indexes=batch[0].indexes,
        )

        original_img = batch[0].original
        original_target = batch[1].original

        return img, pred, target, original_img, original_target
