from typing import Union
from types import ModuleType

import numpy as np
import torch
from tqdm import tqdm


class ProjectionConstantSet:
    operation_code = {"erosion": 0, "dilation": 1}  # TODO: be coherent with bisebase attribute operation_code

    def __init__(self, weights: Union[np.ndarray, torch.tensor], bias: Union[np.ndarray, torch.tensor],) -> None:
        self.weights = weights
        self.bias = bias

        self.final_dist_weight = None
        self.final_dist_bias = None
        self.final_operation = None
        self.S = None

    @staticmethod
    def distance_fn_selem(weights: Union[np.ndarray, torch.tensor], S: Union[np.ndarray, torch.tensor], module_: ModuleType = np) -> float:
        """
        Args:
            weights (Union[np.ndarray, torch.tensor]): shape (nb_weights, nb_params_per_weight)
            S (Union[np.ndarray, torch.tensor]): shape (nb_weights, nb_params_per_weight)
            module_ (ModuleType, optional): Module to work with. Defaults to np.

        Returns:
            float: _description_
        """
        return (weights ** 2).sum(1) - 1 / S.sum(1) * (module_.where(S, weights, 0).sum(axis=1)) ** 2

    @staticmethod
    def find_best_index(w_values: Union[np.ndarray, torch.tensor], module_: ModuleType = np) -> Union[np.ndarray, torch.tensor]:
        r"""Gives the arg maximum of the distance function $\frac{\sum_{k \in S}{W_k}}{\sqrt{card{S}}} = \frac{\sum_{k = 1}^j{w_k}}{\sqrt{j}}$
        with $w_k$ the sorted values in descending order of the weights $W$.

        Args:
            w_values (np.ndarray): (n_chout, prod(kernel_size))

        Returns:
            array (n_chout): the index of the best value, for each channel.
        """
        if module_ == torch:
            arange = module_.sqrt(module_.arange(1, 1+w_values.shape[1])).to(device=w_values.device)
        else:
            arange = module_.sqrt(module_.arange(1, 1+w_values.shape[1]))
        return (module_.cumsum(w_values, axis=1) / arange).argmax(1)


    def find_best_S(self, weights: Union[np.ndarray, torch.tensor], module_: ModuleType = np, verbose: bool = False) -> Union[np.ndarray, torch.tensor]:
        r"""Gives the arg maximum of the distance function $\frac{\sum_{k \in S}{W_k}}{\sqrt{card{S}}} = \frac{\sum_{k = 1}^j{w_k}}{\sqrt{j}}$
        with $w_k$ the sorted values in descending order of the weights $W$.

        Args:
            w_values (np.ndarray): (n_chout, prod(kernel_size))

        Returns:
            array (n_chout): the index of the best value, for each channel.
        """
        W = weights

        w_values = module_.zeros_like(W)

        iterate = range(W.shape[0])
        if verbose:
            iterate = tqdm(iterate, leave=False, desc="Approximate binarization")

        for chout_idx, _ in enumerate(iterate):
            w_value_tmp = self.flip(module_.unique(W[chout_idx]))
            w_values[chout_idx, :len(w_value_tmp)] = w_value_tmp  # We assume that W don't repeat values. TODO: handle case with repeated values. Hint: add micro noise?

        best_idx = self.find_best_index(w_values, module_=module_)
        return (W >= w_values[module_.arange(w_values.shape[0]), best_idx, None])


    def find_approximate_uniform_S(self, weights: Union[np.ndarray, torch.tensor], module_: ModuleType = np, verbose: bool = False) -> Union[np.ndarray, torch.tensor]:
        """ Based on Ternary Weight Networks: https://arxiv.org/pdf/1605.04711.pdf
        """
        W = weights

        best_thresh = W.mean(1) * 2/3
        return W >= best_thresh[:, None]


    def find_approximate_normal_S(self, weights: Union[np.ndarray, torch.tensor], module_: ModuleType = np, verbose: bool = False) -> Union[np.ndarray, torch.tensor]:
        """ Based on Ternary Weight Networks: https://arxiv.org/pdf/1605.04711.pdf
        """
        W = weights

        best_thresh = W.mean(1) * 3/4
        return W >= best_thresh[:, None]


    def compute_dist_weights(self, verbose: bool = True, mode="exact") -> float:
        assert mode in ["exact", "uniform", "normal"], f"mode should be in ['exact', 'approximate'], not {mode}"
        W = self.weights
        if isinstance(W, torch.Tensor):
            module_ = torch

        else:
            module_ = np

        # w_values = module_.zeros_like(W)

        # iterate = range(W.shape[0])
        # if verbose:
        #     iterate = tqdm(iterate, leave=False, desc="Approximate binarization")

        # for chout_idx, _ in enumerate(iterate):
        #     w_value_tmp = self.flip(module_.unique(W[chout_idx]))
        #     w_values[chout_idx, :len(w_value_tmp)] = w_value_tmp  # We assume that W don't repeat values. TODO: handle case with repeated values. Hint: add micro noise?

        # best_idx = self.find_best_index(w_values, module_=module_)
        # self.S = (W >= w_values[module_.arange(w_values.shape[0]), best_idx, None])

        if mode == "exact":
            self.S = self.find_best_S(self.weights, module_=module_, verbose=verbose)
        elif mode == "uniform":
            self.S = self.find_approximate_uniform_S(self.weights, module_=module_, verbose=verbose)
        elif mode == "normal":
            self.S = self.find_approximate_normal_S(self.weights, module_=module_, verbose=verbose)

        self.final_dist_weight = self.distance_fn_selem(weights=W, S=self.S, module_=module_)
        return self.final_dist_weight

    def compute(self, verbose: bool = True, mode="exact") -> "ProjectionConstantSet":
        r"""Computes the projection onto constant set for each weight and bias.

        Args:
            weights (Union[np.ndarray, torch.tensor]): (nb_weight, n_params_per_weight) list of 1D weights
            bias (Union[np.ndarray, torch.tensor]): (nb_weight)
            verbose (bool, optional): Shows progress bar. Defaults to True.

        Returns:
            Union[np.ndarray, torch.tensor]: (nb_weight, n_params_per_weight) the closest constants et
            Union[np.ndarray, torch.tensor]: (nb_weight) the closest operation (erosion or dilation)
            Union[np.ndarray, torch.tensor]: (nb_weight) the distance to the closest activated space of constant set
        """
        # assert mode in ["exact", "uniform", "normal"], f"mode should be in ['exact', 'approximate'], not {mode}"
        W = self.weights
        bias = self.bias
        # if isinstance(W, torch.Tensor):
        #     module_ = torch

        # else:
        #     module_ = np

        # # w_values = module_.zeros_like(W)

        # # iterate = range(W.shape[0])
        # # if verbose:
        # #     iterate = tqdm(iterate, leave=False, desc="Approximate binarization")

        # # for chout_idx, _ in enumerate(iterate):
        # #     w_value_tmp = self.flip(module_.unique(W[chout_idx]))
        # #     w_values[chout_idx, :len(w_value_tmp)] = w_value_tmp  # We assume that W don't repeat values. TODO: handle case with repeated values. Hint: add micro noise?

        # # best_idx = self.find_best_index(w_values, module_=module_)
        # # self.S = (W >= w_values[module_.arange(w_values.shape[0]), best_idx, None])

        # if mode == "exact":
        #     self.S = self.find_best_S(self.weights, module_=np, verbose=verbose)
        # elif mode == "uniform":
        #     self.S = self.find_approximate_uniform_S(self.weights, module_=np, verbose=verbose)
        # elif mode == "normal":
        #     self.S = self.find_approximate_normal_S(self.weights, module_=np, verbose=verbose)

        # self.final_dist_weight = self.distance_fn_selem(weights=W, S=self.S, module_=module_)
        self.final_dist_weight = self.compute_dist_weights(verbose=verbose, mode=mode)

        wsum = W.sum(1)
        self.final_operation = np.empty(W.shape[0], dtype=str)

        if isinstance(W, torch.Tensor):
            dilation_idx = np.array(wsum.cpu().detach().numpy() / 2 >= -bias.cpu().detach().numpy())
        else:
            dilation_idx = np.array(wsum / 2 >= -bias)

        if dilation_idx.any():
            self.final_operation[dilation_idx] = self.operation_code["dilation"]
        if (~dilation_idx).any():
            self.final_operation[~dilation_idx] = self.operation_code["erosion"]

        # self.final_dist_bias = module_.abs(-bias - wsum)
        # self.final_dist = self.final_dist_weight + self.final_dist_bias
        self.final_dist = self.final_dist_weight  # TODO: take bias into account

        return self

    @staticmethod
    def flip(weights: Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:
        if isinstance(weights, torch.Tensor):
            return weights.flip(-1)
        else:
            return weights[..., ::-1]
