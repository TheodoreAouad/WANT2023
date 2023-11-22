from typing import Tuple, TYPE_CHECKING, List
from enum import Enum
from functools import partial

from tqdm import tqdm
import numpy as np
import cvxpy as cp

if TYPE_CHECKING:
    from deep_morpho.bise import BiSE
    import torch


from general.nn.experiments.experiment_methods import ExperimentMethods
from .projection_constant_set import ProjectionConstantSet


class ClosestSelemEnum(Enum):
    MIN_DIST = 1
    MAX_SECOND_DERIVATIVE = 2
    MIN_DIST_DIST_TO_BOUNDS = 3
    MIN_DIST_DIST_TO_CST = 4
    MIN_DIST_ACTIVATED_POSITIVE = 5


class ClosestSelemDistanceEnum(Enum):
    DISTANCE_TO_BOUNDS = 1
    DISTANCE_BETWEEN_BOUNDS = 2
    DISTANCE_TO_AND_BETWEEN_BOUNDS = 3


def distance_agg_min(self, selems: np.ndarray, dists: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx_min = dists.argmin()
    return selems[idx_min], dists[idx_min]


def distance_agg_max_second_derivative(self, selems: np.ndarray, dists: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx_min = (dists[2:] + dists[:-2] - 2 * dists[1:-1]).argmax() + 1
    return selems[idx_min], dists[idx_min]


def distance_to_bounds_base(
    bound_fn, weights: "torch.Tensor", bias: "torch.Tensor", S: np.ndarray, v1: float = 0, v2: float = 1
) -> float:
    assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
    lb, ub = bound_fn(weights=weights, S=S, v1=v1, v2=v2)
    dist_lb = lb + bias  # if dist_lb < 0 : lower bound respected
    dist_ub = -bias - ub  # if dist_ub < 0 : upper bound respected
    return max(dist_lb, dist_ub, 0)


def distance_fn_to_bounds(
    self, weights: "torch.Tensor", bias: "torch.Tensor", operation: str, S: np.ndarray, v1: float, v2: float
) -> float:
    if operation == "dilation":
        bound_fn = self.bise_module.bias_bounds_dilation
    elif operation == "erosion":
        bound_fn = self.bise_module.bias_bounds_erosion
    else:
        raise SyntaxError("operation must be either 'dilation' or 'erosion'.")

    return distance_to_bounds_base(bound_fn, weights=weights, bias=bias, S=S, v1=v1, v2=v2)


def distance_between_bounds_base(
    bound_fn, weights: "torch.Tensor", bias: "torch.Tensor", S: np.ndarray, v1: float = 0, v2: float = 1
) -> float:
    assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
    lb, ub = bound_fn(weights=weights, S=S, v1=v1, v2=v2)
    return lb - ub


def distance_fn_between_bounds(
    self, weights: "torch.Tensor", bias: "torch.Tensor", operation: str, S: np.ndarray, v1: float, v2: float
) -> float:
    if operation == "dilation":
        bound_fn = self.bise_module.bias_bounds_dilation
    elif operation == "erosion":
        bound_fn = self.bise_module.bias_bounds_erosion
    else:
        raise SyntaxError("operation must be either 'dilation' or 'erosion'.")

    return distance_between_bounds_base(bound_fn, weights=weights, bias=bias, S=S, v1=v1, v2=v2)


class BiseClosestSelemHandler(ExperimentMethods):
    def __init__(self, bise_module: "BiSE" = None):
        self.bise_module = bise_module

    def __call__(
        self,
        chans: List[int] = None,
        v1: float = 0,
        v2: float = 1,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the closest selem and operation for all given chans. If no chans are given, for all chans.

        Args:
            chans (list): list of given channels. If non given, all channels computed.
            v1 (float, optional): first argument for almost binary. Defaults to 0.
            v2 (float, optional): second argument for almost binary. Defaults to 1.
            verbose (bool, optional): shows progress bar.

        Returns:
            array (n_chan, *kernel_size) bool, array(n_chan) str, array(n_chan) float: the selem, the operation and the distance to the closest selem
        """
        pass


class BiseClosestSelemWithDistanceAgg(BiseClosestSelemHandler):
    """Children must define:
    distance_fn(self, weights: "torch.Tensor", bias: "torch.Tensor", operation: str, S: np.ndarray, v1: float, v2: float) -> float
        given an operation, weights, bias, selems and almost binary bounds, outputs a similarity measure float
    distance_agg_fn(self, selems: np.ndarray, dists: np.ndarray) -> Tuple[np.ndarray, float]
        given a list of selems and dists in the form of arrays, outputs the best selem given the distances, and its corresponding distance
    """

    def __init__(self, distance_fn=None, distance_agg_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distance_fn = partial(distance_fn, self=self) if distance_fn is not None else None
        self._distance_agg_fn = partial(distance_agg_fn, self=self) if distance_agg_fn is not None else None

    @property
    def distance_fn(self):
        return self._distance_fn

    @property
    def distance_agg_fn(self):
        return self._distance_agg_fn

    def compute_selem_dist_for_operation_chan(
        self, weights, bias, operation: str, chout: int = 0, v1: float = 0, v2: float = 1
    ):
        weights = weights[chout]
        # weight_values = weights.unique().detach().cpu().numpy()
        weight_values = np.unique(weights)
        bias = bias[chout]

        dists = np.zeros_like(weight_values)
        selems = []
        for value_idx, value in enumerate(weight_values):
            selem = weights >= value
            # selem = (weights >= value).cpu().detach().numpy()
            dists[value_idx] = self.distance_fn(weights=weights, bias=bias, operation=operation, S=selem, v1=v1, v2=v2)
            selems.append(selem)

        return selems, dists

    def find_closest_selem_and_operation_chan(self, weights, bias, chout=0, v1=0, v2=1):
        final_dist = np.infty
        weights = weights.detach().cpu().numpy()
        bias = bias.detach().cpu().numpy()
        for operation in ["dilation", "erosion"]:
            selems, dists = self.compute_selem_dist_for_operation_chan(
                weights=weights, bias=bias, chout=chout, operation=operation, v1=v1, v2=v2
            )
            new_selem, new_dist = self.distance_agg_fn(selems=selems, dists=dists)
            if new_dist < final_dist:
                final_dist = new_dist
                final_selem = new_selem
                final_operation = operation

        return final_selem, final_operation, final_dist

    def __call__(
        self,
        chans: List[int] = None,
        v1: float = 0,
        v2: float = 1,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if chans is None:
            chans = range(self.bise_module.out_channels)
        if verbose:
            chans = tqdm(chans, leave=False, desc="Approximate binarization")

        # for chan in chans:
        #     self.find_closest_selem_and_operation_chan(chan)

        closest_selems = np.zeros(self.bise_module.weight.shape, dtype=bool)
        # closest_selems = np.zeros((len(chans), *self.bise_module.kernel_size), dtype=bool)
        closest_operations = np.zeros(len(chans), dtype=str)
        closest_dists = np.zeros(len(chans))

        for chout_idx, chout in enumerate(chans):
            selem, op, dist = self.find_closest_selem_and_operation_chan(
                weights=self.bise_module.weights, bias=self.bise_module.bias, chout=chout, v1=v1, v2=v2
            )
            closest_selems[chout_idx] = selem.astype(bool)
            closest_operations[chout_idx] = self.bise_module.operation_code[op]
            closest_dists[chout_idx] = dist

        return closest_selems, closest_operations, closest_dists


class BiseClosestMinDistBounds(BiseClosestSelemWithDistanceAgg):
    def __init__(self, *args, **kwargs):
        super().__init__(distance_fn=distance_fn_to_bounds, distance_agg_fn=distance_agg_min, *args, **kwargs)


class BiseClosestActivationSpaceIteratedPositive(BiseClosestSelemWithDistanceAgg):
    def __init__(self, *args, **kwargs):
        super().__init__(distance_agg_fn=distance_agg_min, *args, **kwargs)
        self._distance_fn = partial(self.solve)

    def solve(
        self, weights: np.ndarray, bias: np.ndarray, operation: str, S: np.ndarray, v1: float, v2: float
    ) -> float:
        bias = -bias
        if operation == "erosion":
            bias = weights.sum() - bias
        # print("banana")

        weights = weights.flatten()
        S = S.flatten()

        Wvar = cp.Variable(weights.shape)
        bvar = cp.Variable(1)

        constraints = self.dilation_constraints(Wvar, bvar, S)
        # if operation == "dilation":
        #     constraints = self.dilation_constraints(Wvar, bvar, S)
        # elif operation == "erosion":
        #     constraints = self.erosion_constraints(Wvar, bvar, S)
        # else:
        #     raise ValueError("operation must be dilation or erosion")

        objective = cp.Minimize(1 / 2 * cp.sum_squares(Wvar - weights) + 1 / 2 * cp.sum_squares(bvar - bias))
        prob = cp.Problem(objective, constraints)
        prob.solve()

        return prob.value

    def dilation_constraints(self, Wvar: cp.Variable, bvar: cp.Variable, S: np.ndarray):
        self.constraint0 = [cp.sum(Wvar[~S]) <= bvar]
        self.constraintsT = [bvar <= Wvar[S]]
        self.constraintsK = [Wvar >= 0]
        return self.constraint0 + self.constraintsT + self.constraintsK

    def erosion_constraints(self, Wvar: cp.Variable, bvar: cp.Variable, S: np.ndarray):
        self.constraint0 = [cp.sum(Wvar[S]) >= bvar]
        self.constraintsT = [cp.sum(Wvar) - Wvar[S] <= bvar]
        self.constraintsK = [Wvar >= 0]
        return self.constraint0 + self.constraintsT + self.constraintsK


class BiseClosestMinDistOnCstOld(BiseClosestSelemWithDistanceAgg):
    def __init__(self, *args, **kwargs):
        super().__init__(distance_fn=self.distance_fn_selem, distance_agg_fn=distance_agg_min, *args, **kwargs)
        self.distance_fn_bias = partial(distance_fn_between_bounds, self=self)

    @staticmethod
    def distance_fn_selem(
        self, weights: "torch.Tensor", bias: "torch.Tensor", operation: str, S: np.ndarray, v1: float, v2: float
    ) -> float:
        W = weights.cpu().detach().numpy()
        return (W**2).sum() - 1 / S.sum() * (W[S].sum()) ** 2

    def find_closest_selem_and_operation_chan(self, weights, bias, chout=0, v1=0, v2=1):
        final_dist_selem = np.infty
        final_dist_bias = np.infty

        selems, dists = self.compute_selem_dist_for_operation_chan(
            weights=weights, bias=bias, chout=chout, operation=None, v1=v1, v2=v2
        )
        new_selem, new_dist = self.distance_agg_fn(selems=selems, dists=dists)
        if new_dist < final_dist_selem:
            final_dist_selem = new_dist
            final_selem = new_selem

        wsum = weights[chout].sum()
        if -bias[chout] <= wsum / 2:
            final_operation = "dilation"
        else:
            final_operation = "erosion"

        final_dist_bias = (-bias[chout] - wsum).abs().cpu().detach().numpy()

        # for operation in ['dilation', 'erosion']:
        #     dist_bias = self.distance_fn_bias(weights=weights[chout], bias=bias, operation=operation, S=final_selem, v1=v1, v2=v2)
        #     if dist_bias < final_dist_bias:
        #         final_operation = operation

        final_dist = final_dist_selem + final_dist_bias

        return final_selem, final_operation, final_dist


class BiseClosestMinDistOnCst(BiseClosestSelemHandler):
    def __init__(self, mode: str = "exact", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ["exact", "uniform", "normal"]
        self.mode = mode

    # TODO: handle differently the return_np_array to handle both cases: the observable and the loss
    def find_closest_selem_and_operation(
        self,
        weights,
        bias,
        chans=None,
        v1=0,
        v2=1,
        verbose: bool = True,
        return_np_array: bool = True,
    ):
        if chans is None:
            chans = range(self.bise_module.out_channels)

        W = weights[chans]
        bias = bias[chans]
        if return_np_array:
            W = W.cpu().detach().numpy()
            bias = bias.cpu().detach().numpy()

        proj = ProjectionConstantSet(W.reshape(W.shape[0], -1), bias).compute(verbose=verbose, mode=self.mode)
        S, final_operation, final_dist = proj.S, proj.final_operation, proj.final_dist
        S = S.reshape(W.shape)
        return S, final_operation, final_dist

    def __call__(self, *args, **kwargs) -> Tuple[float, np.ndarray, str]:
        return self.find_closest_selem_and_operation(
            weights=self.bise_module.weights, bias=self.bise_module.bias, *args, **kwargs
        )


class BiseClosestMinDistOnCstApproxBase(BiseClosestSelemHandler):
    """Implement the approximation methods introduced in Li et al. 2016 https://arxiv.org/pdf/1605.04711.pdf
    Ternary Weight Networks
    """

    def approximation_threshold(self, weights: np.ndarray) -> np.array:
        pass

    def find_closest_selem_and_operation(
        self,
        weights,
        bias,
        chans=None,
        v1=0,
        v2=1,
        verbose: bool = True,
    ):
        if chans is None:
            chans = range(self.bise_module.out_channels)

        weights = weights[chans].cpu().detach().numpy()
        bias = bias[chans].cpu().detach().numpy()

        W = weights.reshape(weights.shape[0], -1)

        wsum = W.sum(1)
        final_operation = np.empty(W.shape[0], dtype=str)

        if isinstance(W, torch.Tensor):
            dilation_idx = np.array(wsum.cpu().detach().numpy() / 2 >= -bias.cpu().detach().numpy())
        else:
            dilation_idx = np.array(wsum / 2 >= -bias)

        if dilation_idx.any():
            final_operation[dilation_idx] = self.bise_module.operation_code["dilation"]
        if (~dilation_idx).any():
            final_operation[~dilation_idx] = self.bise_module.operation_code["erosion"]

        final_dist_bias = np.abs(-bias - wsum)

        deltas = self.approximation_threshold(W)
        S = W >= deltas[:, None]

        final_dist_selem = ProjectionConstantSet.distance_fn_selem(W, deltas * S)

        final_dist = final_dist_selem + final_dist_bias

        S = S.reshape(weights.shape)
        return S, final_operation, final_dist

    def __call__(self, *args, **kwargs) -> Tuple[float, np.ndarray, str]:
        return self.find_closest_selem_and_operation(
            weights=self.bise_module.weights, bias=self.bise_module.bias, *args, **kwargs
        )
