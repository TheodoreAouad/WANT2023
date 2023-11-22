from typing import Tuple, Union

import torch
import numpy as np

from deep_morpho.tensor_with_attributes import TensorGray


def undersample(min_value: float, max_value: float, n_value: int) -> np.ndarray:
    return np.round(np.linspace(min_value, max_value, n_value)).astype(int)


def level_sets_from_gray(
    ar: Union[np.ndarray, torch.Tensor],
    values: Union[np.ndarray, torch.Tensor] = None,
    n_values: int = "all"
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[TensorGray, torch.Tensor]]:
    """ Given an array with any number of values, outputs an array with one more dimension
    with level sets as binary arrays.

    Args:
        ar (np.ndarray or torch.Tensor): shape (*ar.shape), input gray scale image of any dimension
        values (np.ndarray or torch.Tensor): shape (len(values), ), the values to choose from for level set
        n_values (int or str): if str is given, then all values will be taken. Otherwise, undersample to n_values
                                by taking evenly spaced element (index-wise) of the list values.

    Returns:
        TensorGray: shape (n_values, *ar.shape): the binary image with one more dim, where the first dim is the level sets
        torch.Tensor: shape (n_values): the values used for the level set of the final image

    """
    if isinstance(ar, np.ndarray):
        values = np.unique(ar) if values is None else values  # If no values given, take all values of the image
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        constructor_fn = np.zeros

    elif isinstance(ar, torch.Tensor):
        values = torch.unique(ar) if values is None else values  # If no values given, take all values of the image
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, device=ar.device)  # was torch.Tensor, may cause bugs
        # constructor_fn = partial(torch.zeros, device=ar.device)

        def constructor_fn(x):
            res = TensorGray(size=x)
            res.to(ar.device)
            return res

    else:
        raise ValueError("ar type must be numpy.ndarray or torch.Tensor")

    if isinstance(n_values, int) and n_values > 0:  # If number of values is not all, then we undersample the values
        values = values[undersample(0, len(values)-1, n_values)]

    res = constructor_fn((len(values),) + ar.shape)
    for idx, v in enumerate(values):
        res[idx] = ar >= v

    return res, values


def gray_from_level_sets(ar: Union[np.ndarray, torch.Tensor], values: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """ Given level sets (in {0, 1}) and the corresponding values, outputs the corresponding gray scale array
    """
    v2 = values[1:] - values[:-1]

    if isinstance(v2, np.ndarray):
        v2 = np.concatenate([[values[0]], v2])
    elif isinstance(v2, torch.Tensor):
        v2 = torch.cat([TensorGray([values[0]]).to(v2.device), v2])
        # v2 = v2.to(ar.device)
    else:
        raise ValueError("value type must be numpy.ndarray or torch.Tensor")

    return (ar * v2[:, None, None]).sum(0)
