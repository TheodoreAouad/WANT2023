from typing import Callable, Union, List

import numpy as np
import torch
from torch.nn import Conv2d, Conv3d


def format_for_conv(ar: np.ndarray, device: torch.device) -> torch.Tensor:
    if not isinstance(ar, torch.Tensor):
        ar = torch.tensor(ar)
    return ar.unsqueeze(0).unsqueeze(0).float().to(device)


def conv_fn_gen(ndim, *args, **kwargs):
    conv_layer = {2: Conv2d, 3: Conv3d}[ndim](bias=False, *args, **kwargs)
    for param in conv_layer.parameters():
        del param
    return conv_layer._conv_forward


def array_erosion(ar: np.ndarray, selem: np.ndarray, device: torch.device = "cpu", return_numpy_array: bool = True) -> Union[np.ndarray, torch.Tensor]:
    conv_fn = conv_fn_gen(ar.ndim, padding=selem.shape[0] // 2, padding_mode="replicate", in_channels=1, out_channels=1, kernel_size=selem.shape)

    torch_array = (conv_fn(
        format_for_conv(ar, device=device), format_for_conv(selem, device=device), bias=torch.FloatTensor([0], device=device)
    ) == selem.sum()).squeeze()

    if return_numpy_array:
        return torch_array.to("cpu").int().numpy()
    return torch_array


def array_dilation(ar: np.ndarray, selem: np.ndarray, device: torch.device = "cpu", return_numpy_array: bool = True) -> Union[np.ndarray, torch.Tensor]:
    conv_fn = conv_fn_gen(ar.ndim, padding=selem.shape[0] // 2, padding_mode="replicate", in_channels=1, out_channels=1, kernel_size=selem.shape)

    torch_array = (conv_fn(
        format_for_conv(ar, device=device), format_for_conv(selem, device=device), bias=torch.FloatTensor([0], device=device)
    ) > 0).squeeze()

    if return_numpy_array:
        return torch_array.to("cpu").int().numpy()
    return torch_array


def array_intersection(ars: np.ndarray, axis: int = -1) -> np.ndarray:
    return ars.sum(axis) == ars.shape[axis]


def array_intersection_gray(ars: np.ndarray, axis: int = -1) -> np.ndarray:
    return ars.min(axis)


def array_union_gray(ars: np.ndarray, axis: int = -1) -> np.ndarray:
    return ars.max(axis)


def array_union(ars: np.ndarray, axis: int = -1) -> np.ndarray:
    return ars.sum(axis) > 0


def fn_chans(ar: np.ndarray, fn: Callable, chans: Union[str, List[int]] = 'all', return_numpy_array: bool = True) -> np.ndarray:
    if chans == 'all':
        chans = range(ar.shape[-1])
    if return_numpy_array:
        return fn(np.stack([ar[..., chan] for chan in chans], axis=-1))
    return torch.tensor(fn(np.stack([ar[..., chan] for chan in chans], axis=-1)))


def array_intersection_chans(ar, chans: Union[str, List[int]] = 'all') -> np.ndarray:
    return fn_chans(ar, array_intersection, chans)


def array_union_chans(ar, chans: Union[str, List[int]] = 'all') -> np.ndarray:
    return fn_chans(ar, array_union, chans)


def array_intersection_gray_chans(ar, chans: Union[str, List[int]] = 'all') -> np.ndarray:
    return fn_chans(ar, array_intersection_gray, chans)


def array_union_gray_chans(ar, chans: Union[str, List[int]] = 'all') -> np.ndarray:
    return fn_chans(ar, array_union_gray, chans)
