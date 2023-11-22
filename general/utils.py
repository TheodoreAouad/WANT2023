import copy
from typing import Tuple, Optional, List
from time import time
import logging
import os
from os.path import join
import re

import json
import yaml
import numpy as np
from scipy import ndimage
from skimage.morphology import disk, dilation, erosion, label
from skimage.transform import warp
from sklearn.model_selection import ParameterGrid


def list_dir_joined(folder: str):
    return [os.path.join(folder, k) for k in os.listdir(folder)]


def set_borders_to(ar: np.ndarray, border: Tuple, value: float = 0, ):
    res = ar + 0
    if border[0] != 0:
        res[:border[0], :] = value
        res[-border[0]:, :] = value
    if border[1] != 0:
        res[:, :border[1]] = value
        res[:, -border[1]:] = value
    return res


def one_hot_array(ar: np.ndarray, nb_chans: int = "auto", axis: int = -1, background: float = 0) -> np.ndarray:
    """ Performs one hot encoding of an array. Adds a channel on the axis axis, such that all the channels are binary.

    Args:
        ar (np.ndarray): array to one hot encode.
        axis (np.ndarray): axis where the additional channel is created.
        background (None | float): value to ignore

    Returns:
        np.ndarray: one hot encoded array
    """
    unique_values = sorted(list(set(np.unique(ar)).difference([background])))
    if nb_chans == 'auto':
        nb_chans = len(unique_values)

    if axis == -1:
        res = np.zeros(ar.shape + (nb_chans,))

        for idx, value in enumerate(unique_values):
            res[..., idx] = ar == value
        return res

    elif axis == 0:
        res = np.zeros((nb_chans,) + ar.shape)

        for idx, value in enumerate(unique_values):
            res[idx] = ar == value
        return res

    else:
        raise ValueError("axis must be 0 or -1.")


def save_json(dic, path, sort_keys=True, indent=4):
    with open(path, 'w') as fp:
        json.dump(dic, fp, sort_keys=sort_keys, indent=indent)


def load_json(path):
    with open(path, 'r') as fp:
        return json.load(fp)


def get_next_same_name(parent_dir, pattern='', sep='', crude=False):
    """
    Scans the folder parent dir for files/folders with name 'pattern{}' with {} being an integer. Returns
    the path to the file/folder with name 'pattern-{}' with the highest number +1.

    Args:
        parent_dir (str): path to the parent directory of the files / folders with the pattern
        pattern (str, optional): pattern of the file to look for. Defaults to ''.
        crude (bool, optional): if True, if file does not exist, will return parent_dir/pattern. Else,
                                will add 0.
    """
    if pattern == '':
        sep = ''

    if crude and not os.path.exists(join(parent_dir, pattern)):
        return join(parent_dir, pattern)

    if not os.path.exists(parent_dir):
        return join(parent_dir, '{}{}{}'.format(pattern, sep, 0))

    directories = [o for o in os.listdir(parent_dir) if re.search(r'^{}{}\d+$'.format(pattern, sep), o)]
    if len(directories) == 0:
        max_nb = 0
    else:
        nbrs = [int(re.findall(r'^{}{}(\d+)$'.format(pattern, sep), o)[0]) for o in directories]
        max_nb = max(nbrs) + 1
    return join(parent_dir, '{}{}{}'.format(pattern, sep, max_nb))


def log_console(to_print='', *args, level='info', logger=None, **kwargs):
    if logger is None:
        print(to_print, *args, **kwargs)
    else:
        to_print = '{}'.format(to_print)
        for st in args:
            to_print = to_print + ' {} '.format(st)
        getattr(logger, level.lower())(to_print)


def format_time(s):
    h = s // (3600)
    s %= 3600
    m = s // 60
    s %= 60
    return "%02i:%02i:%02i" % (h, m, s)


def create_logger(logger_name=None, all_logs_path=None, error_path=None, level="info"):

    level_dicts = {'debug': logging.DEBUG, 'info': logging.INFO}
    logging.basicConfig(
        level=level_dicts[level],
        format='%(message)s',
    )

    logger = logging.getLogger(__name__ if logger_name is None else logger_name)

    # Handling logs
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if error_path is not None:
        error_handler = logging.FileHandler(error_path)
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.WARNING)
        logger.addHandler(error_handler)

    if all_logs_path is not None:
        info_handler = logging.FileHandler(all_logs_path)
        info_handler.setFormatter(formatter)
        info_handler.setLevel(logging.DEBUG)
        logger.addHandler(info_handler)

    return logger


def close_handlers(logger):
    handlers = logger.handlers
    for handler in handlers:
        handler.close()


def save_yaml(dic, path):
    # To be able to save abstract classes
    from yaml.representer import Representer
    from abc import ABCMeta
    Representer.add_representer(ABCMeta, Representer.represent_name)

    with open(path, 'w') as f:
        yaml.dump(dic, f)


def dict_cross(dic, copy_dicts=True):
    """
    Does a cross product of all the values of the dict.

    Args:
        dic (dict): dict to unwrap

    Returns:
        list: list of the dict
    """

    if copy_dicts:
        dicts = list(ParameterGrid(dic))
        return [copy.deepcopy(d) for d in dicts]
    return list(ParameterGrid(dic))


def apply_crop(ar: np.ndarray, crop_xs: Tuple, crop_ys: Tuple, crop_zs: Optional[Tuple] = None):
    if len(ar.shape) == 2:
        return ar[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1]]

    if len(ar.shape) == 3:
        return ar[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1], crop_zs[0]:crop_zs[1]]


def reverse_crop(croped_ar: np.ndarray, size: Tuple, crop_xs: Tuple, crop_ys: Tuple, crop_zs: Optional[Tuple] = None,
                 fill_value: float = 0):
    ar = np.zeros(size) + fill_value
    if len(size) == 2:
        ar[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1]] = croped_ar

    if len(size) == 3:
        ar[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1], crop_zs[0]:crop_zs[1]] = croped_ar

    return ar.astype(croped_ar.dtype)


def grad_morp(img: np.ndarray, selem: np.ndarray = disk(1)):
    """
    Gives the morphological gradient of an image.

    Args:
        img (ndarray): image to give gradient of. Shape depends on dimension.
        selem (ndarray, optional): Local region. See dilation or erosion. Defaults to disk(1).

    Returns:
        ndarray: same shape as img.
    """
    return dilation(img, selem=selem) - erosion(img, selem=selem)


def grad_img(img: np.ndarray, mode: str = 'constant'):
    """
    Gives the sobel gradient of an image.

    Args:
        img (ndarray): shape (W, L)
        mode (str, optional): See ndimage.sobel. Defaults to 'constant'.

    Returns:
        ndarray: shape (W, L).
    """
    sx = ndimage.sobel(img, axis=0, mode=mode)
    sy = ndimage.sobel(img, axis=1, mode=mode)

    return np.hypot(sx, sy)


def center_and_crop(img: np.ndarray, mask: np.ndarray, size: Tuple, fill_background: int = 0):
    """
    Center and crop an img around the mask.

    Args:
        img (ndarray): shape (W, L)
        mask (ndarray): shape (W, L). Array of 0, 1 and 2.
        size (tuple): Size of the region.
        fill_background (int): value to fill the background with.

    Returns:
        ndarray: the img cropped around the mask of size size.
    """
    assert set(np.unique(mask)).issubset({0, 1, 2}), 'mask must have for values only 0, 1, 2. Values: {}'.format(np.unique(mask))
    x, y = np.where(mask != 0)

    if type(size) == int:
        size = (size, size)
    if len(x) == 0:
        x, y = (img.shape[0] - 1)/2, (img.shape[1] - 1)/2
    else:
        x = x.mean()
        y = y.mean()

    sx = (size[0] - 1) / 2
    sy = (size[1] - 1) / 2

    res = np.zeros(size) + fill_background

    res_x0 = max(ceil_(sx) - ceil_(x), 0)
    res_x1 = ceil_(sx) + min(floor_(sx) + 1, floor_(img.shape[0] - x))
    res_y0 = max(ceil_(sy) - ceil_(y), 0)
    res_y1 = ceil_(sy) + min(floor_(sy) + 1, floor_(img.shape[1] - y))


    res[res_x0:res_x1, res_y0:res_y1] = img[
        max(ceil_(x) - ceil_(sx), 0): ceil_(x) + floor_(sx) + 1,
        max(ceil_(y) - ceil_(sy), 0): ceil_(y) + floor_(sy) + 1,
    ]

    return res


def floor_(x: float):
    return np.int(np.floor(x))


def ceil_(x: float):
    return np.int(np.ceil(x))


def uniform_sampling(all_slices: np.ndarray, n_slices: int, dtype: type = int):
    if n_slices > 2:
        return np.linspace(all_slices.min(), all_slices.max(), n_slices).astype(dtype)

    if n_slices == 1:
        return [dtype(all_slices.min() + all_slices.max())]

    return np.array([
        (all_slices.min()*.67 + all_slices.max()*.33),
        (all_slices.min()*.33 + all_slices.max()*.67),
    ]).astype(dtype)


def get_arrangements(n: int) -> List[List]:
    """
    Get arrangements of n elements.
    Examples:
        if n = 3, returns
        [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    """
    def f(A, S):
        cur_set = set(A).difference(S)
        if len(cur_set) == 0:
            return [S]
        res = []
        for i in cur_set:
            res += f(A, S + [i])
        return res

    return f(range(n), [])


def get_most_important_labels(labels: np.ndarray, weights: np.ndarray, scope: int = 1, return_weights: bool = False) -> np.ndarray:
    """Returns the most weighted elements in labels.

    Args:
        labels (ndarray): array of elements
        weights (ndarray): array of weights for each elements
        scope (int, optional): number of labels to return. Defaults to 1.
        return_weights (bool, optional): If True, returns weights. Defaults to False.

    Returns:
        ndarray: shape (scope,). Labels with biggest weight.
    """
    labels_sorted = labels[weights.argsort()][::-1]
    if return_weights:
        weights_sorted = weights[weights.argsort()][::-1]
        return labels_sorted[:scope], weights_sorted[:scope]
    return labels_sorted[:scope]


def get_most_important_regions(regions: np.ndarray, weights: np.ndarray = 1, scope: int = 1, background: float = 0) -> np.ndarray:
    """Returns a mask containing only the biggest regions. The mask is labelled.
    The number of regions is the scope.

    Args:
        regions (ndarray): Mask of regions. Either 0 and 1, or already labeled.
        weights (ndarray, optional): weights to give to each label. Defaults to 1.
        scope (int, optional): number of regions. Defaults to 1.
        background (int, optional): pixels that are not part of the most important
                                    regions. Defaults to 0.

    Returns:
        ndarray: same shape as regions. Array like regions but with the less
                important regions being put to background.
    """
    if len(np.unique(regions)) == 2:
        regions = label(regions)
    labels, count = np.unique(regions, return_counts=True)
    count[labels == background] = 0
    weighted = count * weights
    biggest_labels = get_most_important_labels(labels, weighted, scope=scope)
    regions[~np.isin(regions, biggest_labels)] = background
    return regions


def apply_transform(
    ar,
    T,
    center='mid',
    order=1,
    do_scipy_func=True,
    show_time=False,
    **kwargs
):
    """
    Apply a transformation array T to each element of array ar. Returns
    ar[T(i)] for indexes i.

    Args:
        ar (ndarray): array to transform
        T (ndarray): square array of shape len(ar.shape) + 1
        order (int, optional): Interpolation order of B-spline. See skimage.transform.warp for more details.
                               Defaults to 1 for linear interpolation.
        show_time (bool, optional): Show computation time. Defaults to False.

    Returns:
        ndarray: same shape as ar. ar[T(i)] for all indexes i, interpolated.
    """
    start = time()
    if center == "mid":
        center = np.array([(shp -1)/2 for shp in ar.shape])

    if do_scipy_func:
        rot_matrix = T[:-1, :-1]
        trans_matrix = T[:-1, -1]
        if type(center) in [int, float]:
            center = np.zeros_like(trans_matrix) + center
        offset = trans_matrix - rot_matrix @ center + center
        res = ndimage.affine_transform(ar, rot_matrix, offset=offset, order=order, **kwargs)
        if show_time:
            print('Applying scipy transform:', time() - start)

    else:
        if type(center) == np.ndarray:
            center = center[:, np.newaxis]
        coords = np.meshgrid(*[np.arange(n) for n in ar.shape], indexing='ij')
        samples = np.c_[[ar[ar==ar] for ar in coords]] - center
        samples = np.vstack((samples, np.ones(samples.shape[1])))
        new_indexs = (T @ samples)[:-1] + center
        t1 = time()
        if show_time:
            print('Apply transform to indexes:', t1 - start)
        res = warp(ar, new_indexs, order=order, **kwargs).reshape(*ar.shape)
        t2 = time()
        if show_time:
            print('Apply warping:', t2-t1)
    return res


def colormap_1d(colors: np.ndarray, colormap: str = "br") -> np.ndarray:
    if colormap == 'br':
        return np.stack((colors, np.zeros_like(colors), 1 - colors), axis=-1)


def constant_color(color: np.ndarray, size: int) -> np.ndarray:
    return np.stack([color for _ in range(size)], axis=0)


def max_min_norm(ar: np.ndarray) -> np.ndarray:
    armin = ar.min()
    armax = ar.max()

    if armin == armax:
        return ar / armax

    ar = ar + 0
    ar[ar == np.infty] = ar[ar != np.infty].max()
    return (ar - armin) / (armax - armin)


def uniform_sampling_bound(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if isinstance(a, (float, int)):
        a = np.array([a])
    return np.random.rand(*a.shape) * (b - a) + a


def gaussian_sampling(std: np.ndarray) -> np.ndarray:
    return np.random.randn(*std.shape) * std


def read_obj_file(path: str) -> (np.ndarray, np.ndarray):
    """ Reads obj file and returns vertices (not normalized) and faces.
    """

    verts = []
    faces = []
    with open(path, "r") as f:
        lines = f.read().splitlines()

    for line in lines:
        if len(line) == 0:
            continue
        if line[:2] == "v ":
            verts.append(np.array([float(c) for c in line[2:].split(" ")]))
        if line[0] == "f":
            idxs = line[2:].split(" ")[:3]
            final_idx = []
            for idx in idxs:
                idx1, idx2 = idx.split('//')
                assert idx1 == idx2
                final_idx.append(int(idx1))
            faces.append(np.array(final_idx) - 1)
    return np.array(verts), np.array(faces)
