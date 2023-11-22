from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
# from numba import njit
from general.utils import set_borders_to

from .utils import (
    rand_shape_2d,
    invert_proba,
    get_rect_vertices,
    draw_poly
)


def get_sticks_noised(
    size: Tuple, n_shapes: int = 30, lengths_lim: Tuple = (12, 15), widths_lim: Tuple = (3, 4), p_invert: float = 0.5,
    angles: np.ndarray = np.linspace(0, 180, 10), border: Tuple = (0, 0), noise_proba: float = 0.05,
    rng_float=np.random.rand, rng_int=np.random.randint, **kwargs
):
    sticks = np.zeros(size)
    img = Image.fromarray(sticks)
    draw = ImageDraw.Draw(img)

    def draw_shape():
        x = rng_int(0, size[0] - 2)
        y = rng_int(0, size[0] - 2)

        L = rng_int(lengths_lim[0], lengths_lim[1] + 1)
        W = rng_int(widths_lim[0], widths_lim[1] + 1)

        angle = angles[rng_int(0, len(angles))]
        draw_poly(draw, get_rect_vertices(x, y, W, L, angle), fill_value=1)


    for _ in range(n_shapes):
        draw_shape()

    sticks = np.asarray(img) + 0

    sticks_noisy = sticks + 0

    sticks_noisy[rand_shape_2d(sticks.shape, rng_float=rng_float) < noise_proba] = 1  # bernoulli noise
    sticks, sticks_noisy = invert_proba([sticks, sticks_noisy], p_invert, rng_float=rng_float)  # random invert

    sticks = set_borders_to(sticks, border, value=0)
    sticks_noisy = set_borders_to(sticks_noisy, border, value=0)

    return sticks, sticks_noisy


def get_sticks_noised_channels(size: Tuple, squeeze: bool = False, *args, **kwargs):
    """Applies sticks noise to multiple channels.

    Args:
        size (Tuple): (W, L, H)
        squeeze (bool, optional): If True, squeeze the output: if H = 1, returns size (W, L). Defaults to False.

    Raises:
        ValueError: size must be of len 2 or 3, either (W, L) or (W, L, H) with H number of channels.

    Returns:
        np.ndarray: size (W, L) or (W, L, H)
    """
    if len(size) == 3:
        W, L, H = size
    elif len(size) == 2:
        W, L = size
        H = 1
    else:
        raise ValueError(f"size argument must have 3 or 2 values, not f{len(size)}.")

    final_img = np.zeros((W, L, H))
    final_img_noised = np.zeros((W, L, H))
    for chan in range(H):
        final_img[..., chan], final_img_noised[..., chan] = get_sticks_noised((W, L), *args, **kwargs)

    if squeeze:
        return np.squeeze(final_img), np.squeeze(final_img_noised)
    return final_img, final_img_noised
