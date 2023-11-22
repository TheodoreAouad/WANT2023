from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
# from numba import njit
from general.utils import set_borders_to

from .utils import (
    rand_shape_2d,
    invert_proba,
    get_rect_vertices,
    draw_poly,
    draw_ellipse
)


def get_random_rotated_diskorect(
    size: Tuple, n_shapes: int = 30, max_shape: Tuple[int] = (15, 15), p_invert: float = 0.5,
        border=(4, 4), n_holes: int = 15, max_shape_holes: Tuple[int] = (5, 5), noise_proba=0.05,
        rng_float=np.random.rand, rng_int=np.random.randint, **kwargs
):
    diskorect = np.zeros(size)
    img = Image.fromarray(diskorect)
    draw = ImageDraw.Draw(img)

    def draw_shape(max_shape, fill_value):
        x = rng_int(0, size[0] - 2)
        y = rng_int(0, size[0] - 2)

        if rng_float() < .5:
            W = rng_int(1, max_shape[0])
            L = rng_int(1, max_shape[1])

            angle = rng_float() * 45
            draw_poly(draw, get_rect_vertices(x, y, W, L, angle), fill_value=fill_value)

        else:
            rx = rng_int(1, max_shape[0]//2)
            ry = rng_int(1, max_shape[1]//2)
            draw_ellipse(draw, np.array([x, y]), (rx, ry), fill_value=fill_value)

    for _ in range(n_shapes):
        draw_shape(max_shape=max_shape, fill_value=1)

    for _ in range(n_holes):
        draw_shape(max_shape=max_shape_holes, fill_value=0)

    diskorect = np.asarray(img) + 0
    diskorect[rand_shape_2d(diskorect.shape, rng_float=rng_float) < noise_proba] = 1  # bernoulli noise
    diskorect = invert_proba(diskorect, p_invert, rng_float=rng_float)  # random invert

    diskorect = set_borders_to(diskorect, border, value=0)

    return diskorect


def get_random_diskorect_channels(size: Tuple, squeeze: bool = False, *args, **kwargs):
    """Applies diskorect to multiple channels.

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
    for chan in range(H):
        final_img[..., chan] = get_random_rotated_diskorect((W, L), *args, **kwargs)

    if squeeze:
        return np.squeeze(final_img)
    return final_img
