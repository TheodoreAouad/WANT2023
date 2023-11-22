import random
from typing import Tuple, Dict

import numpy as np


def draw_rectangle(ar, top_xy: Tuple, shape: Tuple) -> np.ndarray:
    x, y = top_xy
    W, L = shape
    ar = ar + 0
    ar[x:x + W, y: y+L] = 1
    return ar


def get_random_rectangle_args(
        size: Tuple[int],
        top_xy: Tuple[int] = None,
        shape: Tuple[int] = None,
        max_shape: Tuple[int] = None,
        centered: bool = False,
        top_left_x_props: Tuple[float] = (0, 1),
        top_left_y_props: Tuple[float] = (0, 1),
        width_props: Tuple[float] = (.1, .9),
        length_props: Tuple[float] = (.1, .9),
) -> np.ndarray:
    if max_shape is None:
        max_shape = size

    if shape is None:
        min_W, max_W = int(size[0] * width_props[0]), int(size[1] * width_props[1])
        W = int(random.uniform(min_W, max_W))

        min_L, max_L = int(size[0] * length_props[0]), int(size[1] * length_props[1])
        L = int(random.uniform(min_L, max_L))
    else:
        W, L = shape

    W, L = min(W, max_shape[0]), min(L, max_shape[1])

    if centered:
        top_left_x = (size[0] - W) // 2
        top_left_y = (size[1] - L) // 2

    elif top_xy is None:
        x_min, x_max = [int(top_left_x_props[i] * (size[i] - W)) for i in range(2)]
        y_min, y_max = [int(top_left_y_props[i] * (size[i] - L)) for i in range(2)]
        top_left_x = random.choice(range(x_min, min(x_max, size[0] - W)))
        top_left_y = random.choice(range(y_min, min(y_max, size[1] - L)))
    else:
        top_left_x, top_left_y = top_xy

    return (top_left_x, top_left_y), (W, L)


def random_rectangle(size: Tuple, return_args=False, *args, **kwargs) -> np.ndarray:
    rect = np.zeros(size)
    rect_args = get_random_rectangle_args(size, *args, **kwargs)
    res = draw_rectangle(rect, *rect_args)

    if return_args:
        return res, rect_args
    return res


def random_multi_rect(size: Tuple, n_rectangles: int, max_shape: Tuple[int] = None, return_rects: bool = False, first_rect_args: Dict = {}) -> np.ndarray:
    if n_rectangles == 0:
        return np.zeros(size)

    if max_shape is None:
        max_shape = size

    res, (cur_xy, shape) = random_rectangle(size, max_shape=max_shape, return_args=True, **first_rect_args)

    all_args = [(cur_xy, shape)]

    for idx in range(n_rectangles - 1):
        cur_xy, cur_shape = random.choice(all_args)
        # print("cur args", cur_xy, cur_shape)
        go_left_x = random.randint(0, 1) == 0
        go_left_y = random.randint(0, 1) == 0

        # print("left_x", go_left_x, " || ", "left_y", go_left_y)

        next_x = random.choice(range(cur_xy[0], cur_xy[0] + cur_shape[0]))
        if go_left_x:
            if cur_xy[0] < 0:
                assert False, f"{cur_xy[0]}"
            W = random.choice(range(next_x - cur_xy[0], next_x + 1))
        else:
            W = random.choice(range(cur_xy[0] + cur_shape[0] - next_x - 1, size[0] - next_x))

        next_y = random.choice(range(cur_xy[1], cur_xy[1] + cur_shape[1]))
        if go_left_y:
            if cur_xy[1] < 0:
                assert False, f"{cur_xy[1]}"
            L = random.choice(range(next_y - cur_xy[1], next_y + 1))
        else:
            L = random.choice(range(cur_xy[1] + cur_shape[1] - next_y - 1, size[1] - next_y))

        W = min(max(W, 1), max_shape[0])
        L = min(max(L, 1), max_shape[1])

        if go_left_x:
            next_x = next_x - W
        if go_left_y:
            next_y = next_y - L

        next_x, next_y = max(next_x, 0), max(next_y, 0)  # TODO: check why this is needed...
        res = draw_rectangle(res, (next_x, next_y), (W, L))
        # print("added args:", (next_x, next_y), (W, L))
        all_args.append(((next_x, next_y), (W, L)))

    if return_rects:
        return res, all_args
    return res
