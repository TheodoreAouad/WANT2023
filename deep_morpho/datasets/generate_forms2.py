from typing import Tuple
import random

import numpy as np


def draw_disk(ar: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    XX, YY = np.meshgrid(np.arange(ar.shape[0]), np.arange(ar.shape[1]))
    idxes = np.sqrt((XX - center[0])**2 + (YY - center[1])**2) - radius
    ar[idxes < 0] = 1
    return ar


def get_random_diskorect(
    size: Tuple, n_shapes: int = 30, max_shape: Tuple[int] = (15, 15), borders: Tuple[float] = (.1, .9), p_invert: float = 0.5,
) -> np.ndarray:

    diskorect = np.zeros(size)
    min_x, min_y = int(size[0] * borders[0]), int(size[1] * borders[0])
    max_x, max_y = int(size[0] * borders[1]), int(size[1] * borders[1])

    for _ in range(n_shapes):
        # diskorect = all_diskorect[-1] + 0
        not_0 = np.where(1 - diskorect)
        rand_idx = random.choice(range(len(not_0[0])))
        center_x, center_y = not_0[0][rand_idx], not_0[1][rand_idx]

        center_x = min(max_x - 2, max(min_x + 2, center_x))
        center_y = min(max_y - 2, max(min_y + 2, center_y))


        if np.random.rand() > .5:
            W1 = np.random.randint(1, min(center_x - min_x, max_shape[0] // 2))
            W2 = np.random.randint(1, min(max_x - center_x, max_shape[0] // 2))

            L1 = np.random.randint(1, min(center_y - min_y, max_shape[1] // 2))
            L2 = np.random.randint(1, min(max_y - center_y, max_shape[1] // 2))

            diskorect[center_x-W1:center_x+W2, center_y-L1:center_y+L2] = 1


        else:
            radius = np.random.randint(1, min(center_x - min_x, max_x - center_x, max_shape[0]))
            radius = min(max_x - center_x, center_x - min_x, max_y - center_y, center_y - min_y, radius)
            draw_disk(diskorect, (center_x, center_y), radius)

        # all_diskorect.append(diskorect)
    if random.random() < p_invert:
        diskorect = 1 - diskorect

    return diskorect
    # return all_diskorect[-1]
