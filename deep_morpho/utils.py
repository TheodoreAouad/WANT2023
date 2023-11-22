# from numba import njit
import numpy as np
import random

from general.nn.utils import set_seed as set_seed_torch


# @njit
# def set_seed_numba(seed):
#     np.random.seed(seed)
#     random.seed(seed)


def set_seed(seed=None):
    seed = set_seed_torch(seed)
    # set_seed_numba(seed)
    return seed
