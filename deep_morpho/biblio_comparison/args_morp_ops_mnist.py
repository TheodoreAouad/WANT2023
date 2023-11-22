import numpy as np
import torch
from general.structuring_elements import *

from deep_morpho.morp_operations import ParallelMorpOperations, ParallelMorpOperationsGrayTopHats

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

morp_operations = []

for op in [
    # 'disk', 'hstick', 'vstick', 'scross',
    # 'dcross', 'square'
    # "disk",
    # "hstick",
    # "dcross",
    "scross",
    "bsquare",
    "bdiamond",
    "bcomplex",
]:
    size1 = 5
    size2 = 7
    if op == "disk":
        size1 = size1 // 2
        size2 = size2 // 2

    # morp_operations.append(ParallelMorpOperations.dilation((op, size2), name=f"dilation/{op}"))
    # morp_operations.append(ParallelMorpOperations.erosion((op, size2), name=f"erosion/{op}"))
    morp_operations.append(ParallelMorpOperations.closing((op, size2), name=f"closing/{op}"))
    morp_operations.append(ParallelMorpOperations.opening((op, size2), name=f"opening/{op}"))
