import torch

from .lui import LUI

# TODO: Use metaclass for symetric lui and other types of lui


def reshape_input(fn):
    def wrapper(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x (torch.Tensor): shape (batch_size, in_features)
        """
        if hasattr(self, "_is_reshaped_input") and self._is_reshaped_input:  # Already in wrapper
            return fn(self, x, *args, **kwargs)

        self._is_reshaped_input = True

        out = x[..., None, None]
        out = fn(self, out, *args, **kwargs)

        self._is_reshaped_input = False

        return out[..., 0, 0]
    return wrapper


class DenseLUI(LUI):
    _forward = reshape_input(LUI._forward)
    forward_binary = reshape_input(LUI.forward_binary)
