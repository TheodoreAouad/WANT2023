from torch import Tensor


class TensorGray(Tensor):
    """ torch.Tensor with attributes adapted for gray scale to level sets computations.

    Gray-scale image batch represented as level sets. Shape (batch_size, n_level_sets, *img.shape).

    Attributes:
        indexes (list-like): list of indexes corresponding to each gray scale image in the batch.
                            Ex:
                            >>> self[indexes[0]:indexes[1]]   # first img.
                            >>> self[indexes[1]:indexes[2]]   # second img.
        gray_values (torch.Tensor): gray scale values to recover the gray scale image.
        original (torch.Tensor): original gray image. The reconstructed gray image is an approximation of this image,
                                    depending on the precision of the values given.
    """

    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        if hasattr(self, "gray_values"):
            res.gray_values = self.gray_values.to(*args, **kwargs)
        if hasattr(self, "indexes"):
            res.indexes = self.indexes
        if hasattr(self, "original"):
            res.original = self.original.to(*args, **kwargs)
        return res

    def cuda(self, *args, **kwargs):
        res = super().cuda(*args, **kwargs)
        if hasattr(self, "gray_values"):
            res.gray_values = self.gray_values.to(*args, **kwargs)
        if hasattr(self, "indexes"):
            res.indexes = self.indexes
        if hasattr(self, "original"):
            res.original = self.original.to(*args, **kwargs)
        return res
