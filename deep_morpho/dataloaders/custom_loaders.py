from torch.utils.data import DataLoader


class DataLoaderEpochTracker(DataLoader):
    """ A DataLoader that tracks the current epoch and gives it to the dataset.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = -1

    def __iter__(self):
        self.epoch += 1
        self.dataset.epoch = self.epoch
        return super().__iter__()
