from general.nn.observables import Observable


class UpdateBinary(Observable):
    """Update binary parameters at freq for training batch, and at the end of every training epoch."""

    def __init__(self, freq_batch=None, freq_epoch=1,):
        super().__init__()
        self.freq_batch = freq_batch
        self.freq_epoch = freq_epoch

        self.freq_batch_idx = 0
        self.freq_epoch_idx = 0

    def on_train_batch_end_with_preds(self, trainer, pl_module, *args, **kwargs):
        if self.freq_batch is not None and self.freq_batch_idx % self.freq_batch == 0:
            self.apply_update(pl_module)
        self.freq_batch_idx += 1

    def apply_update(self, pl_module):
        if hasattr(pl_module.model, 'binary_mode'):
            init_mode = pl_module.model.binary_mode
            pl_module.model.binary(mode=True, update_binaries=True)  # Updating all binaries
            pl_module.model.binary(mode=init_mode, update_binaries=False)  # Resetting original binary mode

    def on_train_epoch_end(self, trainer, pl_module):
        # self.apply_update(pl_module)
        if self.freq_epoch is not None and self.freq_epoch_idx % self.freq_epoch == 0:
            self.apply_update(pl_module)
        self.freq_epoch_idx += 1
