from copy import deepcopy
from typing import Callable, Dict, Tuple, Union

from ..extend_signature_and_forward import extend_signature_and_forward



class LossHandler:
    """ Basic class for handling losses. The losses can be given in multiple ways.
    If a callable is given, loss will be this callable. We can also give a constructor and the arguments.
    If the constructor is given, then the model will also be given as argument. This can be used for losses
    using arguments of the model (e.g. regularization loss on weights). Finally, we can also give a
    dictionary if multiple losses are needed.
    >>> loss = nn.MSELoss()
    >>> loss = (nn.MSELoss, {size_average=True})
    >>> loss = {"mse": (nn.MSELoss, {size_average=True}), "regu": (MyReguLoss, {"lambda_": lambda_})}
    """
    def __init__(
        self,
        loss: Union[Callable, Tuple[Callable, Dict], Dict[str, Union[Callable, Tuple[Callable, Dict]]]],
        # reduce_loss_fn: Callable = lambda x: reduce(lambda a, b: a + b, x),
        coefs: Dict[str, float] = None,
        do_compute: Dict[str, bool] = None,
        # pl_module: "pl.LightningModule" = None,
    ):
        self.loss_args = loss
        self.loss = self.instantiate(loss)
        self.coefs = self.configure_coefs(coefs)
        self._do_compute = do_compute
        # if do_compute is None:
        #     do_compute = {k: True for k in self.loss.keys()}
        # self.do_compute = do_compute
        # self.pl_module = pl_module

    @property
    def do_compute(self):
        if self._do_compute is not None:
            return self._do_compute

        return {k: v != 0 for k, v in self.coefs.items()}


    def configure_coefs(self, coefs: Dict[str, float] = None) -> Dict[str, float]:
        """Configures the coefficients of the losses.
        """
        if not isinstance(self.loss, dict):
            return None

        coefs = coefs.copy() if coefs is not None else {}

        for key in self.loss.keys():
            if key not in coefs.keys():
                coefs[key] = 1.0

        return coefs

    def instantiate(self, loss):
        """Instantiates the loss if needed. Avoids the key "loss" to be in the loss dict.
        Ex:
        >>> loss = self.instantiate({"loss": nn.MSELoss()})
        {"loss_0": nn.MSELoss()}
        >>> loss = self.instantiate({"loss": {"mse": nn.MSELoss(), "regu": (MyReguLoss, {"lambda_": lambda_})}})
        {"loss_0": {"mse": nn.MSELoss(), "regu": (MyReguLoss, {"lambda_": lambda_})}}
        >>> loss = self.instantiate({"loss": nn.MSELoss(), "loss_0": nn.CrossEntropyLoss()})
        {"loss_1": nn.MSELoss(), "loss_0": nn.CrossEntropyLoss()}
        """
        if isinstance(loss, dict):
            loss = deepcopy(loss)

            if "loss" in loss.keys():
                i = 0
                while f"loss_{i}" in loss.keys():
                    i += 1
                loss[f"loss_{i}"] = loss["loss"]
            # return loss

            return {k: self.instantiate(v) for (k, v) in loss.items()}

        # Ensure that any extra argument can be given without errors
        if isinstance(loss, tuple):
            return extend_signature_and_forward(loss[0](model=self.model, **loss[1]))

        return extend_signature_and_forward(loss)

    # def log_loss(self, values: dict, state: str = "") -> None:
    #     """
    #     Args:
    #         state (str): state of the loss (train, val, test) for the logs
    #     """
    #     for key, value in values.items():
    #         self.log(f"loss{state}/{key}", value.item())  # put .item() to avoid memory leak

    def compute_loss(self, ypred, ytrue, *args, **kwargs) -> dict:
        """Computes total loss for each component of the loss.
        Args:
            ypred: predictions
            ytrue: ground truth

        Returns:
            values: dict of the losses containing the total loss and the loss for each component, as well as the grads.
            Key "loss" will be backpropagated on.
        """
        values = {}
        assert not ypred.isnan().any(), "NaN in prediction"
        if isinstance(self.loss, dict):
            total_loss = 0
            for key, loss_fn in self.loss.items():
                if self.do_compute is not None and not self.do_compute[key]:
                    continue
                values[key] = loss_fn(ypred, ytrue, *args, **kwargs)
                total_loss += self.coefs[key] * values[key]

            values["loss"] = total_loss
            # values["loss"] = self.reduce_loss_fn(values.values())

        else:
            values["loss"] = self.loss(ypred, ytrue, *args, **kwargs)

        # grad_values = {}
        for key, value in values.items():
            # if key == "loss":
            #     continue

            if value.requires_grad:
                value.retain_grad()  # see graph of each loss term
            # if value.requires_grad:
            #     grad_values[f'{key}_grad'] = value.grad
            #     values[key] = values[key].detach()

        # values['grads'] = grad_values

        return values

    def __call__(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)
