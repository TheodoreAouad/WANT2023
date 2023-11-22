from typing import Any, Dict
from os.path import join
import pathlib

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from general.nn.observables import Observable
from deep_morpho.models import BiSEBase
from general.utils import save_json
from deep_morpho.models import NotBinaryNN


class ActivatednessObservable(Observable):

    def __init__(self, freq={"batch": 1, "epoch": 1}, layers=None, layer_name="layers", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last = {}
        self.freq = freq
        self.freq_idx = {"batch": 0, "epoch": 0}
        self.layers = layers
        self.layer_name = layer_name

    def on_train_epoch_end(self, trainer, pl_module):
        if self.freq["epoch"] is None or self.freq_idx["epoch"] % self.freq["epoch"] != 0:
            self.freq_idx["epoch"] += 1
            return
        self.freq_idx["epoch"] += 1

        self.log_tb(trainer, pl_module, batch_or_epoch="epoch", step=trainer.current_epoch)

    def on_train_batch_end_with_preds(
        self,
        trainer,
        pl_module,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        preds: Any,
    ):
        if self.freq["batch"] is None or self.freq_idx["batch"] % self.freq["batch"] != 0:
            self.freq_idx["batch"] += 1
            return
        self.freq_idx["batch"] += 1

        self.log_tb(trainer, pl_module, batch_or_epoch="batch", step=trainer.global_step)

    def log_tb(self, trainer, pl_module, batch_or_epoch: str, step: int):
        layers = self._get_layers(pl_module)


        self.last["all"] = {
            "n_bise_dilation": 0, "n_bise_erosion": 0, "n_bise_activated": 0, "n_bise_total": 0,
            "n_params_activated": 0, "n_params_total": 0,
        }
        for layer_idx, layer in enumerate(layers):
            self.last[layer_idx] = {
                "n_bise_dilation": 0, "n_bise_erosion": 0, "n_bise_activated": 0, "n_bise_total": 0,
                "n_params_activated": 0, "n_params_total": 0,
            }
            for module_ in layer.modules():
                if isinstance(module_, BiSEBase) and not isinstance(module_, NotBinaryNN):
                    # is_activated = module_.is_activated
                    # self.last[layer_idx]["n_dilation"] += (module_.learned_operation[is_activated] == module_.operation_code["dilation"]).sum()
                    # self.last[layer_idx]["n_erosion"] += (module_.learned_operation[is_activated] == module_.operation_code["erosion"]).sum()

                    # self.last[layer_idx]["n_activated"] += self.last[layer_idx]["n_dilation"] + self.last[layer_idx]["n_erosion"]
                    # self.last[layer_idx]["n_total"] += len(is_activated)
                    self.last[layer_idx]["n_bise_dilation"] += module_.n_dilation_activated
                    self.last[layer_idx]["n_bise_erosion"] += module_.n_erosion_activated
                    self.last[layer_idx]["n_bise_activated"] += module_.n_bise_activated
                    self.last[layer_idx]["n_bise_total"] += module_.n_bise

                    self.last[layer_idx]["n_params_activated"] += module_.n_params_activated
                    self.last[layer_idx]["n_params_total"] += module_.numel_float()

            for key, value in self.last[layer_idx].items():
                self.last["all"][key] += value

            self.last[layer_idx].update({
                "ratio_bise_dilation": self.last[layer_idx]["n_bise_dilation"] / (self.last[layer_idx]["n_bise_total"] + 1e-5),
                "ratio_bise_erosion": self.last[layer_idx]["n_bise_erosion"] / (self.last[layer_idx]["n_bise_total"] + 1e-5),
                "ratio_bise_activated": self.last[layer_idx]["n_bise_activated"] / (self.last[layer_idx]["n_bise_total"] + 1e-5),
                "ratio_params_activated": self.last[layer_idx]["n_params_activated"] / (self.last[layer_idx]["n_params_total"] + 1e-5),
            })

            for key, value in self.last[layer_idx].items():
                trainer.logger.experiment.add_scalar(f"activatedness_details/{batch_or_epoch}/{key}/layer_{layer_idx}", value, step)
                # pl_module.log(f"activatedness/layer_{layer_idx}/{key}", value,)


        self.last["all"].update({
            "ratio_bise_dilation": self.last["all"]["n_bise_dilation"] / (self.last["all"]["n_bise_total"] + 1e-5),
            "ratio_bise_erosion": self.last["all"]["n_bise_erosion"] / (self.last["all"]["n_bise_total"] + 1e-5),
            "ratio_bise_activated": self.last["all"]["n_bise_activated"] / (self.last["all"]["n_bise_total"] + 1e-5),
            "ratio_params_activated": self.last["all"]["n_params_activated"] / (self.last["all"]["n_params_total"] + 1e-5),
        })

        for key, value in self.last["all"].items():
            trainer.logger.experiment.add_scalar(f"activatedness/all/{batch_or_epoch}/{key}", value, step)


    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        dict_str = {}
        for k1, v1 in self.last.items():
            dict_str[f"{k1}"] = {}
            for k2, v2 in v1.items():
                dict_str[f"{k1}"][f"{k2}"] = str(v2)
        save_json(dict_str, join(final_dir, "activatedness.json"))
        return self.last


    def save_hparams(self) -> Dict:
        res = {}
        for layer, dict_key in self.last.items():
            for key, value in dict_key.items():
                res[f"{key}_{layer}"] = value
        return res


    def _get_layers(self, pl_module):

        if self.layers is not None:
            return self.layers

        if hasattr(pl_module.model, self.layer_name):
            return getattr(pl_module.model, self.layer_name)


        raise NotImplementedError('Cannot automatically select layers for model. Give them manually.')


class ClosestDistObservable(Observable):

    def __init__(self, freq={"epoch": 1, "batch": 1}, layers=None, layer_name="layers", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last = {}
        self.freq = freq
        self.freq_idx = {"epoch": 0, "batch": 0}
        self.layers = layers
        self.layer_name = layer_name

    def on_train_epoch_end(self, trainer, pl_module):
        if self.freq["epoch"] is None or self.freq_idx["epoch"] % self.freq["epoch"] != 0:
            self.freq_idx["epoch"] += 1
            return
        self.freq_idx["epoch"] += 1

        self.log_tb(trainer, pl_module, batch_or_epoch="epoch", step=trainer.current_epoch)

    def on_train_batch_end_with_preds(
        self,
        trainer,
        pl_module,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        preds: Any,
    ):
        if self.freq["batch"] is None or self.freq_idx["batch"] % self.freq["batch"] != 0:
            self.freq_idx["batch"] += 1
            return
        self.freq_idx["batch"] += 1

        self.log_tb(trainer, pl_module, batch_or_epoch="batch", step=trainer.global_step)


    def log_tb(self, trainer, pl_module, batch_or_epoch: str, step: int):
        layers = self._get_layers(pl_module)


        self.last["all"] = {"closest_dist": np.array([]), "n_dilation": 0, "n_erosion": 0, "n_total": 0, "n_closest": 0}
        for layer_idx, layer in enumerate(layers):
            self.last[layer_idx] = {"closest_dist": np.array([]), "n_dilation": 0, "n_erosion": 0, "n_total": 0, "n_closest": 0}
            for module_ in layer.modules():
                if isinstance(module_, BiSEBase) and not isinstance(module_, NotBinaryNN):
                    self.last[layer_idx]["closest_dist"] = np.concatenate([self.last[layer_idx]["closest_dist"], module_.closest_selem_dist])

                    is_activated = module_.is_activated
                    self.last[layer_idx]["n_closest"] += len(is_activated) - is_activated.sum()

                    self.last[layer_idx]["n_dilation"] += (module_.closest_operation[~is_activated] == module_.operation_code["dilation"]).sum()
                    self.last[layer_idx]["n_erosion"] += (module_.closest_operation[~is_activated] == module_.operation_code["erosion"]).sum()

                    self.last[layer_idx]["n_total"] += len(is_activated)

            self.last["all"]["closest_dist"] = np.concatenate([self.last["all"]["closest_dist"], self.last[layer_idx]["closest_dist"]])

            for key, value in self.last[layer_idx].items():
                if key == "closest_dist":
                    continue
                self.last["all"][key] += value

            self.last[layer_idx].update({
                "ratio_dilation": self.last[layer_idx]["n_dilation"] / (self.last[layer_idx]["n_total"] + 1e-5),
                "ratio_erosion": self.last[layer_idx]["n_erosion"] / (self.last[layer_idx]["n_total"] + 1e-5),
                "ratio_closest": self.last[layer_idx]["n_closest"] / (self.last[layer_idx]["n_total"] + 1e-5),
            })

            for key, value in self.last[layer_idx].items():
                if key == "closest_dist":
                    continue
                trainer.logger.experiment.add_scalar(f"closest_details/{batch_or_epoch}/{key}/layer_{layer_idx}", value, trainer.current_epoch)
                # pl_module.log(f"closest/layer_{layer_idx}/{key}", value,)

            if len(self.last[layer_idx]["closest_dist"]) > 0:
                trainer.logger.experiment.add_histogram(f"closest_details/{batch_or_epoch}/closest_dist/layer_{layer_idx}", self.last[layer_idx]["closest_dist"], trainer.current_epoch)

            for key, value in self.last["all"].items():
                self.last["all"][key] += value

        self.last["all"].update({
            "ratio_dilation": self.last["all"]["n_dilation"] / (self.last["all"]["n_total"] + 1e-5),
            "ratio_erosion": self.last["all"]["n_erosion"] / (self.last["all"]["n_total"] + 1e-5),
            "ratio_closest": self.last["all"]["n_closest"] / (self.last["all"]["n_total"] + 1e-5),
        })

        for key, value in self.last["all"].items():
            if key == "closest_dist":
                continue
            trainer.logger.experiment.add_scalar(f"closest/all/{batch_or_epoch}/{key}", value, trainer.current_epoch)
            # pl_module.log(f"activatedness/all/{key}", value,)

        if len(self.last["all"]["closest_dist"]) > 0:
            trainer.logger.experiment.add_histogram(f"closest/all/{batch_or_epoch}/closest_dist", self.last["all"]["closest_dist"], trainer.current_epoch)

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        dict_str = {}
        for k1, v1 in self.last.items():
            dict_str[f"{k1}"] = {}
            for k2, v2 in v1.items():
                if k2 == "closest_dist":
                    continue
                dict_str[f"{k1}"][f"{k2}"] = str(v2)
        save_json(dict_str, join(final_dir, "closest_dist.json"))
        return self.last


    def save_hparams(self) -> Dict:
        res = {}
        for layer, dict_key in self.last.items():
            for key, value in dict_key.items():
                if key == "closest_dist":
                    continue
                res[f"{key}_{layer}"] = value
        return res


    def _get_layers(self, pl_module):

        if self.layers is not None:
            return self.layers

        if hasattr(pl_module.model, self.layer_name):
            return getattr(pl_module.model, self.layer_name)


        raise NotImplementedError('Cannot automatically select layers for model. Give them manually.')
