import pathlib
from os.path import join

from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.tensorboard.summary import custom_scalars
import matplotlib.pyplot as plt
import numpy as np

from ..morp_operations import ParallelMorpOperations
from .observable_layers import ObservableLayers, ObservableLayersChans
from ..models import BiSEBase, BiSELBase
from ..binarization.projection_activated import IterativeProjectionPositive
from ..binarization.projection_constant_set import ProjectionConstantSet
from general.nn.observables import Observable
from general.utils import save_json


operation_code_inverse = {v: k for k, v in BiSEBase.operation_code.items()}


# DEPRECATED
class CheckMorpOperation(ObservableLayers):
    def __init__(self, selems, operations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selems = selems
        self.operations = operations

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        layers = self._get_layers(pl_module)
        default_layout = self.get_layout(layers)
        layout = {"default": default_layout}
        trainer.logger.experiment._get_file_writer().add_summary(custom_scalars(layout))

    @staticmethod
    def get_layout(layers):
        default_layout = {}
        for layer_idx, layer in enumerate(layers):
            if isinstance(
                layer,
            ):
                for bise_idx, bise_layer in enumerate(layer.bises):
                    tags_dilation = [
                        f"comparative/weights/bias_{layer_idx}_{bise_idx}/bias",
                        f"comparative/weights/bias_{layer_idx}_{bise_idx}/dilation_lb",
                        f"comparative/weights/bias_{layer_idx}_{bise_idx}/dilation_ub",
                    ]
                    tags_erosion = [
                        f"comparative/weights/bias_{layer_idx}_{bise_idx}/bias",
                        f"comparative/weights/bias_{layer_idx}_{bise_idx}/erosion_lb",
                        f"comparative/weights/bias_{layer_idx}_{bise_idx}/erosion_ub",
                    ]
                    default_layout.update(
                        {
                            f"dilation_{layer_idx}_{bise_idx}": ["Margin", tags_dilation],
                            f"erosion_{layer_idx}_{bise_idx}": ["Margin", tags_erosion],
                        }
                    )

            else:
                tags_dilation = [
                    f"comparative/weights/bias_{layer_idx}/bias",
                    f"comparative/weights/bias_{layer_idx}/dilation_lb",
                    f"comparative/weights/bias_{layer_idx}/dilation_ub",
                ]
                tags_erosion = [
                    f"comparative/weights/bias_{layer_idx}/bias",
                    f"comparative/weights/bias_{layer_idx}/erosion_lb",
                    f"comparative/weights/bias_{layer_idx}/erosion_ub",
                ]
                default_layout.update(
                    {
                        f"dilation_{layer_idx}": ["Margin", tags_dilation],
                        f"erosion_{layer_idx}": ["Margin", tags_erosion],
                    }
                )

        return default_layout

    def on_train_batch_end_layers(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
    ):
        if isinstance(
            layer,
        ):
            for bise_idx, bise_layer in enumerate(layer.bises):
                self.write_scalars_and_metrics(trainer, bise_layer, f"{layer_idx}_{bise_idx}", 2 * layer_idx + bise_idx)

        elif isinstance(layer, BiSEBase):
            self.write_scalars_and_metrics(trainer, layer, layer_idx, layer_idx)

    def write_scalars_and_metrics(self, trainer, layer, current_name, op_idx):
        erosion_lb, erosion_ub = layer.bias_bounds_erosion(self.selems[op_idx])
        dilation_lb, dilation_ub = layer.bias_bounds_dilation(self.selems[op_idx])

        trainer.logger.experiment.add_scalar(
            f"comparative/weights/bias_{current_name}/bias", -layer.bias, trainer.global_step
        )
        trainer.logger.experiment.add_scalar(
            f"comparative/weights/bias_{current_name}/erosion_lb", erosion_lb, trainer.global_step
        )
        trainer.logger.experiment.add_scalar(
            f"comparative/weights/bias_{current_name}/erosion_ub", erosion_ub, trainer.global_step
        )
        trainer.logger.experiment.add_scalar(
            f"comparative/weights/bias_{current_name}/dilation_lb", dilation_lb, trainer.global_step
        )
        trainer.logger.experiment.add_scalar(
            f"comparative/weights/bias_{current_name}/dilation_ub", dilation_ub, trainer.global_step
        )

        if self.operations[op_idx] == "dilation":
            metrics = {
                f"metrics/bias - lb(op)_{current_name}": -layer.bias - dilation_lb,
                f"metrics/ub(op) - bias_{current_name}": dilation_ub - (-layer.bias),
            }
        elif self.operations[op_idx] == "erosion":
            metrics = {
                f"metrics/bias - lb(op)_{current_name}": -layer.bias - erosion_lb,
                f"metrics/ub(op) - bias_{current_name}": erosion_ub - (-layer.bias),
            }
        else:
            raise NotImplementedError("operation must be dilation or erosion.")

        trainer.logger.log_metrics(metrics, trainer.global_step)


# DEPRECATED
class ShowSelemAlmostBinary(Observable):
    def __init__(self, freq=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = 0
        self.last_selem_and_op = {}

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
    ):
        if self.freq_idx % self.freq == 0:
            selems, operations = pl_module.model.get_bise_selems()
            for layer_idx, layer in enumerate(pl_module.model.layers):
                if not isinstance(layer, BiSEBase):
                    # fig = self.default_figure("Not BiSEBase")
                    continue

                elif selems[layer_idx] is None:
                    continue
                    # fig = self.default_figure("No operation found.")

                else:
                    fig = self.selem_fig(selems[layer_idx], operations[layer_idx])

                trainer.logger.experiment.add_figure(
                    f"learned_selem/almost_binary_{layer_idx}", fig, trainer.global_step
                )
                self.last_selem_and_op[layer_idx] = (selems[layer_idx], operations[layer_idx])
        self.freq_idx += 1

    @staticmethod
    def default_figure(text):
        fig = plt.figure(figsize=(5, 5))
        plt.text(2, 2, text, horizontalalignment="center")
        return fig

    @staticmethod
    def selem_fig(selem, operation):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(selem, interpolation="nearest")
        plt.title(operation)
        return fig

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for layer_idx, (selem, operation) in self.last_selem_and_op.items():
            fig = self.selem_fig(selem, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}.png"))
            plt.close(fig)
            saved.append(fig)

        return saved


class ShowSelemBinary(ObservableLayersChans):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_selem_and_op = {}

    def on_train_batch_end_with_preds_layers_chans(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds,
        layer: "nn.Module",
        layer_idx: int,
        chan_input: int,
        chan_output: int,
    ):
        # with torch.no_grad():
        #     bise_layer.find_selem_and_operation_chan(chan_output, v1=0, v2=1)
        is_activated = layer.is_activated_bise(chin=chan_input, chout=chan_output)
        if not is_activated:
            return

        # selem, operation = bise_layer.learned_selem[chan_output], bise_layer.learned_operation[chan_output]
        selem, operation = layer.get_learned_selem_bise(
            chin=chan_input, chout=chan_output
        ), layer.get_learned_operation_bise(chin=chan_input, chout=chan_output)
        operation = operation_code_inverse[operation]

        fig = self.selem_fig(selem, operation)
        trainer.logger.experiment.add_figure(
            f"learned_selem/binary/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig, trainer.global_step
        )
        self.last_selem_and_op[(layer_idx, chan_input, chan_output)] = (selem, operation)

    @staticmethod
    def selem_fig(selem, operation):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(selem, interpolation="nearest", vmin=0, vmax=1, cmap="gray")
        plt.title(operation)
        return fig

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for (layer_idx, chan_input, chan_output), (selem, operation) in self.last_selem_and_op.items():
            fig = self.selem_fig(selem, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
            plt.close(fig)
            saved.append(fig)

        return saved


class ShowLUISetBinary(ObservableLayersChans):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_set_and_op = {}

    def on_train_batch_end_with_preds_layers_chans(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds,
        layer: "nn.Module",
        layer_idx: int,
        chan_input: int,
        chan_output: int,
    ):
        lui_layer = layer.luis[chan_output]
        if not lui_layer._is_activated[0]:
            return

        C, operation = lui_layer.learned_set[0], lui_layer.learned_operation[0]
        operation = operation_code_inverse[operation]

        fig = self.set_fig(C, operation)
        trainer.logger.experiment.add_figure(
            f"learned_set_lui/binary/layer_{layer_idx}_chout_{chan_output}", fig, trainer.global_step
        )
        self.last_set_and_op[(layer_idx, chan_output)] = (C, operation)

    @staticmethod
    def set_fig(C, operation):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(C[:, None].astype(int), interpolation="nearest", vmin=0, vmax=1, cmap="gray")
        plt.xticks([])
        plt.yticks(range(len(C)))
        plt.title(operation)
        return fig

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for (layer_idx, chan_output), (C, operation) in self.last_set_and_op.items():
            fig = self.set_fig(C, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}_chout_{chan_output}.png"))
            plt.close(fig)
            saved.append(fig)

        return saved


class ShowClosestSelemBinary(ObservableLayersChans):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_elts = {}
        self.last_selems = {}
        self.freq_idx2 = 0

    def on_train_batch_end_with_preds_layers_chans(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds,
        layer: "nn.Module",
        layer_idx: int,
        chan_input: int,
        chan_output: int,
    ):
        with torch.no_grad():
            # layer.bises[chan_input].find_closest_selem_and_operation_chan(chan_output)
            # selem = layer.bises[chan_input].closest_selem[chan_output]
            # distance = layer.bises[chan_input].closest_selem_dist[chan_output]
            # operation = layer.bises[chan_input].closest_operation[chan_output]

            selem = layer.get_closest_selem_bise(chin=chan_input, chout=chan_output)
            distance = layer.get_closest_selem_dist_bise(chin=chan_input, chout=chan_output)
            operation = layer.get_closest_operation_bise(chin=chan_input, chout=chan_output)

            operation = operation_code_inverse[operation]

            # selem, operation, distance = layer.bises[chan_input].find_closest_selem_and_operation_chan(chan_output, v1=0, v2=1)

        trainer.logger.experiment.add_scalar(
            f"comparative/closest_binary_dist/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            distance,
            trainer.global_step,
        )

        fig = self.selem_fig(selem, f"{operation} dist {distance:.2e}")
        trainer.logger.experiment.add_figure(
            f"closest_selem/binary/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig, trainer.global_step
        )
        self.last_elts[str((layer_idx, chan_input, chan_output))] = {"operation": operation, "distance": str(distance)}
        self.last_selems[(layer_idx, chan_input, chan_output)] = selem

    @staticmethod
    def selem_fig(selem, title):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(selem, interpolation="nearest", vmin=0, vmax=1, cmap="gray")
        plt.title(title)
        return fig

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        numpy_dir = join(final_dir, "selem_npy")
        png_dir = join(final_dir, "selem_png")
        pathlib.Path(numpy_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(png_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for (layer_idx, chan_input, chan_output), selem in self.last_selems.items():
            filename = f"layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}"

            elts = self.last_elts[str((layer_idx, chan_input, chan_output))]
            operation, distance = elts["operation"], elts["distance"]

            fig = self.selem_fig(selem, f"{operation} dist {distance}")
            fig.savefig(join(png_dir, f"{filename}.png"))
            plt.close(fig)
            saved.append(fig)
            np.save(join(numpy_dir, f"{filename}.npy"), selem.astype(np.uint8))

        save_json(self.last_elts, join(final_dir, "operation_distance.json"))
        saved.append(self.last_elts)

        return saved


# class DistToSelem(ObservableLayersChans):

#     def __init__(self, target_morp_operation: ParallelMorpOperations, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.target_morp_operation = target_morp_operation

#     def on_train_batch_end_with_preds_layers_chans(
#         self,
#         trainer: 'pl.Trainer',
#         pl_module: 'pl.LightningModule',
#         outputs: "STEP_OUTPUT",
#         batch: "Any",
#         batch_idx: int,
#         preds,
#         layer: "nn.Module",
#         layer_idx: int,
#         chan_input: int,
#         chan_output: int,
#     ):
#         selem = self.target_morp_operation.get_selem(layer_idx=layer_idx, chan_input=chan_input, chan_output=chan_output)
#         operation = self.target_morp_operation.get_operation_name(layer_idx=layer_idx, chan_input=chan_input, chan_output=chan_output)
#         weight = layer.get_weight_bise(chin=chan_input, chout=chan_output).detach().cpu().numpy()
#         bias = layer.get_bias_bise(chin=chan_input, chout=chan_output).detach().cpu().numpy()

#         dist = IterativeProjectionPositive(Wini=weight, bini=-bias, S=selem, operation=operation)

#             # layer.bises[chan_input].find_closest_selem_and_operation_chan(chan_output)
#             # selem = layer.bises[chan_input].closest_selem[chan_output]
#             # distance = layer.bises[chan_input].closest_selem_dist[chan_output]
#             # operation = layer.bises[chan_input].closest_operation[chan_output]

#             selem = layer.get_closest_selem_bise(chin=chan_input, chout=chan_output)
#             distance = layer.get_closest_selem_dist_bise(chin=chan_input, chout=chan_output)
#             operation = layer.get_closest_operation_bise(chin=chan_input, chout=chan_output)

#             operation = operation_code_inverse[operation]

#             # selem, operation, distance = layer.bises[chan_input].find_closest_selem_and_operation_chan(chan_output, v1=0, v2=1)

#         trainer.logger.experiment.add_scalar(f"comparative/closest_binary_dist/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}", distance, trainer.global_step)

#         fig = self.selem_fig(selem, f"{operation} dist {distance:.2e}")
#         trainer.logger.experiment.add_figure(f"closest_selem/binary/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig, trainer.global_step)
#         self.last_elts[str((layer_idx, chan_input, chan_output))] = {"operation": operation, "distance": str(distance)}
#         self.last_selems[(layer_idx, chan_input, chan_output)] = selem


class DistToMorpOperation(Observable):
    def __init__(
        self,
        target_morp_operation: ParallelMorpOperations,
        freq={"batch": 1, "epoch": 1},
        layers=None,
        layer_name="layers",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.last = {}
        self.target_morp_operation = target_morp_operation
        self.last_luis = {}
        self.last_bises = {}
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
        ui_arrays = self.target_morp_operation.ui_arrays

        self.last_bises = {}
        self.last_luis = {}

        for layer_idx, layer in enumerate(layers):
            if not isinstance(layer, BiSELBase):
                continue

            for chout in range(layer.out_channels):
                for chin in range(layer.in_channels):
                    selem = self.target_morp_operation.get_selem(
                        layer_idx=layer_idx, chan_input=chin, chan_output=chout
                    ).astype(bool)
                    operation = self.target_morp_operation.get_operation_name(
                        layer_idx=layer_idx, chan_input=chin, chan_output=chout
                    )
                    weight = layer.get_weight_bise(chin=chin, chout=chout).detach().cpu().numpy()
                    bias = layer.get_bias_bise(chin=chin, chout=chout).detach().cpu().numpy()

                    dist_activated = (
                        IterativeProjectionPositive(Wini=weight, bini=-bias, S=selem, operation=operation).solve().value
                    )
                    dict_constant = ProjectionConstantSet.distance_fn_selem(
                        weights=weight.reshape(-1)[None, :], S=selem.reshape(-1)[None, :]
                    )[0]

                    self.last_bises[(layer_idx, chout, chin)] = {"activated": dist_activated, "constant": dict_constant}

                    trainer.logger.experiment.add_scalars(
                        f"dist_proj/{batch_or_epoch}/bise_layer_{layer_idx}_chout_{chout}_chin_{chin}",
                        self.last_bises[(layer_idx, chout, chin)],
                        step,
                    )

                operation, selem = ui_arrays[layer_idx][chout]
                selem = selem.astype(bool)
                weight = layer.get_coef_lui(chout=chout).detach().cpu().numpy()
                bias = layer.bias_lui[chout].detach().cpu().numpy()

                dist_activated = (
                    IterativeProjectionPositive(Wini=weight, bini=-bias, S=selem, operation=operation).solve().value
                )
                dict_constant = ProjectionConstantSet.distance_fn_selem(
                    weights=weight.reshape(-1)[None, :], S=selem.reshape(-1)[None, :]
                )[0]

                self.last_luis[(layer_idx, chout)] = {"activated": dist_activated, "constant": dict_constant}
                trainer.logger.experiment.add_scalars(
                    f"dist_proj/{batch_or_epoch}/lui_layer_{layer_idx}_chout_{chout}",
                    self.last_luis[(layer_idx, chout)],
                    step,
                )

        # self.last.update(self.last_bises)
        # self.last.update(self.last_luis)

        trainer.logger.experiment.add_histogram(
            f"dist_proj_to_activation/{batch_or_epoch}/bise",
            np.array([v["activated"] for v in self.last_bises.values()]),
            step,
        )

        trainer.logger.experiment.add_histogram(
            f"dist_proj_to_activation/{batch_or_epoch}/lui",
            np.array([v["activated"] for v in self.last_luis.values()]),
            step,
        )

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        # dict_str = {}
        # for k1, v1 in self.last.items():
        #     dict_str[f"{k1}"] = {}
        #     for k2, v2 in v1.items():
        #         dict_str[f"{k1}"][f"{k2}"] = str(v2)
        last_bises_str = {}
        for k1, v1 in self.last_bises.items():
            last_bises_str[f"{k1}"] = v1

        last_luis_str = {}
        for k1, v1 in self.last_luis.items():
            last_luis_str[f"{k1}"] = v1

        save_json(last_bises_str, join(final_dir, "dist_proj_bise.json"))
        save_json(last_luis_str, join(final_dir, "dist_proj_lui.json"))
        return self

    def save_hparams(self) -> dict:
        res = {}
        res["dist_proj_bise_activated"] = np.mean([v["activated"] for v in self.last_bises.values()])
        res["dist_proj_bise_constant"] = np.mean([v["constant"] for v in self.last_bises.values()])
        res["dist_proj_lui_activated"] = np.mean([v["activated"] for v in self.last_luis.values()])
        res["dist_proj_lui_constant"] = np.mean([v["constant"] for v in self.last_luis.values()])

        return res

    def _get_layers(self, pl_module):
        if self.layers is not None:
            return self.layers

        if hasattr(pl_module.model, self.layer_name):
            return getattr(pl_module.model, self.layer_name)

        raise NotImplementedError("Cannot automatically select layers for model. Give them manually.")
