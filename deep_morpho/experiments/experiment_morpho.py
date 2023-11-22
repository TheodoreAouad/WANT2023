from typing import List
import pathlib
from .context import Task
from os.path import join

import matplotlib.pyplot as plt

from.enforcers import ArgsMorpho, ArgsDiskorect, ArgsNoisti
from .experiment_base import ExperimentBase
from .load_observables_fn import (
    load_observables_bimonn_morpho_binary, load_observables_bimonn_morpho_grayscale, load_observables_morpho_binary,
    load_observables_ste_morpho_binary
)
# from .load_model_fn import load_model_bimonn_classical

# TODO: improve this kwargs passing for load observables...


class ExperimentMorphoBase(ExperimentBase):
    """Experiment class for learning morphological operators."""

    def __init__(self, *args, **kwargs):
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsMorpho()]
        super().__init__(*args, **kwargs)

    def _check_args(self) -> None:
        super()._check_args()
        assert f"morp_operation{self.args.dataset_args_suffix}" in self.args, f"Argument {f'morp_operation{self.args.dataset_args_suffix}'} is not given"

    def log_tensorboard(self):
        super().log_tensorboard()

        with Task("Logging morphological operations to Tensorboard", self.console_logger):
            self._log_morp_operations()

    def _log_morp_operations(self):
        pathlib.Path(join(self.log_dir, "target_SE")).mkdir(exist_ok=True, parents=True)
        figs_selems = self.target_morp_operation.plot_selem_arrays()
        for (layer_idx, chan_input, chan_output), fig in figs_selems.items():
            fig.savefig(join(self.log_dir, "target_SE", f"target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
            self.tb_logger.experiment.add_figure(f"target_SE/target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig)
            plt.close(fig)

        pathlib.Path(join(self.log_dir, "target_UI")).mkdir(exist_ok=True, parents=True)
        figs_ui = self.target_morp_operation.plot_ui_arrays()
        for (layer_idx, chan_output), fig in figs_ui.items():
            fig.savefig(join(self.log_dir, "target_UI", f"target_UI_l_{layer_idx}_chin_chout_{chan_output}.png"))
            self.tb_logger.experiment.add_figure(f"target_UI/target_UI_l_{layer_idx}_chin_chout_{chan_output}", fig)
            plt.close(fig)

        pathlib.Path(join(self.log_dir, "morp_operations")).mkdir(exist_ok=True, parents=True)
        fig_morp_operation = self.target_morp_operation.vizualise().fig
        fig_morp_operation.savefig(join(self.log_dir, "morp_operations", "morp_operations.png"))
        self.tb_logger.experiment.add_figure("target_operations/morp_operations", fig_morp_operation)
        plt.close(fig_morp_operation)

    def get_experiment_name(self) -> str:
        name = super().get_experiment_name()
        name = join(name, self.target_morp_operation.name)
        return name

    @property
    def target_morp_operation(self):
        return self.args['morp_operation']


class ExperimentMorphoBinary(ExperimentMorphoBase):
    def __init__(self, *args, **kwargs):
        kwargs["load_observables_fn"] = load_observables_morpho_binary
        super().__init__(*args, **kwargs)


class ExperimentNoisti(ExperimentMorphoBinary):
    def __init__(self, *args, **kwargs):
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsNoisti()]
        super().__init__(*args, **kwargs)

    def _check_args(self) -> None:
        pass

    @property
    def target_morp_operation(self):
        return self.trainloader.dataset.default_morp_operation

    def get_experiment_name(self) -> str:
        return ExperimentBase.get_experiment_name(self)


class ExperimentBimonnNoisti(ExperimentNoisti):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_observables_fn = load_observables_bimonn_morpho_binary


class ExperimentSteNoisti(ExperimentNoisti):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_observables_fn = load_observables_ste_morpho_binary


class ExperimentMorphoGrayScale(ExperimentMorphoBase):
    def __init__(self, *args, **kwargs):
        kwargs["load_observables_fn"] = load_observables_bimonn_morpho_grayscale
        super().__init__(*args, **kwargs)


class ExperimentDiskorect(ExperimentMorphoBinary):
    def __init__(self, *args, **kwargs):
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsDiskorect()]
        super().__init__(*args, **kwargs)


class ExperimentBimonnMorphoBinary(ExperimentMorphoBase):
    def __init__(self, *args, **kwargs):
        kwargs["load_observables_fn"] = load_observables_bimonn_morpho_binary
        super().__init__(*args, **kwargs)


class ExperimentSteMorphoBinary(ExperimentMorphoBase):
    def __init__(self, *args, **kwargs):
        kwargs["load_observables_fn"] = load_observables_ste_morpho_binary
        super().__init__(*args, **kwargs)


class ExperimentSteDiskorect(ExperimentSteMorphoBinary):
    def __init__(self, *args, **kwargs):
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsDiskorect()]
        super().__init__(*args, **kwargs)


class ExperimentBimonnMorphoGrayScale(ExperimentMorphoBase):
    def __init__(self, *args, **kwargs):
        kwargs["load_observables_fn"] = load_observables_bimonn_morpho_grayscale
        super().__init__(*args, **kwargs)


class ExperimentBimonnDiskorect(ExperimentBimonnMorphoBinary):
    def __init__(self, *args, **kwargs):
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsDiskorect()]
        super().__init__(*args, **kwargs)
