from .experiment_base import ExperimentBase
from .load_observables_fn import load_observables_classification_bimonn, load_observables_classification_channel_bimonn
from .load_model_fn import load_model_bimonn_classical_classification
from.enforcers import ArgsClassification, ArgsClassifChannel, ArgsClassifActivation


class ExperimentClassification(ExperimentBase):
    """Experiment class for classificaton task."""


    def __init__(self, *args, **kwargs):
        # kwargs["load_model_fn"] = load_model_bimonn_classical_classification
        kwargs["load_observables_fn"] = load_observables_classification_bimonn
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsClassification(), ArgsClassifActivation()]
        super().__init__(*args, **kwargs)


class ExperimentClassificationChannel(ExperimentBase):
    def __init__(self, *args, **kwargs):
        # kwargs["load_model_fn"] = load_model_bimonn_classical_classification
        kwargs["load_observables_fn"] = load_observables_classification_channel_bimonn
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsClassification(), ArgsClassifChannel(), ArgsClassifActivation()]
        super().__init__(*args, **kwargs)
