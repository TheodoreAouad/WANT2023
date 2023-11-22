from abc import ABC
from .bimonn import (
    BiMoNN, BiMoNNClassifierMaxPool, BiMoNNClassifierMaxPoolNotBinary, BiMoNNClassifierLastLinearNotBinary,
    BiMoNNClassifierLastLinear
)
from .specific_bimonn import BimonnDense, BimonnDenseNotBinary, BimonnBiselDenseNotBinary
from .generic_lightning_model import GenericLightningModel


class LightningBiMoNN(GenericLightningModel):
    model_class = BiMoNN

    @classmethod
    def get_model_from_experiment(cls, experiment: "ExperimentBase") -> GenericLightningModel:
        args = experiment.args
        inpt = experiment.input_sample

        if isinstance(args["initializer_args"], dict):
            args["initializer_args"]["input_mean"] = inpt.mean().item()
        elif isinstance(args["initializer_args"], list):
            args["initializer_args"][0]["input_mean"] = inpt.mean().item()

        model_args = args.model_args()

        model_args.update({
            "kernel_size": [args['kernel_size'] for _ in range(args['n_atoms'])],
            "atomic_element": args["atomic_element"].replace('dual_', ''),
            "lui_kwargs": {"force_identity": args['force_lui_identity']},
        })

        model = cls(
            model_args=model_args,
            learning_rate=args["learning_rate"],
            loss=args["loss"],
            optimizer=args["optimizer"],
            optimizer_args=args["optimizer_args"],
            observables=experiment.observables,
        )
        model.to(experiment.device)
        return model


class LightningBiMoNNClassifier(LightningBiMoNN, ABC):

    @classmethod
    def get_model_from_experiment(cls, experiment: "ExperimentBase") -> GenericLightningModel:
        args = experiment.args
        inpt = experiment.input_sample

        if isinstance(args["initializer_args"], dict):
            args["initializer_args"]["input_mean"] = inpt.mean().item()
        elif isinstance(args["initializer_args"], list):
            args["initializer_args"][0]["input_mean"] = inpt.mean().item()

        model_args = args.model_args()

        model_args.update({
            "input_size": inpt.shape[1:],
            "n_classes": experiment.trainloader.dataset.n_classes,
            "input_mean": inpt.mean().item(),
        })

        model = cls(
            model_args=model_args,
            learning_rate=args["learning_rate"],
            loss=args["loss"],
            optimizer=args["optimizer"],
            optimizer_args=args["optimizer_args"],
            observables=experiment.observables,
        )
        model.to(experiment.device)
        return model



class LightningBiMoNNClassifierMaxPool(LightningBiMoNNClassifier):
    model_class = BiMoNNClassifierMaxPool


class LightningBiMoNNClassifierMaxPoolNotBinary(LightningBiMoNNClassifier):
    model_class = BiMoNNClassifierMaxPoolNotBinary


class LightningBiMoNNClassifierLastLinearNotBinary(LightningBiMoNNClassifier):
    model_class = BiMoNNClassifierLastLinearNotBinary


class LightningBiMoNNClassifierLastLinear(LightningBiMoNNClassifier):
    model_class = BiMoNNClassifierLastLinear


class LightningBimonnDense(LightningBiMoNNClassifier):
    model_class = BimonnDense

    @classmethod
    def get_model_from_experiment(cls, experiment: "ExperimentBase") -> GenericLightningModel:
        args = experiment.args
        inpt = experiment.input_sample

        args["initializer_method"] = args["initializer_args"]["bise_init_method"]
        args["initializer_args"] = args["initializer_args"]["bise_init_args"]

        model_args = args.model_args()

        model_args.update({
            "input_size": inpt.shape[1:],
            "n_classes": experiment.trainloader.dataset.n_classes,
            "input_mean": inpt.mean().item(),
        })

        model = cls(
            model_args=model_args,
            learning_rate=args["learning_rate"],
            loss=args["loss"],
            optimizer=args["optimizer"],
            optimizer_args=args["optimizer_args"],
            observables=experiment.observables,
        )
        model.to(experiment.device)
        return model


class LightningBimonnDenseNotBinary(LightningBimonnDense):
    model_class = BimonnDenseNotBinary


class LightningBimonnBiselDenseNotBinary(LightningBiMoNNClassifier):
    model_class = BimonnBiselDenseNotBinary

    @classmethod
    def get_model_from_experiment(cls, experiment: "ExperimentBase") -> GenericLightningModel:
        args = experiment.args
        inpt = experiment.input_sample

        args["initializer_bise_method"] = args["initializer_args"]["bise_init_method"]
        args["initializer_bise_args"] = args["initializer_args"]["bise_init_args"]
        args["initializer_lui_method"] = args["initializer_args"]["lui_init_method"]

        model_args = args.model_args()

        model_args.update({
            "input_size": inpt.shape[1:],
            "n_classes": experiment.trainloader.dataset.n_classes,
            "input_mean": inpt.mean().item(),
        })

        model = cls(
            model_args=model_args,
            learning_rate=args["learning_rate"],
            loss=args["loss"],
            optimizer=args["optimizer"],
            optimizer_args=args["optimizer_args"],
            observables=experiment.observables,
        )
        model.to(experiment.device)
        return model
