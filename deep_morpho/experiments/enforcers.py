from abc import ABC, abstractmethod

from torch.nn import CrossEntropyLoss, BCELoss

from deep_morpho.loss import BCENormalizedLoss
from deep_morpho.datasets import NoistiDataset

from general.nn.experiments.experiment_methods import ExperimentMethods


class ArgsEnforcer(ExperimentMethods, ABC):
    def __init__(self):
        self.enforcers = []
        self.add_enforcer()

    def enforce(self, experiment: "ExperimentBase"):
        for enforce_fn in self.enforcers:
            enforce_fn(experiment)

    @abstractmethod
    def add_enforcer(self):
        pass


class ArgsMorpho(ArgsEnforcer):
    def add_enforcer(self):
        def enforce_fn(experiment):
            if experiment.args["kernel_size"] == "adapt":
                experiment.args["kernel_size"] = int(max(experiment.args["morp_operation"].max_selem_shape))

            if experiment.args["channels"] == "adapt":
                experiment.args["channels"] = experiment.args["morp_operation"].in_channels + [
                    experiment.args["morp_operation"].out_channels[-1]
                ]

            if experiment.args["n_atoms"] == "adapt":
                experiment.args["n_atoms"] = len(experiment.args["morp_operation"])

            # experiment.args["model"] = "BiMoNN"

        self.enforcers.append(enforce_fn)


class ArgsSymetricBinary(ArgsEnforcer):
    def add_enforcer(self):
        def enforce_fn(experiment):
            experiment.args["do_symetric_output"] = True

            if experiment.args["loss_data_str"] == "BCELoss":
                experiment.args["loss_data_str"] = "BCENormalizedLoss"
                experiment.args["kwargs_loss"].update({"vmin": -1, "vmax": 1})
                experiment.args["loss_data"] = BCENormalizedLoss(**experiment.args["kwargs_loss"])
                experiment.args["loss"] = {"loss_data": experiment.args["loss_data"]}

        self.enforcers.append(enforce_fn)


class ArgsNotMorpho(ArgsEnforcer):
    def add_enforcer(self):
        def enforce_fn(experiment: "ExperimentBase"):
            experiment.args["morp_operation"] = None

        self.enforcers.append(enforce_fn)


class ArgsGeneration(ArgsEnforcer):
    def add_enforcer(self):
        def enforce_fn(experiment: "ExperimentBase"):
            experiment.args["n_inputs.train"] = experiment.args["n_steps"] * experiment.args["batch_size"]
            experiment.args["n_inputs.val"] = experiment.args["batch_size"]
            experiment.args["n_inputs.test"] = experiment.args["batch_size"]

        self.enforcers.append(enforce_fn)


class ArgsNoisti(ArgsGeneration):
    def add_enforcer(self):
        super().add_enforcer()

        def enforce_fn(experiment):
            experiment.args["angles"] = experiment.args["sticks_noised_angles"]
            for key, value in experiment.args["sticks_noised_args"].items():
                experiment.args[key] = value

            experiment.args["morp_operation"] = NoistiDataset.get_default_morp_operation(
                lengths_lim=experiment.args["lengths_lim"], angles=experiment.args["angles"]
            )

        self.enforcers.append(enforce_fn)


class ArgsDiskorect(ArgsGeneration):
    def add_enforcer(self):
        super().add_enforcer()

        def enforce_fn(experiment: "ExperimentBase"):
            experiment.args["random_gen_args"] = experiment.args["random_gen_args"].copy()
            experiment.args["random_gen_args"]["size"] = experiment.args["random_gen_args"]["size"] + (
                experiment.args["morp_operation"].in_channels[0],
            )

        self.enforcers.append(enforce_fn)


class ArgsClassification(ArgsEnforcer):
    def add_enforcer(self):
        def enforce_fn(experiment: "ExperimentBase"):
            if experiment.args["n_atoms"] == "adapt":
                experiment.args["n_atoms"] = len(experiment.args["channels"]) - 1

        self.enforcers.append(enforce_fn)


class ArgsMnist(ArgsEnforcer):
    def add_enforcer(self):
        def enforce_fn(experiment: "ExperimentBase"):
            experiment.args["n_inputs.train"] = 50_000
            experiment.args["n_inputs.val"] = 10_000
            experiment.args["n_inputs.test"] = 10_000

            # warnings.warn("DEBUG" + __file__)
            # experiment.args["n_inputs.train"] = 1_000
            # experiment.args["n_inputs.val"] = 1_000
            # experiment.args["n_inputs.test"] = 1_000

        self.enforcers.append(enforce_fn)


class ArgsCifar(ArgsEnforcer):
    def add_enforcer(self):
        def enforce_fn(experiment: "ExperimentBase"):
            import torchvision.transforms as transforms
            from deep_morpho.datasets.cifar_dataset import transform_default

            experiment.args["n_inputs.train"] = 45_000
            experiment.args["n_inputs.val"] = 5_000
            experiment.args["n_inputs.test"] = 10_000
            experiment.args["transform.train"] = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=10),
                    transform_default,
                ]
            )

        self.enforcers.append(enforce_fn)


class ArgsClassifActivation(ArgsEnforcer):
    def add_enforcer(self):
        def enforce_fn(experiment: "ExperimentBase"):
            args = experiment.args
            if "apply_last_activation.net" in args:
                if not args["apply_last_activation"] and args["loss_data_str"] == "BCELoss":
                    args.update(
                        {
                            "loss_data_str": "CrossEntropyLoss",
                            "loss_data": CrossEntropyLoss(),
                            "loss": {"loss_data": CrossEntropyLoss()},
                        }
                    )

                elif args["apply_last_activation"] and args["loss_data_str"] == "CrossEntropyLoss":
                    args.update({"loss_data_str": "BCELoss", "loss_data": BCELoss(), "loss": {"loss_data": BCELoss()}})

        self.enforcers.append(enforce_fn)


class ArgsClassifChannel(ArgsEnforcer):
    def add_enforcer(self):
        def enforce_fn(experiment: "ExperimentBase"):
            experiment.args["levelset_handler_mode"] = experiment.args["channel_classif_args"]["levelset_handler_mode"]
            experiment.args["levelset_handler_args"] = experiment.args["channel_classif_args"]["levelset_handler_args"]

        self.enforcers.append(enforce_fn)
