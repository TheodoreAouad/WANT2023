from deep_morpho.models import GenericLightningModel


def default_load_model_fn(experiment: "ExperimentBase",) -> GenericLightningModel:
    model = GenericLightningModel.select(experiment.args["model"]).get_model_from_experiment(experiment)
    return model


def load_model_bimonn_classical_classification(experiment: "ExperimentBase") -> GenericLightningModel:
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

    model = GenericLightningModel.select(args["model"])(
        model_args=model_args,
        learning_rate=args["learning_rate"],
        loss=args["loss"],
        optimizer=args["optimizer"],
        optimizer_args=args["optimizer_args"],
        observables=experiment.observables,
        # reduce_loss_fn=args["reduce_loss_fn"],
        # initializer=args["initializer"],
        # initializer_args=args["initializer_args"],
    )
    model.to(experiment.device)
    return model
