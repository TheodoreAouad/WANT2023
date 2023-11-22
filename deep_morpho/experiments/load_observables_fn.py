from os.path import join
from typing import Tuple

import numpy as np

from general.nn.observables import CalculateAndLogMetrics
from deep_morpho.metrics import dice, accuracy
import deep_morpho.observables as obs
from pytorch_lightning.callbacks import ModelCheckpoint


def default_load_observables_fn(
    experiment: "ExperimentBase",
) -> Tuple:
    args = experiment.args

    metrics = {
        "accuracy": accuracy,
    }

    metric_float_obs = CalculateAndLogMetrics(
        metrics=metrics,
        keep_preds_for_epoch=False,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
    )

    metric_binary_obs = None

    observables = [
        # obs.RandomObservable(freq=args['freq_scalars']),
        obs.SaveLoss(freq=1),
        # obs.CountInputs(freq=args['freq_scalars']),
        metric_float_obs,
        obs.PlotPredsDefault(freq_batch={"train": np.infty, "val": np.infty, "test": np.infty}),
        # "InputAsPredMetric": obs.InputAsPredMetric(metrics, freq=args['freq_scalars']),
        # obs.ActivationHistogramBimonn(freq={'train': args['freq_hist'], 'val': 10000 // args['batch_size']}),
        # obs.PlotPreds(
        #     freq={'train': args['freq_imgs'], 'val': 10000 // args['batch_size']},
        #     fig_kwargs={"vmax": 1, "vmin": -1 if args['atomic_element'] == 'sybisel' else 0}
        # ),
        # "PlotParametersBiSE": obs.PlotParametersBiSE(freq=args['freq_scalars']),
        # "PlotLUIParametersBiSEL": obs.PlotLUIParametersBiSEL(freq=args['freq_scalars']),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # obs.PlotParametersBiseEllipse(freq=args['freq_scalars']),
        # obs.ActivationPHistogramBimonn(freq={'train': args['freq_hist'], 'val': None}),
        # "PlotWeightsBiSE": plot_weights_fn(freq=args['freq_imgs']),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        # "PlotGradientBise": plot_grad_obs,
        obs.ConvergenceMetrics(metrics, freq=args["freq_scalars"]),
        # obs.UpdateBinary(freq_batch=args["freq_update_binary_batch"], freq_epoch=args["freq_update_binary_epoch"]),
        # "PlotBimonn": obs.PlotBimonn(freq=args['freq_imgs'], figsize=(10, 5)),
        # "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=400),
        # "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": False}, dpi=600),
        # obs.ActivationHistogramBinaryBimonn(freq={'train': args['freq_hist'], 'val': 10000 // args['batch_size']}),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        # "ShowSelemBinary": obs.ShowSelemBinary(freq=args['freq_imgs']),
        # "ShowClosestSelemBinary": obs.ShowClosestSelemBinary(freq=args['freq_imgs']),
        # "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        # metric_binary_obs,
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        # "ConvergenceBinary": obs.ConvergenceBinary(freq=args['freq_imgs']),
        # obs.EpochValEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        # "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss_batch'], mode="min"),
        # "BatchEarlyStoppingBinaryDice": obs.BatchEarlyStopping(name="binary_dice", monitor="binary_mode/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
        # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
        obs.EpochReduceLrOnPlateau(patience=args["patience_reduce_lr"], on_train=True),
        obs.CheckLearningRate(freq=2 * args["freq_scalars"]),
    ]

    if "early_stopping" in experiment.args:
        observables += experiment.args["early_stopping"]
    else:
        observables += [
            obs.EpochValEarlyStopping(
                name="loss", monitor="loss/train/loss", patience=args["patience_loss"], mode="min"
            ),
        ]

    model_checkpoint_obs = ModelCheckpoint(
        monitor="metrics_epoch_mean/per_batch_step/loss_val",
        dirpath=join(experiment.log_dir, "best_weights"),
        save_weights_only=False,
        save_last=True,
    )
    callbacks = [model_checkpoint_obs]

    return observables, callbacks, metric_float_obs, metric_binary_obs, model_checkpoint_obs


def default_binary_load_observables_fn(
    experiment: "ExperimentBase",
) -> Tuple:
    args = experiment.args

    metrics = {
        "accuracy": accuracy,
    }

    metric_float_obs = CalculateAndLogMetrics(
        metrics=metrics,
        keep_preds_for_epoch=False,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
    )

    metric_binary_obs = obs.BinaryModeMetricMorpho(
        metrics=metrics,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
        plot_freq={"train": None, "val": None, "test": None},
    )

    observables = [
        # obs.RandomObservable(freq=args['freq_scalars']),
        obs.SaveLoss(freq=1),
        # obs.CountInputs(freq=args['freq_scalars']),
        metric_float_obs,
        # "InputAsPredMetric": obs.InputAsPredMetric(metrics, freq=args['freq_scalars']),
        # obs.ActivationHistogramBimonn(freq={'train': args['freq_hist'], 'val': 10000 // args['batch_size']}),
        # obs.PlotPreds(
        #     freq={'train': args['freq_imgs'], 'val': 10000 // args['batch_size']},
        #     fig_kwargs={"vmax": 1, "vmin": -1 if args['atomic_element'] == 'sybisel' else 0}
        # ),
        # "PlotParametersBiSE": obs.PlotParametersBiSE(freq=args['freq_scalars']),
        # "PlotLUIParametersBiSEL": obs.PlotLUIParametersBiSEL(freq=args['freq_scalars']),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # obs.PlotParametersBiseEllipse(freq=args['freq_scalars']),
        # obs.ActivationPHistogramBimonn(freq={'train': args['freq_hist'], 'val': None}),
        # "PlotWeightsBiSE": plot_weights_fn(freq=args['freq_imgs']),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        # "PlotGradientBise": plot_grad_obs,
        obs.ConvergenceMetrics(metrics, freq=args["freq_scalars"]),
        obs.UpdateBinary(freq_batch=args["freq_update_binary_batch"], freq_epoch=args["freq_update_binary_epoch"]),
        # "PlotBimonn": obs.PlotBimonn(freq=args['freq_imgs'], figsize=(10, 5)),
        # "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=400),
        # "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": False}, dpi=600),
        # obs.ActivationHistogramBinaryBimonn(freq={'train': args['freq_hist'], 'val': 10000 // args['batch_size']}),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        # "ShowSelemBinary": obs.ShowSelemBinary(freq=args['freq_imgs']),
        # "ShowClosestSelemBinary": obs.ShowClosestSelemBinary(freq=args['freq_imgs']),
        # "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        metric_binary_obs,
        obs.ActivatednessObservable(freq=args["freq_update_binary_epoch"]),
        obs.ClosestDistObservable(freq=args["freq_update_binary_epoch"]),
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        # "ConvergenceBinary": obs.ConvergenceBinary(freq=args['freq_imgs']),
        # obs.EpochValEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        # "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss_batch'], mode="min"),
        # "BatchEarlyStoppingBinaryDice": obs.BatchEarlyStopping(name="binary_dice", monitor="binary_mode/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
        # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
        obs.EpochReduceLrOnPlateau(patience=args["patience_reduce_lr"], on_train=True),
        obs.CheckLearningRate(freq=2 * args["freq_scalars"]),
    ]

    if "early_stopping" in experiment.args:
        observables += experiment.args["early_stopping"]
    else:
        observables += [
            obs.EpochValEarlyStopping(
                name="loss", monitor="loss/train/loss", patience=args["patience_loss"], mode="min"
            ),
        ]

    model_checkpoint_obs = ModelCheckpoint(
        monitor="metrics_epoch_mean/per_batch_step/loss_val",
        dirpath=join(experiment.log_dir, "best_weights"),
        save_weights_only=False,
        save_last=True,
    )
    callbacks = [model_checkpoint_obs]

    return observables, callbacks, metric_float_obs, metric_binary_obs, model_checkpoint_obs


def load_observables_bimonn_morpho_binary(experiment):
    args = experiment.args
    metrics = {
        "dice": lambda y_true, y_pred: dice(
            y_true, y_pred, threshold=0 if args["atomic_element"] == "sybisel" else 0.5
        ).mean(),
    }

    metric_float_obs = CalculateAndLogMetrics(
        metrics=metrics,
        keep_preds_for_epoch=False,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
    )

    metric_binary_obs = obs.BinaryModeMetricMorpho(
        metrics=metrics,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
        plot_freq={
            "train": args["freq_imgs"],
            "val": args["n_inputs.val"] // args["batch_size"],
            "test": args["freq_imgs"],
        },
    )

    observables = [
        obs.RandomObservable(freq=args["freq_scalars"]),
        obs.SaveLoss(freq=1),
        obs.CountInputs(freq=args["freq_scalars"]),
        metric_float_obs,
        obs.InputAsPredMetric(metrics, freq=args["freq_scalars"]),
        obs.ActivationHistogramBimonn(
            freq={"train": args["freq_hist"], "val": args["n_inputs.val"] // args["batch_size"]}
        ),
        obs.PlotPreds(
            freq={"train": args["freq_imgs"], "val": args["n_inputs.val"] // args["batch_size"]},
            fig_kwargs={"vmax": 1, "vmin": -1 if args["atomic_element"] == "sybisel" else 0},
        ),
        obs.PlotParametersBiSE(freq=args["freq_scalars"]),
        obs.PlotLUIParametersBiSEL(freq=args["freq_scalars"]),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        obs.PlotParametersBiseEllipse(freq=args["freq_scalars"]),
        obs.ActivationPHistogramBimonn(freq={"train": args["freq_hist"], "val": None}),
        obs.PlotWeightsBiSE(freq=args["freq_imgs"]),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        obs.PlotGradientBise(freq=args["freq_imgs"]),
        obs.ConvergenceMetrics(metrics, freq=args["freq_scalars"]),
        obs.DistToMorpOperation(
            freq={"batch": 20, "epoch": None},
            target_morp_operation=args["morp_operation"],
        ),
        obs.UpdateBinary(freq_batch=args["freq_update_binary_batch"], freq_epoch=args["freq_update_binary_epoch"]),
        obs.ActivatednessObservable(
            freq={"epoch": args["freq_update_binary_epoch"], "batch": args["freq_update_binary_batch"]}
        ),
        obs.ClosestDistObservable(
            freq={"epoch": args["freq_update_binary_epoch"], "batch": args["freq_update_binary_batch"]}
        ),
        obs.PlotBimonn(freq=args["freq_imgs"], figsize=(10, 5)),
        # "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=400),
        # "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": False}, dpi=600),
        obs.ActivationHistogramBinaryBimonn(
            freq={"train": args["freq_hist"], "val": args["n_inputs.val"] // args["batch_size"]}
        ),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        obs.ShowSelemBinary(freq=args["freq_imgs"]),
        obs.ShowClosestSelemBinary(freq=args["freq_imgs"]),
        # "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        metric_binary_obs,
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        # "ConvergenceBinary": obs.ConvergenceBinary(freq=args['freq_imgs']),
        obs.EpochReduceLrOnPlateau(patience=args["patience_reduce_lr"], on_train=True),
        obs.CheckLearningRate(freq=2 * args["freq_scalars"]),
        # obs.EpochValEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        # "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss_batch'], mode="min"),
        # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
        # obs.BatchEarlyStopping(name="binary_dice", monitor="binary_mode/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
    ]

    if "early_stopping" in experiment.args:
        observables += experiment.args["early_stopping"]
    else:
        observables += [
            obs.EpochValEarlyStopping(
                name="loss", monitor="loss/train/loss", patience=args["patience_loss"], mode="min"
            ),
            obs.CombineEarlyStopping(
                name="dice_and_ativated",
                early_stoppers=[
                    obs.BatchActivatedEarlyStopping(patience=0),
                    obs.BatchEarlyStopping(
                        name="binary_dice",
                        monitor="binary_mode/dice_train",
                        stopping_threshold=1,
                        patience=np.infty,
                        mode="max",
                    ),
                ],
                decision_rule="and",
            ),
        ]

    model_checkpoint_obs = ModelCheckpoint(
        monitor="metrics_epoch_mean/per_batch_step/loss_val",
        dirpath=join(experiment.log_dir, "best_weights"),
        save_weights_only=False,
        save_last=True,
    )
    callbacks = [model_checkpoint_obs]

    return observables, callbacks, metric_float_obs, metric_binary_obs, model_checkpoint_obs


def load_observables_morpho_binary(experiment):
    args = experiment.args
    metrics = {
        "dice": lambda y_true, y_pred: dice(
            y_true, y_pred, threshold=0 if args["atomic_element"] == "sybisel" else 0.5
        ).mean(),
    }

    metric_float_obs = CalculateAndLogMetrics(
        metrics=metrics,
        keep_preds_for_epoch=False,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
    )

    metric_binary_obs = obs.BinaryModeMetricMorpho(
        metrics=metrics,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
        plot_freq={
            "train": args["freq_imgs"],
            "val": args["n_inputs.val"] // args["batch_size"],
            "test": args["freq_imgs"],
        },
    )

    observables = [
        obs.RandomObservable(freq=args["freq_scalars"]),
        obs.SaveLoss(freq=1),
        obs.CountInputs(freq=args["freq_scalars"]),
        metric_float_obs,
        obs.InputAsPredMetric(metrics, freq=args["freq_scalars"]),
        # obs.ActivationHistogramBimonn(freq={'train': args['freq_hist'], 'val': args["n_inputs.val"] // args['batch_size']}),
        obs.PlotPreds(
            freq={"train": args["freq_imgs"], "val": args["n_inputs.val"] // args["batch_size"]},
            fig_kwargs={
                "vmax": 1,
                "vmin": -1 if (args["atomic_element"] == "sybisel") or (args["do_symetric_output"]) else 0,
            },
        ),
        # obs.PlotParametersBiSE(freq=args['freq_scalars']),
        # obs.PlotLUIParametersBiSEL(freq=args['freq_scalars']),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # obs.PlotParametersBiseEllipse(freq=args['freq_scalars']),
        # obs.ActivationPHistogramBimonn(freq={'train': args['freq_hist'], 'val': None}),
        # obs.PlotWeightsBiSE(freq=args['freq_imgs']),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        # obs.PlotGradientBise(freq=args['freq_imgs']),
        obs.ConvergenceMetrics(metrics, freq=args["freq_scalars"]),
        obs.UpdateBinary(freq_batch=args["freq_update_binary_batch"], freq_epoch=args["freq_update_binary_epoch"]),
        # obs.ActivatednessObservable(freq=args["freq_update_binary_epoch"]),
        # obs.ClosestDistObservable(freq=args["freq_update_binary_epoch"]),
        # obs.PlotBimonn(freq=args['freq_imgs'], figsize=(10, 5)),
        # "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=400),
        # "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": False}, dpi=600),
        # obs.ActivationHistogramBinaryBimonn(freq={'train': args['freq_hist'], 'val': args["n_inputs.val"] // args['batch_size']}),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        # obs.ShowSelemBinary(freq=args['freq_imgs']),
        # obs.ShowClosestSelemBinary(freq=args['freq_imgs']),
        # "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        metric_binary_obs,
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        # "ConvergenceBinary": obs.ConvergenceBinary(freq=args['freq_imgs']),
        obs.EpochReduceLrOnPlateau(patience=args["patience_reduce_lr"], on_train=True),
        obs.CheckLearningRate(freq=2 * args["freq_scalars"]),
        # obs.EpochValEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        # "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss_batch'], mode="min"),
        # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
        # obs.BatchEarlyStopping(name="binary_dice", monitor="binary_mode/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
    ]

    if "early_stopping" in experiment.args:
        observables += experiment.args["early_stopping"]
    else:
        observables += [
            obs.EpochValEarlyStopping(
                name="loss", monitor="loss/train/loss", patience=args["patience_loss"], mode="min"
            ),
        ]

    model_checkpoint_obs = ModelCheckpoint(
        monitor="metrics_epoch_mean/per_batch_step/loss_val",
        dirpath=join(experiment.log_dir, "best_weights"),
        save_weights_only=False,
        save_last=True,
    )
    callbacks = [model_checkpoint_obs]

    return observables, callbacks, metric_float_obs, metric_binary_obs, model_checkpoint_obs


def load_observables_ste_morpho_binary(experiment):
    args = experiment.args
    metrics = {
        "dice": lambda y_true, y_pred: dice(
            y_true, y_pred, threshold=0 if args["atomic_element"] == "sybisel" else 0.5
        ).mean(),
    }

    metric_float_obs = CalculateAndLogMetrics(
        metrics=metrics,
        keep_preds_for_epoch=False,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
    )

    metric_binary_obs = obs.BinaryModeMetricMorpho(
        metrics=metrics,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
        plot_freq={
            "train": args["freq_imgs"],
            "val": args["n_inputs.val"] // args["batch_size"],
            "test": args["freq_imgs"],
        },
    )

    observables = [
        obs.RandomObservable(freq=args["freq_scalars"]),
        obs.SaveLoss(freq=1),
        obs.CountInputs(freq=args["freq_scalars"]),
        metric_float_obs,
        obs.InputAsPredMetric(metrics, freq=args["freq_scalars"]),
        # obs.ActivationHistogramBimonn(freq={'train': args['freq_hist'], 'val': args["n_inputs.val"] // args['batch_size']}),
        obs.PlotPreds(
            freq={"train": args["freq_imgs"], "val": args["n_inputs.val"] // args["batch_size"]},
            fig_kwargs={
                "vmax": 1,
                "vmin": -1 if (args["atomic_element"] == "sybisel") or (args["do_symetric_output"]) else 0,
            },
        ),
        obs.PlotSteWeights(freq=args["freq_imgs"]),
        obs.PlotSTE(freq=args["freq_imgs"]),
        # obs.PlotParametersBiSE(freq=args['freq_scalars']),
        # obs.PlotLUIParametersBiSEL(freq=args['freq_scalars']),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # obs.PlotParametersBiseEllipse(freq=args['freq_scalars']),
        # obs.ActivationPHistogramBimonn(freq={'train': args['freq_hist'], 'val': None}),
        # obs.PlotWeightsBiSE(freq=args['freq_imgs']),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        # obs.PlotGradientBise(freq=args['freq_imgs']),
        obs.ConvergenceMetrics(metrics, freq=args["freq_scalars"]),
        # obs.UpdateBinary(freq_batch=args["freq_update_binary_batch"], freq_epoch=args["freq_update_binary_epoch"]),
        # obs.ActivatednessObservable(freq=args["freq_update_binary_epoch"]),
        # obs.ClosestDistObservable(freq=args["freq_update_binary_epoch"]),
        # obs.PlotBimonn(freq=args['freq_imgs'], figsize=(10, 5)),
        # "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=400),
        # "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": False}, dpi=600),
        # obs.ActivationHistogramBinaryBimonn(freq={'train': args['freq_hist'], 'val': args["n_inputs.val"] // args['batch_size']}),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        # obs.ShowSelemBinary(freq=args['freq_imgs']),
        # obs.ShowClosestSelemBinary(freq=args['freq_imgs']),
        # "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        # metric_binary_obs,
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        # "ConvergenceBinary": obs.ConvergenceBinary(freq=args['freq_imgs']),
        obs.EpochReduceLrOnPlateau(patience=args["patience_reduce_lr"], on_train=True),
        obs.CheckLearningRate(freq=2 * args["freq_scalars"]),
        # obs.EpochValEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        # "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss_batch'], mode="min"),
        # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
        # obs.BatchEarlyStopping(name="binary_dice", monitor="binary_mode/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
    ]

    if "early_stopping" in experiment.args:
        observables += experiment.args["early_stopping"]
    else:
        observables += [
            obs.EpochValEarlyStopping(
                name="loss", monitor="loss/train/loss", patience=args["patience_loss"], mode="min"
            ),
        ]

    model_checkpoint_obs = ModelCheckpoint(
        monitor="metrics_epoch_mean/per_batch_step/loss_val",
        dirpath=join(experiment.log_dir, "best_weights"),
        save_weights_only=False,
        save_last=True,
    )
    callbacks = [model_checkpoint_obs]

    return observables, callbacks, metric_float_obs, metric_binary_obs, model_checkpoint_obs


def load_observables_bimonn_morpho_grayscale(experiment):
    args = experiment.args
    metrics = {
        "dice": lambda y_true, y_pred: dice(
            y_true, y_pred, threshold=0 if args["atomic_element"] == "sybisel" else 0.5
        ).mean(),
    }
    metrics_gray_scale = {"mse": lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean()}

    metric_float_obs = CalculateAndLogMetrics(
        metrics=metrics,
        keep_preds_for_epoch=False,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
    )

    metric_binary_obs = obs.BinaryModeMetricMorpho(
        metrics=metrics,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
        plot_freq={
            "train": args["freq_imgs"],
            "val": args["n_inputs.val"] // args["batch_size"],
            "test": args["freq_imgs"],
        },
    )

    observables = [
        obs.RandomObservable(freq=args["freq_scalars"]),
        obs.SaveLoss(freq=1),
        obs.CountInputs(freq=args["freq_scalars"]),
        metric_float_obs,
        obs.InputAsPredMetricGrayScale(metrics_gray_scale, freq=args["freq_scalars"]),
        obs.BinaryModeMetricGrayScale(metrics_gray_scale, freq=args["freq_imgs"]),
        # "InputAsPredMetric": obs.InputAsPredMetric(metrics, freq=args['freq_scalars']),
        obs.ActivationHistogramBimonn(
            freq={"train": args["freq_hist"], "val": args["n_inputs.val"] // args["batch_size"]}
        ),
        obs.PlotPredsGrayscale(
            freq={"train": args["freq_imgs"], "val": args["n_inputs.val"] // args["batch_size"]},
        ),
        obs.PlotParametersBiSE(freq=args["freq_scalars"]),
        obs.PlotLUIParametersBiSEL(freq=args["freq_scalars"]),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        obs.PlotParametersBiseEllipse(freq=args["freq_scalars"]),
        obs.ActivationPHistogramBimonn(freq={"train": args["freq_hist"], "val": None}),
        obs.PlotWeightsBiSE(freq=args["freq_imgs"]),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        obs.PlotGradientBise(freq=args["freq_imgs"]),
        obs.ConvergenceMetrics(metrics, freq=args["freq_scalars"]),
        obs.UpdateBinary(freq_batch=args["freq_update_binary_batch"], freq_epoch=args["freq_update_binary_epoch"]),
        obs.ActivatednessObservable(freq=args["freq_update_binary_epoch"]),
        obs.ClosestDistObservable(freq=args["freq_update_binary_epoch"]),
        obs.PlotBimonn(freq=args["freq_imgs"], figsize=(10, 5)),
        # "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=400),
        # "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": False}, dpi=600),
        obs.ActivationHistogramBinaryBimonn(
            freq={"train": args["freq_hist"], "val": args["n_inputs.val"] // args["batch_size"]}
        ),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        obs.ShowSelemBinary(freq=args["freq_imgs"]),
        obs.ShowClosestSelemBinary(freq=args["freq_imgs"]),
        # "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        metric_binary_obs,
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        # "ConvergenceBinary": obs.ConvergenceBinary(freq=args['freq_imgs']),
        obs.EpochReduceLrOnPlateau(patience=args["patience_reduce_lr"], on_train=True),
        obs.CheckLearningRate(freq=2 * args["freq_scalars"]),
        # obs.EpochValEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        # # "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss_batch'], mode="min"),
        # # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
        # obs.BatchEarlyStopping(name="binary_dice", monitor="binary_mode/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
    ]

    if "early_stopping" in experiment.args:
        observables += experiment.args["early_stopping"]
    else:
        observables += [
            obs.EpochValEarlyStopping(
                name="loss", monitor="loss/train/loss", patience=args["patience_loss"], mode="min"
            ),
            # "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss_batch'], mode="min"),
            # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
            obs.BatchEarlyStopping(
                name="binary_dice",
                monitor="binary_mode/dice_train",
                stopping_threshold=1,
                patience=np.infty,
                mode="max",
            ),
        ]

    model_checkpoint_obs = ModelCheckpoint(
        monitor="metrics_epoch_mean/per_batch_step/loss_val",
        dirpath=join(experiment.log_dir, "best_weights"),
        save_weights_only=False,
        save_last=True,
    )
    callbacks = [model_checkpoint_obs]

    return observables, callbacks, metric_float_obs, metric_binary_obs, model_checkpoint_obs


def load_observables_classification_bimonn(experiment):
    args = experiment.args
    metrics = {"accuracy": accuracy}

    metric_float_obs = CalculateAndLogMetrics(
        metrics=metrics,
        keep_preds_for_epoch=False,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
    )

    metric_binary_obs = obs.BinaryModeMetricClassif(
        metrics=metrics,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
        plot_freq={
            "train": args["freq_imgs"],
            "val": args["n_inputs.val"] // args["batch_size"],
            "test": args["freq_imgs"],
        },
    )

    observables = [
        obs.RandomObservable(freq=args["freq_scalars"]),
        obs.SaveLoss(freq=1),
        obs.CountInputs(freq=args["freq_scalars"]),
        metric_float_obs,
        # obs.InputAsPredMetricGrayScale(metrics_gray_scale, freq=args['freq_scalars']),
        # obs.BinaryModeMetricGrayScale(metrics_gray_scale, freq=args['freq_imgs']),
        # "InputAsPredMetric": obs.InputAsPredMetric(metrics, freq=args['freq_scalars']),
        obs.ActivationHistogramBimonn(
            freq={"train": args["freq_hist"], "val": args["n_inputs.val"] // args["batch_size"]}
        ),
        obs.PlotPredsClassif(
            freq={"train": args["freq_imgs"], "val": args["n_inputs.val"] // args["batch_size"]},
        ),
        # obs.PlotParametersBiSE(freq=args['freq_scalars']),
        # obs.PlotLUIParametersBiSEL(freq=args['freq_scalars']),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # obs.PlotParametersBiseEllipse(freq=args['freq_scalars']),
        obs.ActivationPHistogramBimonn(freq={"train": args["freq_hist"], "val": None}),
        # obs.PlotWeightsBiSE(freq=args['freq_imgs']),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        # obs.PlotGradientBise(freq=args['freq_imgs']),
        obs.ConvergenceMetrics(metrics, freq=args["freq_scalars"]),
        obs.UpdateBinary(freq_batch=args["freq_update_binary_batch"], freq_epoch=args["freq_update_binary_epoch"]),
        obs.ActivatednessObservable(freq=args["freq_update_binary_epoch"]),
        obs.ClosestDistObservable(freq=args["freq_update_binary_epoch"]),
        # obs.PlotBimonn(freq=args['freq_imgs'], figsize=(10, 5)),
        # "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=400),
        # "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": False}, dpi=600),
        obs.ActivationHistogramBinaryBimonn(
            freq={"train": args["freq_hist"], "val": args["n_inputs.val"] // args["batch_size"]}
        ),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        # obs.ShowSelemBinary(freq=args['freq_imgs']),
        # obs.ShowClosestSelemBinary(freq=args['freq_imgs']),
        # "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        metric_binary_obs,
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        # "ConvergenceBinary": obs.ConvergenceBinary(freq=args['freq_imgs']),
        # obs.EpochValEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        # "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss_batch'], mode="min"),
        # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
        # obs.BatchEarlyStopping(name="binary_dice", monitor="binary_mode/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
        obs.EpochReduceLrOnPlateau(patience=args["patience_reduce_lr"], on_train=True),
        obs.CheckLearningRate(freq=2 * args["freq_scalars"]),
    ]

    if "early_stopping" in experiment.args:
        observables += experiment.args["early_stopping"]
    else:
        observables += [
            obs.EpochValEarlyStopping(
                name="loss", monitor="loss/train/loss", patience=args["patience_loss"], mode="min"
            ),
        ]

    model_checkpoint_obs = ModelCheckpoint(
        monitor="metrics_epoch_mean/per_batch_step/loss_val",
        dirpath=join(experiment.log_dir, "best_weights"),
        save_weights_only=False,
        save_last=True,
    )
    callbacks = [model_checkpoint_obs]

    return observables, callbacks, metric_float_obs, metric_binary_obs, model_checkpoint_obs


def load_observables_classification_channel_bimonn(experiment):
    args = experiment.args
    metrics = {"accuracy": accuracy}

    metric_float_obs = CalculateAndLogMetrics(
        metrics=metrics,
        keep_preds_for_epoch=False,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
    )

    metric_binary_obs = obs.BinaryModeMetricClassifChannel(
        metrics=metrics,
        freq={"train": args["freq_scalars"], "val": 1, "test": 1},
        plot_freq={
            "train": args["freq_imgs"],
            "val": args["n_inputs.val"] // args["batch_size"],
            "test": args["freq_imgs"],
        },
        dataset=experiment.trainloader.dataset,
    )

    observables = [
        obs.RandomObservable(freq=args["freq_scalars"]),
        obs.SaveLoss(freq=1),
        obs.CountInputs(freq=args["freq_scalars"]),
        metric_float_obs,
        # obs.InputAsPredMetricGrayScale(metrics_gray_scale, freq=args['freq_scalars']),
        # obs.BinaryModeMetricGrayScale(metrics_gray_scale, freq=args['freq_imgs']),
        # "InputAsPredMetric": obs.InputAsPredMetric(metrics, freq=args['freq_scalars']),
        obs.ActivationHistogramBimonn(
            freq={"train": args["freq_hist"], "val": args["n_inputs.val"] // args["batch_size"]}
        ),
        obs.PlotPredsClassifChannel(
            freq={"train": args["freq_imgs"], "val": args["n_inputs.val"] // args["batch_size"]},
            dataset=experiment.trainloader.dataset,
        ),
        # obs.PlotParametersBiSE(freq=args['freq_scalars']),
        # obs.PlotLUIParametersBiSEL(freq=args['freq_scalars']),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # obs.PlotParametersBiseEllipse(freq=args['freq_scalars']),
        obs.ActivationPHistogramBimonn(freq={"train": args["freq_hist"], "val": None}),
        # obs.PlotWeightsBiSE(freq=args['freq_imgs']),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        # obs.PlotGradientBise(freq=args['freq_imgs']),
        obs.ConvergenceMetrics(metrics, freq=args["freq_scalars"]),
        obs.UpdateBinary(freq_batch=args["freq_update_binary_batch"], freq_epoch=args["freq_update_binary_epoch"]),
        obs.ActivatednessObservable(
            freq={"epoch": args["freq_update_binary_epoch"], "batch": args["freq_update_binary_batch"]}
        ),
        obs.ClosestDistObservable(
            freq={"epoch": args["freq_update_binary_epoch"], "batch": args["freq_update_binary_batch"]}
        ),
        # obs.PlotBimonn(freq=args['freq_imgs'], figsize=(10, 5)),
        # "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=400),
        # "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": False}, dpi=600),
        obs.ActivationHistogramBinaryBimonn(
            freq={"train": args["freq_hist"], "val": args["n_inputs.val"] // args["batch_size"]}
        ),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        # obs.ShowSelemBinary(freq=args['freq_imgs']),
        # obs.ShowClosestSelemBinary(freq=args['freq_imgs']),
        # "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        metric_binary_obs,
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        # "ConvergenceBinary": obs.ConvergenceBinary(freq=args['freq_imgs']),
        # obs.EpochValEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        # "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss_batch'], mode="min"),
        # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
        # obs.BatchEarlyStopping(name="binary_dice", monitor="binary_mode/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
        obs.EpochReduceLrOnPlateau(patience=args["patience_reduce_lr"], on_train=True),
        obs.CheckLearningRate(freq=2 * args["freq_scalars"]),
    ]

    if "early_stopping" in experiment.args:
        observables += experiment.args["early_stopping"]
    else:
        observables += [
            obs.EpochValEarlyStopping(
                name="loss", monitor="loss/train/loss", patience=args["patience_loss"], mode="min"
            ),
        ]

    model_checkpoint_obs = ModelCheckpoint(
        monitor="metrics_epoch_mean/per_batch_step/loss_val",
        dirpath=join(experiment.log_dir, "best_weights"),
        save_weights_only=False,
        save_last=True,
    )
    callbacks = [model_checkpoint_obs]

    return observables, callbacks, metric_float_obs, metric_binary_obs, model_checkpoint_obs
