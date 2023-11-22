from time import time
import os
from os.path import join
import pathlib

import pandas as pd
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from deep_morpho.datasets.mnist_dataset import MnistMorphoDataset
from deep_morpho.utils import set_seed


# from deep_morpho.datasets.generate_forms2 import get_random_diskorect
# from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect
from deep_morpho.datasets.diskorect_dataset import DiskorectDataset, MultiRectDataset
from deep_morpho.datasets.axspa_roi_dataset import AxspaROIDataset, AxspaROISimpleDataset
import deep_morpho.observables as obs
import deep_morpho.biblio_comparison.observables as biblio_obs
from deep_morpho.datasets.sticks_noised_dataset import NoistiDataset
from general.nn.observables import CalculateAndLogMetrics
from general.utils import format_time, log_console, create_logger, save_yaml, close_handlers
from general.nn.utils import train_val_test_split
from deep_morpho.metrics import masked_dice
from deep_morpho.biblio_comparison.args import final_args as all_args
from general.code_saver import CodeSaver
from deep_morpho.biblio_comparison.lightning_models import LightningLMorph, LightningSMorph, LightningAdaptativeMorphologicalLayer

model_dict = {
    "adaptative": LightningAdaptativeMorphologicalLayer,
    "smorph": LightningSMorph,
    "lmorph": LightningLMorph,
}


def get_dataloader(args):

    if args['dataset_type'] == 'diskorect':
        # if (args['dataset_path'] is not None) and (args['dataset_path'] != 'generate'):
        #     dataloader = MultiRectDataset.get_loader(
        #         batch_size=args['batch_size'], dataset_path=args['dataset_path'], do_load_in_ram=args['in_ram'],
        #         morp_operation=args['morp_operation'], logger=console_logger, n_inputs=args['n_inputs'],
        #         num_workers=args['num_workers']
        #     )
        # else:
        trainloader = DiskorectDataset.get_loader(
            batch_size=args['batch_size'],
            n_inputs=args['n_inputs'],
            max_generation_nb=args['nb_batch_indep'],
            random_gen_fn=args['random_gen_fn'],
            random_gen_args=args['random_gen_args'],
            morp_operation=args['morp_operation'],
            seed=args['seed'],
            device=device,
            num_workers=args['num_workers'],
            do_symetric_output=False,
            # persistent_workers=True,
            # pin_memory=True,
        )
        valloader = None
        testloader = None

    elif args['dataset_type'] == 'axspa_roi':
        data = pd.read_csv(args['dataset_path'])
        prop_train, prop_val, prop_test = args['train_test_split']
        max_res = data['resolution'].value_counts(sort=True, ascending=False).index[0]
        data = data[data['resolution'] == max_res]
        trainloader, valloader, testloader = AxspaROISimpleDataset.get_train_val_test_loader(
            *train_val_test_split(
                data,
                train_size=int(prop_train * len(data)),
                val_size=int(prop_val * len(data)),
                test_size=int(prop_test * len(data))
            ),
            batch_size=args['batch_size'],
            preprocessing=args['preprocessing'],
            shuffle=True,
            do_symetric_output=False,
        )

    elif args['dataset_type'] in ["mnist", "inverted_mnist"]:
        prop_train, prop_val, prop_test = args['train_test_split']
        trainloader, valloader, testloader = MnistMorphoDataset.get_train_val_test_loader(
            n_inputs_train=int(prop_train * args['n_inputs']),
            n_inputs_val=int(prop_val * args['n_inputs']),
            n_inputs_test=int(prop_test * args['n_inputs']),
            batch_size=args['batch_size'],
            morp_operation=args['morp_operation'],
            preprocessing=args['preprocessing'],
            # shuffle=True,
            num_workers=args['num_workers'],
            do_symetric_output=False,
            **args['mnist_args']
        )

    elif args['dataset_type'] == "sticks_noised":
        trainloader = NoistiDataset.get_loader(
            batch_size=args['batch_size'],
            n_inputs=args['n_inputs'],
            max_generation_nb=args['nb_batch_indep'],
            seed=args['seed'],
            num_workers=args['num_workers'],
            do_symetric_output=False,
            **args['sticks_noised_args']
        )
        valloader = None
        testloader = None

    return trainloader, valloader, testloader

    return trainloader, valloader, testloader



def main(args, logger):
    args['seed'] = set_seed(args['batch_seed'])
    with open(join(logger.log_dir, "seed.txt"), "w") as f:
        f.write(f"{args['seed']}")

    trainloader, valloader, testloader = get_dataloader(args)
    metrics = {'dice': lambda y_true, y_pred: masked_dice(y_true, y_pred, border=(args['kernel_size'] // 2, args['kernel_size'] // 2), threshold=.5).mean()}

    observables_dict = {
        "RandomObservable": obs.RandomObservable(),
        "SaveLoss": obs.SaveLoss(),
        "CalculateAndLogMetric": CalculateAndLogMetrics(
            metrics=metrics,
            keep_preds_for_epoch=False,
        ),
        "PlotPreds": obs.PlotPreds(freq={'train': args['freq_imgs'], 'val': 2}),
        "PlotBiblioModel": biblio_obs.PlotBiblioModel(freq=args['freq_imgs'], figsize=(10, 5)),
        "InputAsPredMetric": obs.InputAsPredMetric(metrics),
        "CountInputs": obs.CountInputs(),
        "PlotParameters": biblio_obs.PlotParameters(freq=1),
        "PlotWeights": biblio_obs.PlotWeights(freq=args['freq_imgs']),
        # "PlotLUIParametersBiSEL": obs.PlotLUIParametersBiSEL(),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # "PlotGradientBise": obs.PlotGradientBise(freq=args['freq_imgs']),
        "ConvergenceMetrics": obs.ConvergenceMetrics(metrics),
        "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        # "BatchEarlyStoppingDice": obs.BatchEarlyStopping(name="dice", monitor="metrics_batch/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
        "BatchEarlyStoppingLossZero": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", stopping_threshold=1e-5, patience=np.infty, mode="min"),
        "BatchReduceLrOnPlateau": obs.BatchReduceLrOnPlateau(patience=args['patience_reduce_lr'], on_train=True),
        "CheckLearningRate": obs.CheckLearningRate(freq=2),
    }

    observables = list(observables_dict.values())

    model = model_dict[args['model']](
        model_args={
            "kernel_size": [(args['kernel_size'], args['kernel_size']) for _ in range(args['n_atoms'])],
        },
        learning_rate=args['learning_rate'],
        loss=args['loss'],
        optimizer=args['optimizer'],
        observables=observables,
    )

    model.to(device)

    logger.experiment.add_graph(model, torch.ones(1, 1, 50, 50).to(device))


    pathlib.Path(join(logger.log_dir, "target_SE")).mkdir(exist_ok=True, parents=True)
    figs_selems = args['morp_operation'].plot_selem_arrays()
    for (layer_idx, chan_input, chan_output), fig in figs_selems.items():
        fig.savefig(join(logger.log_dir, "target_SE", f"target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
        logger.experiment.add_figure(f"target_SE/target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig)
        plt.close(fig)

    # pathlib.Path(join(logger.log_dir, "target_UI")).mkdir(exist_ok=True, parents=True)
    # figs_ui = args['morp_operation'].plot_ui_arrays()
    # for (layer_idx, chan_output), fig in figs_ui.items():
    #     fig.savefig(join(logger.log_dir, "target_UI", f"target_UI_l_{layer_idx}_chin_chout_{chan_output}.png"))
    #     logger.experiment.add_figure(f"target_UI/target_UI_l_{layer_idx}_chin_chout_{chan_output}", fig)

    pathlib.Path(join(logger.log_dir, "morp_operations")).mkdir(exist_ok=True, parents=True)
    fig_morp_operation = args['morp_operation'].vizualise().fig
    fig_morp_operation.savefig(join(logger.log_dir, "morp_operations", "morp_operations.png"))
    logger.experiment.add_figure("target_operations/morp_operations", fig_morp_operation)
    plt.close(fig_morp_operation)

    trainer = Trainer(
        max_epochs=args['n_epochs'],
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        # progress_bar_refresh_rate=10,
        callbacks=observables.copy(),
        log_every_n_steps=10,
        deterministic=True,
        num_sanity_val_steps=1,
    )

    trainer.fit(model, trainloader, valloader)

    for observable in observables:
        observable.save(join(trainer.log_dir, 'observables'))



if __name__ == '__main__':
    start_all = time()

    code_saver = CodeSaver(
        src_path=os.getcwd(),
        temporary_path="deep_morpho/results",
        ignore_patterns=("*__pycache__*", "*results*", "data", "*.ipynb", '.git', 'ssm'),
    )

    code_saver.save_in_temporary_file()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    bugged = []
    results = []
    console_logger = None

    for args_idx, args in enumerate(all_args):

        name = join(args["experiment_name"], args['experiment_subname'])

        # name += f"_{args['atomic_element']}"

        logger = TensorBoardLogger("deep_morpho/results/results_tensorboards", name=name, default_hp_metric=False)
        code_saver.save_in_final_file(join(logger.log_dir, 'code'))
        save_yaml(args, join(logger.log_dir, 'args.yaml'))

        if console_logger is not None:
            close_handlers(console_logger)

        console_logger = create_logger(
            f'args_{args_idx}', all_logs_path=join(logger.log_dir, 'all_logs.log'), error_path=join(logger.log_dir, 'error_logs.log')
        )

        log_console('Device: {}'.format(device), logger=console_logger)
        log_console('==================', logger=console_logger)
        log_console('==================', logger=console_logger)
        log_console(f'Args number {args_idx + 1} / {len(all_args)}', logger=console_logger)
        log_console('Time since beginning: {} '.format(format_time(time() - start_all)), logger=console_logger)
        log_console(logger.log_dir, logger=console_logger)
        log_console(args['morp_operation'], logger.log_dir, logger=console_logger)

        results.append(main(args, logger))

        # try:
        #     main(args, logger)
        # except Exception:
        #     console_logger.exception(
        #         f'Args nb {args_idx + 1} / {len(all_args)} failed : ')
        #     bugged.append(args_idx+1)

        log_console("Done.", logger=console_logger)

    code_saver.delete_temporary_file()


    log_console(f'{len(bugged)} Args Bugged: ', bugged, logger=console_logger)
    log_console(f'{len(all_args)} args done in {format_time(time() - start_all)} ', logger=console_logger)
