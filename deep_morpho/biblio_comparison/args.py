import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from deep_morpho.datasets.generate_forms3 import get_random_diskorect_channels
from general.utils import dict_cross
from general.nn.loss import DiceLoss
from deep_morpho.datasets.sticks_noised_dataset import NoistiDataset
from deep_morpho.loss import (
    MaskedMSELoss, MaskedDiceLoss, MaskedBCELoss, QuadraticBoundRegularization, LinearBoundRegularization,
    MaskedBCENormalizedLoss, MaskedNormalizedDiceLoss, BCENormalizedLoss,
)
from .args_morp_ops_mnist import morp_operations as morp_operations_mnist
from .args_morp_ops_diskorect import morp_operations as morp_operations_diskorect
from .lightning_models import LightningLMorph, LightningSMorph, LightningAdaptativeMorphologicalLayer


loss_dict = {
    "MaskedMSELoss": MaskedMSELoss,
    "MaskedDiceLoss": MaskedDiceLoss,
    "DiceLoss": DiceLoss,
    "MaskedBCELoss": MaskedBCELoss,
    "MaskedBCENormalizedLoss": MaskedBCENormalizedLoss,
    "quadratic": QuadraticBoundRegularization,
    "linear": LinearBoundRegularization,
    "MaskedNormalizedDiceLoss": MaskedNormalizedDiceLoss,
    "MSELoss": nn.MSELoss,
    "BCENormalizedLoss": BCENormalizedLoss,
    "BCELoss": nn.BCELoss,
}

do_args = {
    ('smorph', 'diskorect'): True,
    ('smorph', 'mnist'): False,
    ('smorph', 'inverted_mnist'): False,
    ('smorph', 'sticks_noised'): False,

    ('lmorph', 'diskorect'): False,
    ('lmorph', 'mnist'): False,
    ('lmorph', 'inverted_mnist'): False,
    ('lmorph', 'sticks_noised'): False,
    ('adaptative', 'diskorect'): False,

    ('adaptative', 'mnist'): False,
    ('adaptative', 'inverted_mnist'): False,
    ('adaptative', 'sticks_noised'): False,
}


all_args = {}
all_args_dict = {}

all_args['batch_seed'] = [None]


all_args['n_try'] = [0]
# all_args['n_try'] = range(1, 11)

all_args['experiment_name'] = [
    "JMIV/biblio/sandbox"
    # "JMIV/biblio/reprod2"
]

#########################

morp_operations = []


# DATA ARGS
all_args['preprocessing'] = [  # for axspa roi
    None,
]
all_args['dataset_path'] = [
    # 'data/deep_morpho/dataset_0',
    'generate',
]
all_args['in_ram'] = [
    # False,
    True,
]
all_args['random_gen_fn'] = [
    # get_random_rotated_diskorect,
    get_random_diskorect_channels
]
all_args['random_gen_args'] = [
    {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, 'border': (0, 0)}
    # {'size': (50, 50), 'n_shapes': 30, 'max_shape': (15, 15), 'p_invert': 0.5, 'n_holes': 15, 'max_shape_holes': (7, 7)}

]

# all_args['n_inputs'] = [
#     1_000_000,
#     # 100_000,
#     # 70000,
# ]
all_args['train_test_split'] = [(1, 1, 0)]

all_args['patience_loss'] = [2100]
all_args['patience_reduce_lr'] = [700]

# TRAINING ARGS

all_args['loss_data_str'] = ['MSELoss']
all_args['num_workers'] = [
    # 20,
    0,
]
all_args['freq_imgs'] = [300]
all_args['n_epochs'] = [5]


# MODEL ARGS
# all_args['n_atoms'] = [
#     # 'adapt',
#     4,
# ]

all_args['n_steps'] = [10000]
all_args['nb_batch_indep'] = [0]

all_args['n_atoms'] = ['adapt']
all_args['kernel_size'] = ["adapt"]

##################
# LMorph, SMorph #
##################

all_args['model'] = [
    # "lmorph",
    "smorph",
]
all_args['optimizer'] = [optim.Adam]
all_args['learning_rate'] = [0.001]


# MNIST
all_args['batch_size'] = [32]
all_args['n_epochs'] = [1000]

all_args['patience_loss'] = [60000 // 32 * 10]
all_args['patience_reduce_lr'] = [60000 // 32 * 5]

all_args['mnist_args'] = [
    {"threshold": 30, "size": (28, 28), "invert_input_proba": 0}
    # {"threshold": 30, "size": (50, 50), "invert_input_proba": 0}
    # {"threshold": 30, "size": (50, 50), "invert_input_proba": 1}
]

all_args_dict['smorph', 'mnist'] = dict_cross(dict(**all_args, **{'dataset_type': ["mnist"], "morp_operation": morp_operations_mnist}))
# all_args_lsmorph += dict_cross(dict(**all_args, **{'dataset_type': ["mnist"], "morp_operation": morp_operations_mnist}))

# DISKORECT
all_args['batch_size'] = [256]
all_args['n_epochs'] = [1]

all_args['patience_loss'] = [2100]
all_args['patience_reduce_lr'] = [700]

all_args['kernel_size'] = [
    # "adapt",
    3,
]
all_args['n_atoms'] = [
    # 'adapt',
    10,
]

all_args_dict['smorph', 'diskorect'] = dict_cross(dict(**all_args, **{'dataset_type': ["diskorect"], "morp_operation": morp_operations_diskorect}))


# STICKS NOISED
all_args['sticks_noised_angles'] = [
    # [0, 90],
    # [30, 60]
    # [30],
    # [30, 120],
    # [0],
    [0, 90],
    # np.linspace(0, 160, 5),
    # np.linspace(0, 180, 5),
]
all_args['sticks_noised_args'] = [
    {
        "size": (50, 50),
        "n_shapes": 15,
        "lengths_lim": (12, 15),
        "widths_lim": (0, 0),
        "p_invert": 0,
        "border": (0, 0),
        "noise_proba": 0.1,
    }
]

all_args['n_atoms'] = [2]

all_args_dict['smorph', 'sticks_noised'] = dict_cross(dict(**all_args, **{'dataset_type': ["sticks_noised"]}))


##############
# Adaptative #
##############

all_args['model'] = ["adaptative"]
all_args['optimizer'] = [optim.Adam]
all_args['learning_rate'] = [1e-3]
all_args['batch_size'] = [64]

all_args['n_atoms'] = ['adapt']


all_args_dict['adaptative', 'mnist'] = dict_cross(dict(**all_args, **{'dataset_type': ["mnist"], "morp_operation": morp_operations_mnist}))
all_args_dict['adaptative', 'diskorect'] = dict_cross(dict(**all_args, **{'dataset_type': ["diskorect"], "morp_operation": morp_operations_diskorect}))

# all_args_adaptative = (
#     dict_cross(dict(**all_args, **{'dataset_type': ["mnist"], "morp_operation": morp_operations_mnist})) +
#     # dict_cross(dict(**all_args, **{'dataset_type': ["diskorect"], "morp_operation": morp_operations_diskorect})) +
#     []
# )

#############################

final_args = []

for model in ['adaptative', 'smorph', 'lmorph']:
    for dataset in ['diskorect', 'mnist', 'inverted_mnist', 'sticks_noised']:
        if do_args[model, dataset]:
            final_args += all_args_dict[model, dataset]

# all_args = (
#     all_args_lsmorph +
#     # all_args_adaptative +
#     []
# )

#
for idx, args in enumerate(final_args):


    # args['kernel_size'] = 'adapt'

    kwargs_loss = {}
    args['loss_data'] = loss_dict[args['loss_data_str']](**kwargs_loss)
    args['loss'] = {"loss_data": args['loss_data']}

    if args['dataset_type'] == "sticks_noised":
        args["sticks_noised_args"] = args["sticks_noised_args"].copy()
        args['sticks_noised_args']['angles'] = args['sticks_noised_angles']
        args['morp_operation'] = NoistiDataset.get_default_morp_operation(
            lengths_lim=args['sticks_noised_args']['lengths_lim'],
            angles=args['sticks_noised_args']['angles'],
        )
        args['sticks_noised_args']['size'] = args['sticks_noised_args']['size'] + (args["morp_operation"].in_channels[0],)

    if args["kernel_size"] == "adapt":
        args["kernel_size"] = args["morp_operation"].selems[0][0][0].shape[0]

    args['experiment_subname'] = f"{args['model']}/{args['dataset_type']}/{args['morp_operation'].name}"

    if args["n_atoms"] == 'adapt':
        args['n_atoms'] = len(args['morp_operation'])


    if args['dataset_type'] in ["diskorect", "sticks_noised"]:
        args['n_epochs'] = 1
        args['n_inputs'] = args['n_steps'] * args['batch_size']
        args["random_gen_args"] = args["random_gen_args"].copy()
        args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)
        args['random_gen_args']['size'] = args['random_gen_args']['size'] + (args["morp_operation"].in_channels[0],)


    if args['dataset_type'] == "mnist":
        args['n_inputs'] = 70_000
        if args['mnist_args']['invert_input_proba'] == 1:
            args['experiment_subname'] = f"inverted_{args['experiment_subname']}"
