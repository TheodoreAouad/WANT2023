""" This file contains the arguments for the experiments on the noised sticks dataset.
"""
import torch.optim as optim

from deep_morpho.datasets.generate_forms3 import get_random_diskorect_channels
from deep_morpho.datasets.gray_to_channels_dataset import LevelsetValuesEqualIndex


from deep_morpho.models.bise_base import ClosestSelemEnum, BiseBiasOptimEnum, BiseWeightsOptimEnum
from deep_morpho.models.activations import NormalizedTanh
from deep_morpho.initializer import InitBimonnEnum, InitBiseEnum
from deep_morpho.experiments.parser import GridParser

from ..args_enforcers import enforcers


all_args = GridParser()

all_args["batch_seed"] = [2249939862]

all_args["n_try"] = [0]

all_args["experiment_name"] = [
    "debug_noisti",
]

all_args["model"] = [
    ##### MORPHO ####
    "BiMoNN",
    ##### CLASSIFIERS #####
    # "BimonnDenseNotBinary",
    ##### CLASSICAL MODELS ####
    # "MLPBatchNormClassical",
    # "ResNet18",
    # "ResNet34",
    # "ResNet50",
    ###### BIBLIO ######
    # "BNNConv",
    # "MLPBinaryConnectMNIST",
    # "ConvNetBinaryConnectCifar10",
]

all_args["dataset"] = [
    ##### MORPHO ####
    "noistidataset",
    ##### CLASSIFICATION #####
    # "mnistclassifchanneldataset",
]


# DATA ARGS
all_args["morp_operation"] = [None]


all_args["in_ram"] = [
    # False,
    True,
]

##### Noisti Args #####
all_args["sticks_noised_angles"] = [[0, 45, 90]]
all_args["sticks_noised_args"] = [
    {
        "size": (70, 70),
        "n_shapes": 30,
        "lengths_lim": (12, 15),
        "widths_lim": (0, 0),
        "p_invert": 0,
        "border": (0, 0),
        "noise_proba": 0.1,
    }
]
########################


all_args["n_steps"] = [100]
all_args["nb_batch_indep"] = [0]

# TRAINING ARGS
all_args["learning_rate"] = [
    1e-2,
]

all_args["loss_data_str"] = [
    # nn.BCELoss(),
    # "MaskedBCENormalizedLoss",
    # "MaskedMSELoss",
    # "MaskedNormalizedDiceLoss",
    # "MaskedBCELoss",
    # "BCENormalizedLoss",
    # "BCELoss",
    # "CrossEntropyLoss",
    # "SquaredHingeLoss",
    "MSELoss",
    # "DiceLoss",
    # "MaskedDiceLoss",
    # "NormalizedDiceLoss",
]
all_args["loss_regu"] = [
    # ("quadratic", {"lower_bound": 0, "upper_bound": np.infty, "lambda_": 0.01})
    # "linear",
    "None",
    # ("RegularizationProjConstant", {"mode": "exact"}),
    # ("RegularizationProjConstant", {"mode": "uniform"}),
    # ("RegularizationProjConstant", {"mode": "normal"}),
    # ("RegularizationProjActivated", {}),
]
all_args["loss_coefs"] = [
    {"loss_data": 1, "loss_regu": 0},
    # {"loss_data": 1, "loss_regu": 100000},
    # {"loss_data": 1, "loss_regu": 0.1},
    # {"loss_data": 1, "loss_regu": 0.01},
    # {"loss_data": 1, "loss_regu": 0.001},
]
all_args["optimizer"] = [
    optim.Adam,
    # optim.SGD
]
all_args["optimizer_args"] = [{}]
all_args["batch_size"] = [256]
all_args["num_workers"] = [
    # 20,
    5
    # 0
]
all_args["freq_imgs"] = ["epoch"]
all_args["freq_hist"] = ["epoch"]
all_args["freq_update_binary_batch"] = [
    # 1
    None
]
all_args["freq_update_binary_epoch"] = [
    1,
    # None,
]
all_args["freq_scalars"] = [2]
# all_args["max_epochs.trainer"] = [3]  # DEBUG
all_args["max_epochs.trainer"] = [200]

all_args["patience_loss_batch"] = [2100]
# all_args['patience_loss_epoch'] = [15]
all_args["patience_loss_epoch"] = [1]  # DEBUG
all_args["patience_reduce_lr"] = [1 / 3]
# all_args["patience_reduce_lr"] = [5]  # DEBUG
all_args["early_stopping_on"] = [
    "batch",
    # "epoch"
]


# MODEL ARGS

all_args["atomic_element"] = [
    # "bisel",
    "dual_bisel",
    # "sybisel",
]
all_args["n_atoms"] = [
    "adapt",
]

all_args["kernel_size"] = [
    # [5, 5]
    "adapt",
]
all_args["channels"] = [
    # [1; 3, 3]
    "adapt",
]
all_args["closest_selem_method"] = [
    # ClosestSelemEnum.MIN_DIST
    # ClosestSelemEnum.MAX_SECOND_DERIVATIVE
    # ClosestSelemEnum.MIN_DIST_DIST_TO_CST,
    ClosestSelemEnum.MIN_DIST_ACTIVATED_POSITIVE,
]

all_args["bias_optim_mode"] = [
    # BiseBiasOptimEnum.RAW,
    BiseBiasOptimEnum.POSITIVE,
    # BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED,
    # BiseBiasOptimEnum.POSITIVE_INTERVAL_REPARAMETRIZED
]
all_args["bias_optim_args"] = [{"offset": 0}]
all_args["weights_optim_mode"] = [BiseWeightsOptimEnum.THRESHOLDED]

all_args["threshold_mode"] = [
    {
        "weight": "softplus",
        "activation": "tanh",
    },
]
all_args["weights_optim_args"] = [{"constant_P": True, "factor": 1}]

all_args["initializer_method"] = [
    InitBimonnEnum.INPUT_MEAN,
]
all_args["initializer_args"] = [
    {
        "bise_init_method": InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
        "lui_init_method": InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_RANDOM_BIAS,
        "bise_init_args": {"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto"},
    },
]

all_args["activation_P"] = [0]
all_args["constant_activation_P"] = [False]
all_args["force_lui_identity"] = [False]
all_args["constant_P_lui"] = [False]

all_args["observables"] = [[]]

all_args["args_enforcers"] = enforcers

all_args.parse_args()
