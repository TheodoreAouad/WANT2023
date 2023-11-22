""" This file contains the arguments for the experiments on mnist.
It launches 100 random experiments. We apply regularization. We use positive weights reparametrization.
The changing hyperparmaeters are:
    - learning rate
    - loss function
    - regularization approximative: uniform vs normal
    - different coefficients for the loss
    - different delay for the regularization
    - bias reparametrization
"""
import torch.optim as optim

from deep_morpho.datasets.gray_to_channels_dataset import LevelsetValuesEqualIndex
from deep_morpho.models.bise_base import ClosestSelemEnum, BiseBiasOptimEnum, BiseWeightsOptimEnum
from deep_morpho.initializer import InitBimonnEnum, InitBiseEnum
from deep_morpho.experiments.random_parser import RandomParser, UniformSearch
from ..args_enforcers import enforcers


all_args = RandomParser(n_tries=100)

##### CHANGING ARGS #####

all_args["apply_one_hot_target"] = [
    False,  # For cross entropy loss
    True,
]

all_args["loss_regu_delay"] = [
    0,
    1000,
    5000,
    10000,
]

all_args["bias_optim_mode"] = [
    BiseBiasOptimEnum.RAW,
    BiseBiasOptimEnum.POSITIVE,
    BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED,
    BiseBiasOptimEnum.POSITIVE_INTERVAL_REPARAMETRIZED,
]


all_args["learning_rate"] = 10 ** UniformSearch(-3, -1)

all_args["loss_coefs"] = [
    # {"loss_data": 0, "loss_regu": 0.1},
    # {"loss_data": 1, "loss_regu": 0.1},
    {"loss_data": 1, "loss_regu": 0.01},
    {"loss_data": 1, "loss_regu": 0.001},
    # {"loss_data": 1, "loss_regu": 0.1},
]


all_args["loss_regu"] = [
    # ("RegularizationProjConstant", {"mode": "exact"}),
    ("RegularizationProjConstant", {"mode": "uniform"}),
    ("RegularizationProjConstant", {"mode": "normal"}),
    # ("RegularizationProjActivated", {}),
]

#########################


all_args["batch_seed"] = [None]

all_args["n_try"] = [0]

all_args["experiment_name"] = [
    "regu_constant_approx",
]

all_args["model"] = [
    ##### MORPHO ####
    # "BiMoNN",
    ##### CLASSIFIERS #####
    "BimonnDenseNotBinary",
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
    # "noistidataset",
    ##### CLASSIFICATION #####
    "mnistclassifchanneldataset",
]


# DATA ARGS
all_args["morp_operation"] = [None]


all_args["preprocessing"] = [  # for axspa roi
    None,
]

all_args["mnist_args"] = [
    {
        "threshold": 30,
        "size": (50, 50),
        "invert_input_proba": 0,
    },
]
all_args["mnist_gray_args"] = [
    {
        "n_gray_scale_values": 20,
        "size": (50, 50),
    }
]

all_args["fashionmnist_gray_args"] = [
    {
        "n_gray_scale_values": 20,
        "size": (50, 50),
    }
]

all_args["channel_classif_args"] = [
    {
        "levelset_handler_mode": LevelsetValuesEqualIndex,
        # "levelset_handler_args": {"n_values": 10},
        "levelset_handler_args": {"n_values": 1},
    }
]


all_args["n_inputs_train"] = [50_000]
all_args["n_inputs_val"] = [10_000]
all_args["n_inputs_test"] = [10_000]

# TRAINING ARGS

all_args["loss_data_str"] = ["BCELoss"]

all_args["optimizer"] = [
    optim.Adam,
    # optim.SGD
]
all_args["optimizer_args"] = [{}]
all_args["batch_size"] = [64]
all_args["num_workers"] = [
    20,
    # 0,
]
all_args["freq_imgs"] = [
    5e100,
    # "epoch"
]
all_args["freq_hist"] = [
    5e100,
    # "epoch"
]
all_args["freq_update_binary_batch"] = [
    # 1
    None
]
all_args["freq_update_binary_epoch"] = [
    1,
    # None,
]
all_args["freq_scalars"] = [50]
# all_args['max_epochs.trainer'] = [1]
all_args["max_epochs.trainer"] = [200]

all_args["patience_loss_batch"] = [2100]
all_args["patience_loss_epoch"] = [10]
all_args["patience_reduce_lr"] = [1 / 5]
all_args["early_stopping_on"] = [
    # 'batch',
    "epoch"
]


# MODEL ARGS

all_args["apply_last_activation"] = [
    # False,
    True,
]

all_args["atomic_element"] = [
    "bisel",
    # "dual_bisel",
]
all_args["n_atoms"] = [
    "adapt",
]


all_args["channels"] = [[4096]]
all_args["closest_selem_method"] = [
    ClosestSelemEnum.MIN_DIST_DIST_TO_CST,
    # ClosestSelemEnum.MIN_DIST_ACTIVATED_POSITIVE,
]


all_args["bias_optim_args"] = [{"offset": 0}]
all_args["weights_optim_mode"] = [
    BiseWeightsOptimEnum.THRESHOLDED,
]


all_args["threshold_mode"] = [
    {
        "weight": "softplus",
        "activation": "tanh",
    }
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
