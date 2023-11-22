import re
from typing import Any, Dict
import os

import warnings

from deep_morpho.models import BiseBiasOptimEnum, ClosestSelemDistanceEnum, ClosestSelemEnum, InitBiseEnum, InitBimonnEnum


all_keys_line = [
    "experiment_name",
    "experiment_subname",
    "name",
    "dataset_type",
    "dataset_path",
    "n_inputs",
    "learning_rate",
    "batch_size",
    "num_workers",
    "freq_imgs",
    "n_epochs",
    "n_atoms",
    "atomic_element",
    "kernel_size",
    "init_weight_mode",
    "activation_P",
    "constant_activation_P",
    "constant_weight_P",
    "weight",
    "apply_one_hot_target.datamodule",
    # "activation"
    # "threshold_mode.net",
    "alpha_init",
    "share_weights",
    "loss",
    "patience_loss",
    "patience_reduce_lr",
    "init_weight_mode",
    "closest_selem_method",
    "closest_selem_distance_fn",
    "bias_optim_mode",
    "loss_regu_delay",
    "loss_regu_str",
    "loss_data_str",
    "binary_params",
    "float_params",
    "n_values",   # DEBUG: FOR LEVELST HANDLER, TODO: GENERALIZE
]
bise_keys = ["init_weight_mode", "initializer_method", "initializer_args", "bise_init_method", "closest_selem_method", "closest_selem_distance_fn", "bias_optim_mode"]



def load_args_from_str(yaml_str: str,) -> Dict:
    args = {}
    # args['loss'] = parse_yaml_dict_loss(yaml_str)
    args['optimizer'] = parse_yaml_dict_optimizer(yaml_str)
    # args['operations'] = parse_yaml_dict_operations(yaml_str)
    args['loss_data'] = parse_yaml_dict_loss_data(yaml_str)
    args['threshold_activation'] = parse_yaml_threshold_activation(yaml_str)
    args['kwargs_loss_regu'] = parse_yaml_kwargs_loss_regu(yaml_str)
    args['loss_coef_regu'] = parse_yaml_loss_regu_coefs_regu(yaml_str)
    args['loss_coef_data'] = parse_yaml_loss_regu_coefs_data(yaml_str)

    for key in all_keys_line:
        args[key] = parse_yaml_dict_key_line(yaml_str, key)


    for bise_key in bise_keys:
        args[bise_key] = parse_yaml_bise_arguments(yaml_str, bise_key)

    return args


def load_args(path: str) -> Dict:
    if not os.path.exists(path):
        warnings.warn(f"{path} not found.")
        args = {k: None for k in all_keys_line + bise_keys + ['optimizer', 'operations', 'loss_data', ]}
        return args

    with open(path, "r") as f:
        yaml_str = f.read()

    return load_args_from_str(yaml_str)


def regex_find_or_none(regex: str, st: str, *args, group_nb: int = -1, **kwargs,):
    exps = re.findall(regex, st, *args, **kwargs)
    if len(exps) == 0:
        return None
    
    if len(exps) > 1:
        warnings.warn(f"More than one match for {regex} in {st}")
        exps = [exps[0]]
    # assert len(exps) == 1, exps

    # for multiple parenthesis, we have to select the group. If there is only one group, -1
    if group_nb == -1:
        return exps[0]
    return exps[0][group_nb]


def parse_yaml_dict_key_line(yaml_str: str, key: str) -> Any:
    return regex_find_or_none(f"( |^|\n){key}: ([^\n]+)\n", yaml_str, group_nb=1)


def parse_yaml_dict_loss(yaml_str: str) -> Any:
    return regex_find_or_none(r"\n?loss[^\n]+\.(\w+)\n", yaml_str)


def parse_yaml_dict_loss_data(yaml_str: str) -> Any:
    return regex_find_or_none(r"\n?   loss_data[^\n]+\.(\w+)\n", yaml_str)


def parse_yaml_dict_optimizer(yaml_str: str) -> Any:
    return regex_find_or_none(r"\n?optimizer[^\n]+\.(\w+)[ \n]", yaml_str)


def parse_yaml_dict_bise_init_method(yaml_str: str) -> Any:
    return regex_find_or_none(r"\n?\t?bise_init_method[^\n]+\n +- (\d+)", yaml_str)


def parse_yaml_threshold_activation(yaml_str: str) -> Any:
    # return regex_find_or_none(r'threshold_mode\.net:(.*?)(?=^\s*\w+\.)', yaml_str, re.MULTILINE | re.DOTALL)
    return parse_yaml_dict_key_line(yaml_str, "activation")


def parse_yaml_kwargs_loss_regu(yaml_str: str) -> Any:
    return regex_find_or_none(r'kwargs_loss_regu:\s*\n\s*mode:\s*(.*?)\s*\n', yaml_str,)


def parse_yaml_loss_regu_coefs_regu(yaml_str: str) -> Any:
    return regex_find_or_none(r'loss_coefs:.*?loss_regu:(\s*(.*?)\s*(?:\n|$))', yaml_str, re.DOTALL)


def parse_yaml_loss_regu_coefs_data(yaml_str: str) -> Any:
    return regex_find_or_none(r'loss_coefs:.*?loss_data:(\s*(.*?)\s*(?:\n|$))', yaml_str, re.DOTALL)


# deprecated
def parse_yaml_dict_operations(yaml_str: str) -> Any:
    idx0 = yaml_str.find('operations')
    if idx0 == -1:
        return None

    idx = idx0 + len('operations:\n')
    all_ops = []
    while True and idx < len(yaml_str):
        sent = ""
        while yaml_str[idx] == " ":
            idx += 1

        if yaml_str[idx] != "-":
            break

        while yaml_str[idx] != "\n":
            sent += yaml_str[idx]
            idx += 1

        all_ops.append(sent[2:])
        idx += 1

    return all_ops


def parse_yaml_bise_arguments(yaml_str: str, key: str) -> Any:
    key_to_enum = {
        'closest_selem_method': ClosestSelemEnum,
        'closest_selem_distance_fn': ClosestSelemDistanceEnum,
        'bias_optim_mode': BiseBiasOptimEnum,
        'init_weight_mode': InitBiseEnum,
        'initializer_method': InitBimonnEnum,
        'bise_init_method': InitBiseEnum,
    }
    if f'{key}_str' in yaml_str:
        res = parse_yaml_dict_key_line(yaml_str, f'{key}_str')

    else:
        if key == "bise_init_method":
            enum_int = parse_yaml_dict_bise_init_method(yaml_str)
        else:
            enum_int = regex_find_or_none(rf"\n?{key}[^\n]+\n- (\d+)\n", yaml_str)

        if enum_int is not None:
            res = str(key_to_enum[key](int(enum_int)))
        else:
            return None

    if "." in res:
        return res.split('.')[-1]
    return res
