import pathlib
from os.path import join
import os

import numpy as np
import pandas as pd

from general.utils import list_dir_joined, load_json
from deep_morpho.models import LightningBiMoNN


PATH_OUT = "deep_morpho/metric_results/DGMM_2022.csv"
pathlib.Path(PATH_OUT).parent.mkdir(exist_ok=True, parents=True)


version_dict = {"version_0": "disk", "version_1": "stick", "version_2": "cross"}

paths_tb = {
    'diskorect': "deep_morpho/results/ICIP_2022/sandbox/4/diskorect",
    'mnist': "deep_morpho/results/ICIP_2022/sandbox/5/mnist",
    'inverted_mnist': "deep_morpho/results/ICIP_2022/sandbox/5/inverted_mnist",
}

selems = ['disk', 'hstick', 'dcross']
pd_data = (
    # DISKORECT
    [['diskorect', 'dilation', selem, f'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/dilation/version_{nb}'] for selem, nb in zip(selems, [0, 1, 2])] +
    [['diskorect', 'erosion', selem, f'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/erosion/version_{nb}'] for selem, nb in zip(selems, [0, 1, 2])] +
    [['diskorect', 'opening', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/opening/version_{nb}'] for selem, nb in zip(selems, [0, 1, 2])] +
    [['diskorect', 'closing', 'disk', 'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/closing/version_0']] +
    [['diskorect', 'closing', 'hstick', 'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/closing/version_2']] +
    [['diskorect', 'closing', 'dcross', 'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/closing/version_2']] +
    [['diskorect', 'black_tophat', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/black_tophat/version_{nb}'] for selem, nb in zip(selems, [0, 5, 10])] +
    [['diskorect', 'white_tophat', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/white_tophat/version_{nb}'] for selem, nb in zip(selems, [18, 19, 2])] +
    # MNIST
    [['mnist', 'dilation', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/dilation/version_{nb}'] for selem, nb in zip(selems, [0, 1, 2])] +
    [['mnist', 'erosion', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/erosion/version_{nb}'] for selem, nb in zip(selems, [0, 1, 2])] +
    [['mnist', 'opening', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/opening/version_{nb}'] for selem, nb in zip(selems, [4, 1, 2])] +
    [['mnist', 'closing', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/closing/version_{nb}'] for selem, nb in zip(selems, [0, 9, 6])] +
    [['mnist', 'black_tophat', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/black_tophat/version_{nb}'] for selem, nb in zip(selems, [0, 1, 2])] +
    [['mnist', 'white_tophat', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/white_tophat/version_{nb}'] for selem, nb in zip(selems, [0, 1, 2])] +
    # INVERTED MNIST
    [['inverted_mnist', 'dilation', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/dilation/version_{nb}'] for selem, nb in zip(selems, [0, 1, 2])] +
    [['inverted_mnist', 'erosion', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/erosion/version_{nb}'] for selem, nb in zip(selems, [0, 1, 2])] +
    [['inverted_mnist', 'opening', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/opening/version_{nb}'] for selem, nb in zip(selems, [3, 1, 4])] +
    [['inverted_mnist', 'closing', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/closing/version_{nb}'] for selem, nb in zip(selems, [5, 4, 2])] +
    [['inverted_mnist', 'black_tophat', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/black_tophat/version_{nb}'] for selem, nb in zip(selems, [0, 1, 2])] +
    [['inverted_mnist', 'white_tophat', selem, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/white_tophat/version_{nb}'] for selem, nb in zip(selems, [0, 1, 2])] +
    []
)

df = pd.DataFrame(
    columns=['dataset', 'op', 'selem', 'tb_path'],
    data=pd_data
)

df['dice'] = None
df['dice_binary'] = None
df['convergence_dice_train'] = None
df['convergence_dice_val'] = None
df['convergence_dice_test'] = None
df['convergence_selem_0'] = None
df['convergence_selem_1'] = None

# DILATION
# for dataset in ['diskorect', 'mnist', 'inverted_mnist']:
#     for op in ['dilation', 'erosion', 'opening', 'closing']:
#         # pathlib.Path(join(PATH_OUT, dataset, op)).mkdir(exist_ok=True, parents=True)
#         for tb_version in version_dict.keys():
for idx in range(len(df)):
    dataset = df['dataset'].iloc[idx]
    op = df['op'].iloc[idx]
    selem = df['selem'].iloc[idx]
    tb_path = df['tb_path'].iloc[idx]

    # tb_path = join(paths_tb[dataset], op, "bisel", tb_version)


    cur_dice = join(tb_path, 'observables', 'CalculateAndLogMetrics', 'metrics.json')
    cur_dice = eval(load_json(cur_dice)['dice'])
    df['dice'].iloc[idx] = cur_dice

    cur_dice = join(tb_path, 'observables', 'BinaryModeMetric', 'metrics.json')
    cur_dice = eval(load_json(cur_dice)['dice'])
    df['dice_binary'].iloc[idx] = cur_dice


    if "tophat" not in op:
        cur_conv = join(tb_path, 'observables', 'ConvergenceMetrics', 'convergence_step.json')
        cur_conv = load_json(cur_conv)
        for state, v in cur_conv.items():
            df[f'convergence_dice_{state}'].iloc[idx] = v['dice']

        binary_selem = join(tb_path, 'observables', 'ConvergenceBinary', 'convergence_step.json')
        binary_selem = load_json(binary_selem)['bisel']
        for layer_idx, v in binary_selem.items():
            df[f'convergence_selem_{eval(layer_idx)[0]}'].iloc[idx] = v

    # all_dfs.append(pd.DataFrame(dict(
    #     **{
    #         "tb_path": [tb_path],
    #         "dataset": [dataset],
    #         "operation": [op],
    #         "selem": [version_dict[tb_version]],
    #         "dice": [cur_dice],
    #     },
    #     **{f'convergence_dice_{state}': [v['dice']] for state, v in cur_conv.items()},
    #     **{f'convergence_selem_{eval(layer_idx)[0]}': [v] for layer_idx, v in binary_selem.items()},
    # )))


# all_dfs = pd.concat(all_dfs).reset_index(drop=True)
# all_dfs.to_csv(PATH_OUT, index=False)
df.to_csv(PATH_OUT, index=False)
