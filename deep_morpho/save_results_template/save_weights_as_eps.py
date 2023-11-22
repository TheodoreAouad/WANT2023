import pathlib
from importlib import import_module
from unittest.mock import _patch_dict
import warnings
from PIL import Image
from os.path import join
import os

import numpy as np
import matplotlib.pyplot as plt


from general.utils import list_dir_joined



# TB_PATHS = (
#     # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/dilation_size_7x7_bise')) +
#     # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/erosion_size_7x7_bise')) +
#     # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/opening_size_7x7_bise')) +
#     # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/closing_size_7x7_bise'))
#     # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_46/opening_bisel'))
#     sum([sorted(list_dir_joined(f'deep_morpho/results/ICIP_2022/sandbox/4/diskorect/{op}/bisel')) for op in ['dilation', 'erosion', 'opening', 'closing']], start=[]) +
#     sum([sorted(list_dir_joined(f'deep_morpho/results/ICIP_2022/sandbox/5/inverted_mnist/{op}/bisel')) for op in ['dilation', 'erosion', 'opening', 'closing']], start=[]) +
#     sum([sorted(list_dir_joined(f'deep_morpho/results/ICIP_2022/sandbox/5/mnist/{op}/bisel')) for op in ['dilation', 'erosion', 'opening', 'closing']], start=[])
# )

EXT = "png"
# EXT = "eps"
PATH_OUT = f"deep_morpho/weights_{EXT}/dgmm_2022_poster"
pathlib.Path(PATH_OUT).mkdir(exist_ok=True, parents=True)
pathlib.Path(PATH_OUT.replace("weights", "binarized")).mkdir(exist_ok=True, parents=True)


def save_img(ar, savepath, cmap='viridis'):
    cm = plt.get_cmap(cmap)
    ar = cm(ar)[..., :-1]
    ar = np.uint8(ar  / ar.max() * 255)
    img = Image.fromarray(ar).resize((50, 50), resample=Image.NEAREST)
    img.save(savepath)


selems = ['disk', 'hstick', 'dcross']
tb_path_dict = dict(
    # DISKORECT
    **{str(('diskorect', 'dilation', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/0/softplus/diskorect/dilation/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('diskorect', 'erosion', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/0/softplus/diskorect/erosion/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('diskorect', 'opening', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/diskorect/opening/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('diskorect', 'closing', 'disk')): 'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/0/softplus/diskorect/closing/version_0'},
    **{str(('diskorect', 'closing', 'hstick')): 'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/diskorect/closing/version_2'},
    **{str(('diskorect', 'closing', 'dcross')): 'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/0/softplus/diskorect/closing/version_2'},
    **{str(('diskorect', 'black_tophat', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/diskorect/black_tophat/version_{nb}' for selem, nb in zip(selems, [0, 5, 10])},
    **{str(('diskorect', 'white_tophat', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/diskorect/white_tophat/version_{nb}' for selem, nb in zip(selems, [18, 19, 2])},
    # # MNIST
    **{str(('mnist', 'dilation', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/mnist/dilation/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('mnist', 'erosion', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/mnist/erosion/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('mnist', 'opening', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/mnist/opening/version_{nb}' for selem, nb in zip(selems, [4, 1, 2])},
    **{str(('mnist', 'closing', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/mnist/closing/version_{nb}' for selem, nb in zip(selems, [0, 9, 6])},
    **{str(('mnist', 'black_tophat', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/mnist/black_tophat/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('mnist', 'white_tophat', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/mnist/white_tophat/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    # # INVERTED MNIST
    **{str(('inverted_mnist', 'dilation', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/inverted_mnist/dilation/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('inverted_mnist', 'erosion', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/inverted_mnist/erosion/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('inverted_mnist', 'opening', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/inverted_mnist/opening/version_{nb}' for selem, nb in zip(selems, [3, 1, 4])},
    **{str(('inverted_mnist', 'closing', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/inverted_mnist/closing/version_{nb}' for selem, nb in zip(selems, [5, 4, 2])},
    **{str(('inverted_mnist', 'black_tophat', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/inverted_mnist/black_tophat/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('inverted_mnist', 'white_tophat', selem)): f'deep_morpho/results/results_tensorboards/DGMM_2022/sandbox/1/softplus/inverted_mnist/white_tophat/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
)

# DILATION
for dataset in ['diskorect', 'mnist', 'inverted_mnist']:
    for selem in ['disk', 'hstick', 'dcross']:
        # for op in ['dilation', 'erosion']:
        #     if str((dataset, op, selem)) not in tb_path_dict.keys():
        #         warnings.warn(f"{str((dataset, op, selem))} not found.")
        #         continue

        #     tb_path = tb_path_dict[str((dataset, op, selem))]
        #     path_weights = os.listdir(join(tb_path, "checkpoints"))[0]

        #     light_bimonn = import_module(f"{tb_path.replace('/', '.')}.code.deep_morpho.models.lightning_bimonn")
        #     # light_bimonn = import_from_path(join(tb_path, "code", "deep_morpho", "models", "lightning_bimonn.py"))

        #     model = light_bimonn.LightningBiMoNN.load_from_checkpoint(join(tb_path, 'checkpoints', path_weights))
        #     selem1, operation, distance = model.model.layer1.bises[0].find_closest_selem_and_operation_chan(0, v1=0, v2=1)


        #     save_img(
        #         model.model.layer1.normalized_weight.detach().cpu()[0, 0].numpy(),
        #         join(PATH_OUT, f"{dataset}_{op}_{selem}.{EXT}")
        #     )
        #     save_img(
        #         selem1,
        #         join(PATH_OUT, f"{dataset}_{op}_{selem}.{EXT}").replace("weights", "binarized"), cmap='gray'
        #     )

        # for op in ['opening', 'closing']:

        #     tb_path = tb_path_dict[str((dataset, op, selem))]
        #     path_weights = os.listdir(join(tb_path, "checkpoints"))[0]

        #     light_bimonn = import_module(f"{tb_path.replace('/', '.')}.code.deep_morpho.models.lightning_bimonn")
        #     model = light_bimonn.LightningBiMoNN.load_from_checkpoint(join(tb_path, 'checkpoints', path_weights))

        #     selem1, _, _ = model.model.layer1.bises[0].find_closest_selem_and_operation_chan(0, v1=0, v2=1)
        #     selem2, _, _ = model.model.layer2.bises[0].find_closest_selem_and_operation_chan(0, v1=0, v2=1)

        #     save_img(
        #         model.model.layer1.normalized_weight.detach().cpu()[0, 0].numpy(),
        #         join(PATH_OUT, f"{dataset}_{op}_{selem}1.{EXT}")
        #     )
        #     save_img(
        #         model.model.layer2.normalized_weight.detach().cpu()[0, 0].numpy(),
        #         join(PATH_OUT, f"{dataset}_{op}_{selem}2.{EXT}")
        #     )

        #     save_img(
        #         selem1,
        #         join(PATH_OUT, f"{dataset}_{op}_{selem}1.{EXT}").replace("weights", "binarized"),
        #         cmap='gray'
        #     )
        #     save_img(
        #         selem2,
        #         join(PATH_OUT, f"{dataset}_{op}_{selem}2.{EXT}").replace("weights", "binarized"),
        #         cmap='gray'
        #     )

        for op in ['black_tophat', 'white_tophat']:
            tb_path = tb_path_dict[str((dataset, op, selem))]
            path_weights = os.listdir(join(tb_path, "checkpoints"))[0]

            light_bimonn = import_module(f"{tb_path.replace('/', '.')}.code.deep_morpho.models.lightning_bimonn")

            model = light_bimonn.LightningBiMoNN.load_from_checkpoint(join(tb_path, 'checkpoints', path_weights))


            bimonn_viz = import_module(f"deep_morpho.viz.bimonn_viz")
            # bimonn_viz = import_module(f"{tb_path.replace('/', '.')}.code.deep_morpho.viz.bimonn_viz")

            bimonn_viz.BimonnVizualiser(model.model, mode="weights").save_fig(join(PATH_OUT, f"{dataset}_{op}_{selem}.{EXT}"))
            bimonn_viz.BimonnVizualiser(model.model, mode="closest").save_fig(join(PATH_OUT, f"{dataset}_{op}_{selem}.{EXT}").replace("weights", "binarized"))
