from typing import Union, Tuple, List, Callable, Any, Dict
import copy

from matplotlib import pyplot as plt
import numpy as np
import torch
import skimage.morphology as morp

from general.structuring_elements import *
from general.array_morphology import (
    array_erosion, array_dilation, array_union_chans, array_intersection_chans, array_intersection_gray_chans, array_union_gray_chans
)
from .viz.morp_operations_viz import MorpOperationsVizualiser


# erosion, dilation, union, intersection = morp.binary_erosion, morp.binary_dilation, array_union_chans, array_intersection_chans
erosion, dilation, union, intersection = array_erosion, array_dilation, array_union_chans, array_intersection_chans


class ParallelMorpOperations:
    """
    Class to apply sequences of intersection / union of dilations / erosions
    """
    str_to_selem_fn = {
        'disk': disk, 'vstick': vstick, 'hstick': hstick, 'square': square, 'dcross': dcross, 'scross': scross,
        'vertical_stick': vstick, 'horizontal_stick': hstick, 'diagonal_cross': dcross, 'straight_cross': scross,
        'identity': identity, 'bsquare': bsquare, 'bdiamond': bdiamond, 'bcomplex': bcomplex,
    }

    def __init__(
        self,
        operations: List[List[List[Union[Tuple[Union[Callable, str], Union[Callable, Tuple[str, int]]], 'str', Callable]]]],
        device="cpu",
        # return_numpy_array: bool = False,
        name: str = None,
        str_to_fn: Dict = {'dilation': dilation, 'erosion': erosion},
        str_to_ui_fn: Dict = {'union': union, 'intersection': intersection}
    ):
        self.operations_original = operations
        self.device = device
        # self.return_numpy_array = return_numpy_array
        self.name = name
        self.str_to_fn = str_to_fn
        self.str_to_ui_fn = str_to_ui_fn

        self.operations = None
        self.operation_names = None
        self.selem_names = None
        self.selem_args = None
        self.in_channels = None
        self.out_channels = None
        self.selems = None
        self.do_complementation = None
        self.max_selem_shape = None

        self.convert_ops(operations)


    def _erodila_selem_converter(self, args):
        """ Scraps the argument of the creation of the structuring element.

        Args:
            args (np.ndarray | tuple):
                np.ndarray: the structuring element
                tuple: len == 2, the selem name (str | callable) and the selem args (Any)

        Returns:
            np.ndarray: the structuring element
            (str | None): the name of the structuring element
            (Any | None): the arguments of the creation of the structuring element
        """
        selem_name = None
        selem_args = None
        if not isinstance(args, tuple):
            return args, selem_name, selem_args
        if isinstance(args[0], str):
            selem_name = args[0]
            selem_op = self.str_to_selem_fn[selem_name]
        else:
            selem_op = args[0]

        selem_args = args[1]
        return selem_op(selem_args), selem_name, selem_args


    def _erodila_op_converter(self, args):
        """ Scraps the argument of the creation of an erosion / dilation operation.

        Args:
            args (callable | tuple):
                callable: the erosion or dilation function
                tuple:
                    len(args) == 2: (operation name: str, operation selem: (tuple | np.ndarray))
                    len(args) == 3: (operation name: str, operation selem: (tuple | np.ndarray), do complementation: bool)

        Returns:
            callable: the operation
            (str | None): the name of the operation
            (str | None): the name of the selem
            (str | None): the name of the arguments of the selem
            (np.ndarray | None): the selem
        """

        op_name = None
        selem_name = None
        selem_args = None
        selem = None
        do_complementation = False

        if not isinstance(args, tuple):
            return args, op_name, selem_name, selem_args, selem

        if isinstance(args[0], str):
            op_name = args[0]
            op_fn = self.str_to_fn[args[0]]
        else:
            op_fn = args[0]

        selem, selem_name, selem_args = self._erodila_selem_converter(args[1])

        op_fn2 = lambda x: op_fn(x, selem, )

        if len(args) == 3 and args[-1]:
            final_operation = lambda x: 1 - op_fn2(x)
            # final_operation = lambda x: ~op_fn2(x)
            do_complementation = True
        else:
            final_operation = op_fn2

        return (
            final_operation,
            op_name,
            selem_name,
            selem_args,
            selem,
            do_complementation,
        )


    def _ui_converter(self, args):
        """ Scraps the arguments for the creation of a union / intersection of channels.

        Args:
            args (callable | tuple):
                callable: the union / intersection operation
                tuple: len == 2. The operation (str | callable) and the arguments (str | list)

        Returns:
            callable: the union / intersection operation
            (str | None): the name of the operation
            str: the arguments of the operation
        """
        ui_name = None
        ui_args = "all"

        if isinstance(args, tuple):
            ui_fn = args[0]
            ui_args = args[1]
        else:
            ui_fn = args

        if isinstance(ui_fn, str):
            ui_name = ui_fn
            ui_fn = self.str_to_ui_fn[ui_name]

        return lambda x: ui_fn(x, ui_args), ui_name, ui_args


    def convert_ops(self, all_op_str):
        alls = {key: [] for key in ["op_fn", "op_names", "selem_names", "selem_args", "selems", "do_complementation"]}
        layers = {key: [] for key in ["op_fn", "op_names", "selem_names", "selem_args", "selems", "do_complementation"]}
        chans = {key: [] for key in ["op_fn", "op_names", "selem_names", "selem_args", "selems", "do_complementation"]}

        max_selem_shape = []
        in_channels = []
        out_channels = []

        for layer_str in all_op_str:
            for key in layers.keys():
                layers[key] = []
            out_channels.append(len(layer_str))
            in_channels.append(len(layer_str[0]) - 1)
            cur_max_shape = (0, 0)

            for chan_str in layer_str:

                for key in chans.keys():
                    chans[key] = []

                for cur_op_str in chan_str[:-1]:
                    # in all_erodila_res: op_fn, op_name, selem_name, selem_args, selem, do_complementation
                    all_erodila_res = self._erodila_op_converter(cur_op_str)
                    cur_max_shape = np.maximum(all_erodila_res[4].shape, cur_max_shape)

                    for key, res in zip(chans.keys(), all_erodila_res):
                        chans[key].append(res)

                    # chans["op_fn"].append(op_fn)
                    # chans["op_names"].append(op_name)
                    # chans['selem_names'].append(selem_name)
                    # chans['selem_args'].append(selem_args)
                    # chans['selems'].append(selem)
                    # chans['']

                # in all_ui_res: ui_fn, ui_name, ui_args
                all_ui_res = self._ui_converter(chan_str[-1])
                # ui_fn, ui_name, ui_args = self._ui_converter(chan_str[-1])

                for key, res in zip(["op_fn", "op_names", "selem_args"], all_ui_res):
                    chans[key].append(res)
                chans['selem_names'].append("channels")

                # chans["op_fn"].append(ui_fn)
                # chans["op_names"].append(ui_name)
                # chans["selem_args"].append(ui_args)

                for key in layers.keys():
                    layers[key].append(chans[key])

            for key in alls.keys():
                alls[key].append(layers[key])
            max_selem_shape.append(max(cur_max_shape))

        self.operations = alls['op_fn']
        self.operation_names = alls['op_names']
        self.selem_names = alls['selem_names']
        self.selem_args = alls['selem_args']
        self.selems = alls['selems']
        self.do_complementation = alls['do_complementation']

        assert in_channels[1:] == out_channels[:-1]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_selem_shape = max_selem_shape

        return alls

    def apply_layer(self, layer, x):
        # next_x = torch.zeros(x.shape[:-1] + (len(layer),))
        next_x = np.zeros(x.shape[:-1] + (len(layer),))
        for chan_idx, chan in enumerate(layer):
            morps, ui = chan[:-1], chan[-1]
            next_x[..., chan_idx] = ui(
                np.stack([morps[idx](x[..., idx]) for idx in range(len(morps))], axis=-1)
            )
        return next_x

    def apply_ops(self, ar):
        x = ar + 0
        for layer in self.operations:
            # next_x = torch.zeros(x.shape[:-1] + (len(layer),))
            # for chan_idx, chan in enumerate(layer):
            #     morps, ui = chan[:-1], chan[-1]
            #     next_x[..., chan_idx] = ui(
            #         np.stack([morps[idx](x[..., idx]) for idx in range(len(morps))], axis=-1)
            #     )
            # x = next_x
            x = self.apply_layer(layer, x)
        # if not self.return_numpy_array:
        #     if not isinstance(x, torch.Tensor):
        #         return torch.tensor(x)
        return x

    def __call__(self, ar):
        return self.apply_ops(ar)

    def __len__(self):
        return len(self.selems)

    def __repr__(self):
        repr_ = f'{self.__class__.__name__}(in_channels={self.in_channels[0]}, out_channels={self.out_channels[-1]})'
        for layer_idx in range(len(self.operation_names)):
            layer = self.operation_names[layer_idx]
            repr_ += f"\n{' '*4}Layer{layer_idx}(in_channels={self.in_channels[layer_idx]}, out_channels={self.out_channels[layer_idx]})"
            for chan_idx in range(len(layer)):
                chan = layer[chan_idx]

                if chan[-1] == 'intersection':
                    ui_name = 'inter'
                else:
                    ui_name = chan[-1]

                repr_ += f"\n{' '*8}Out{chan_idx}: {ui_name}(chans={self.selem_args[layer_idx][chan_idx][-1]}) |"
                for op_idx in range(len(self.operation_names[layer_idx][chan_idx]) - 1):
                    to_add = f"{chan[op_idx]}({self.selem_names[layer_idx][chan_idx][op_idx]}({self.selem_args[layer_idx][chan_idx][op_idx]}))"
                    if self.do_complementation[layer_idx][chan_idx][op_idx]:
                        to_add = f"complement({to_add})"
                    repr_ += f" {to_add}"

        return repr_


    def get_saved_key(self):
        return (
            '=>'.join(self.operations) +
            ' -- ' +
            "=>".join([f'{fn.__name__}({arg})' for fn, arg in zip(self._selem_fn, self._selem_arg)])
        )

    @staticmethod
    def complementation(size: int, *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'opening')
        return ParallelMorpOperations(
            name='complementation',
            operations=[[[('erosion', ('identity', size), True), 'union']]],
            *args,
            **kwargs
        )

    @staticmethod
    def identity(size: int, *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'opening')
        return ParallelMorpOperations(
            name='complementation',
            operations=[[[('erosion', ('identity', size), False), 'union']]],
            *args,
            **kwargs
        )

    @staticmethod
    def erosion(selem: Union[Callable, np.ndarray, Tuple[Union[Callable, str], Any]], *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'erosion')
        return ParallelMorpOperations(
            operations=[[[('erosion', selem, False), 'union']]],
            *args,
            **kwargs
        )

    @staticmethod
    def erosion_gray(selem: Union[Callable, np.ndarray, Tuple[Union[Callable, str], Any]], *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'erosion_gray')
        return ParallelMorpOperations(
            operations=[[[('erosion', selem, False), 'union']]],
            str_to_fn={"dilation": morp.dilation, "erosion": morp.erosion},
            str_to_ui_fn={"intersection": lambda *x: x[0][..., 0], "union": lambda *x: x[0][..., 0]},
            *args,
            **kwargs
        )

    @staticmethod
    def dilation(selem: Union[Callable, np.ndarray, Tuple[Union[Callable, str], Any]], *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'dilation')
        return ParallelMorpOperations(
            operations=[[[('dilation', selem, False), 'union']]],
            *args,
            **kwargs
        )

    @staticmethod
    def dilation_gray(selem: Union[Callable, np.ndarray, Tuple[Union[Callable, str], Any]], *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'dilation_gray')
        return ParallelMorpOperations(
            operations=[[[('dilation', selem, False), 'union']]],
            str_to_fn={"dilation": morp.dilation, "erosion": morp.erosion},
            str_to_ui_fn={"intersection": lambda *x: x[0][..., 0], "union": lambda *x: x[0][..., 0]},
            *args,
            **kwargs
        )

    @staticmethod
    def opening(selem: Union[Callable, np.ndarray, Tuple[Union[Callable, str], Any]], *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'opening')
        return ParallelMorpOperations(
            operations=[
                [[('erosion', selem, False), 'union']],
                [[('dilation', selem, False), 'union']],
            ],
            *args,
            **kwargs
        )

    @staticmethod
    def opening_gray(selem: Union[Callable, np.ndarray, Tuple[Union[Callable, str], Any]], *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'opening_gray')
        return ParallelMorpOperations(
            operations=[
                [[('erosion', selem, False), 'union']],
                [[('dilation', selem, False), 'union']],
            ],
            str_to_fn={"dilation": morp.dilation, "erosion": morp.erosion},
            str_to_ui_fn={"intersection": array_intersection_gray_chans, "union": array_union_gray_chans},
            # str_to_ui_fn={"intersection": lambda *x: x[0][..., 0], "union": lambda *x: x[0][..., 0]},
            *args,
            **kwargs
        )

    @staticmethod
    def closing(selem: Union[Callable, np.ndarray, Tuple[Union[Callable, str], Any]], *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'closing')
        return ParallelMorpOperations(
            operations=[
                [[('dilation', selem, False), 'union']],
                [[('erosion', selem, False), 'union']],
            ],
            *args,
            **kwargs
        )

    @staticmethod
    def closing_gray(selem: Union[Callable, np.ndarray, Tuple[Union[Callable, str], Any]], *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'closing_gray')
        return ParallelMorpOperations(
            operations=[
                [[('dilation', selem, False), 'union']],
                [[('erosion', selem, False), 'union']],
            ],
            str_to_fn={"dilation": morp.dilation, "erosion": morp.erosion},
            str_to_ui_fn={"intersection": array_intersection_gray_chans, "union": array_union_gray_chans},
            # str_to_ui_fn={"intersection": lambda *x: x[0][..., 0], "union": lambda *x: x[0][..., 0]},
            *args,
            **kwargs
        )

    @staticmethod
    def white_tophat(selem: Union[Callable, np.ndarray, Tuple[Union[Callable, str], Any]], *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'white_tophat')
        identity = ('dilation', ('disk', 0), False)
        return ParallelMorpOperations(
            operations=[
                [
                    [identity, 'union'],
                    [('erosion', selem, False), 'union'],
                ],
                [[identity, ('dilation', selem, True), 'intersection']]
            ],
            **kwargs
        )


    @staticmethod
    def black_tophat(selem: Union[Callable, np.ndarray, Tuple[Union[Callable, str], Any]], *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'black_tophat')
        identity1 = ('dilation', ('disk', 0), False)
        identity2 = ('dilation', ('disk', 0), True)
        return ParallelMorpOperations(
            operations=[
                [
                    [identity1, 'union'],
                    [('dilation', selem, False), 'union'],
                ],
                [[identity2, ('erosion', selem, False), 'intersection']]
            ],
            str_to_fn={"dilation": morp.dilation, "erosion": morp.erosion},
            str_to_ui_fn={"intersection": array_intersection_gray_chans, "union": array_union_gray_chans},
            **kwargs
        )

    @staticmethod
    def hit_or_miss_transform(selem1, selem2, *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'hit_or_miss_transform')
        identity = ('dilation', ('disk', 0), False)
        return ParallelMorpOperations(
            operations=[
                [
                    [identity, 'union'],
                    [identity, 'union'],
                ],
                [
                    [('erosion', selem1, False), ('dilation', selem2, True), 'intersection']
                ]
            ],
            **kwargs,
        )

    @staticmethod
    def remove_isolated_points():
        selem1 = np.zeros((3, 3))
        selem1[1, 1] = 1

        selem2 = np.ones((3, 3))
        selem2[1:-1, 1:-1] = 0

        identity = ('dilation', ('identity', 3), False)
        return ParallelMorpOperations(
            name='remove_isolated_points',
            operations=[
                [
                    [identity, 'union'],
                    [identity, 'union'],
                    [identity, 'union'],
                    [identity, 'union'],
                ],
                [
                    [identity, ('erosion', selem1, True), identity, identity, ('intersection', [0, 1])],
                    [identity, identity, identity, ('dilation', selem2, False), ('intersection', [2, 3])],
                ],
                [
                    [identity, identity, 'union']
                ]
            ]
        )

    @staticmethod
    def translation(tx: int, ty: int, *args, **kwargs):
        kwargs["name"] = kwargs.get("name", "translation")
        tx, ty = int(tx), int(ty)
        final_shape = 2 * max(np.abs(tx), np.abs(ty)) + 1

        if final_shape % 2 == 0:
            final_shape += 1

        selem = np.zeros((final_shape, final_shape))
        selem[final_shape // 2 - tx, final_shape // 2 - ty] = 1

        print(selem)

        return ParallelMorpOperations(operations=[[[("dilation", selem, False), "union"]]], *args, **kwargs)

    @property
    def ui_arrays(self):
        ui_ars = []
        for layer in self.operations_original:
            layer_ars = []
            for chan_output in layer:
                ui_fn, ui_name, ui_args = self._ui_converter(chan_output[-1])
                n_ops = len(chan_output) - 1
                ui_args = range(n_ops) if isinstance(ui_args, str) else ui_args

                cur_ar = np.zeros(n_ops, dtype=np.uint8)
                cur_ar[ui_args] = 1

                layer_ars.append((ui_name, cur_ar))
            ui_ars.append(layer_ars)
        return ui_ars

    def plot_ui_arrays(self, *args, **kwargs):
        """ Gives figures with the binary arrays of the intersection / union.

        Returns:
            dict(figs): all the figures of the binary arrays.
        """
        all_figs = {}
        ui_ars = self.ui_arrays
        for layer_idx, layer in enumerate(ui_ars):
            for chan_output, (op_name, op_ar) in enumerate(layer):
                fig, ax = plt.subplots(1, 1, *args, **kwargs)
                ax.set_title(f"layer {layer_idx} | chan output {chan_output} | {op_name}")

                array_no_compl = np.where(op_ar, ~np.array(self.do_complementation[layer_idx][chan_output]), 0)
                array_compl = np.where(op_ar, np.array(self.do_complementation[layer_idx][chan_output]), 0)
                final_array = np.concatenate([array_no_compl, array_compl])

                ax.imshow(final_array[:, None], vmin=0, vmax=1, interpolation='nearest', cmap="gray")
                ax.set_xticks([])
                ax.set_yticks(range(len(final_array)))
                all_figs[(layer_idx, chan_output)] = fig
        return all_figs


    def plot_selem_arrays(self, *args, **kwargs):
        all_figs = {}
        for layer_idx, layer in enumerate(self.selems):
            for chan_output, selems in enumerate(layer):
                for chan_input, selem in enumerate(selems):
                    fig, ax = plt.subplots(1, 1, *args, **kwargs)
                    op_str = self.operation_names[layer_idx][chan_output][chan_input]
                    if self.do_complementation[layer_idx][chan_output][chan_input]:
                        op_str = f'complement({op_str})'
                    ax.set_title(
                        f"layer {layer_idx} | (chan in, chan out) ({chan_input}, {chan_output})  "
                        f"| {op_str}")
                    ax.imshow(selem, vmin=0, vmax=1, interpolation='nearest', cmap="gray")
                    ax.set_xticks(range(selem.shape[0]))
                    ax.set_yticks(range(selem.shape[1]))
                    all_figs[(layer_idx, chan_input, chan_output)] = fig
        return all_figs

    def vizualise(self, **kwargs):
        vizualiser = MorpOperationsVizualiser(self)
        canva = vizualiser.draw(**kwargs)
        return canva

    def concatenate(self, morp_operations: List["ParallelMorpOperations"], name: str = None) -> "ParallelMorpOperations":
        if isinstance(morp_operations, ParallelMorpOperations):
            morp_operations = [morp_operations]

        res = copy.deepcopy(self)
        final_name = res.name

        for morp_operation in morp_operations:
            assert res.out_channels[-1] == morp_operation.in_channels[0]
            for attr in [
                "operations",
                "operation_names",
                "selem_names",
                "selem_args",
                "selems",
                "do_complementation",
                "in_channels",
                "out_channels",
                "max_selem_shape",
            ]:
                getattr(res, attr).extend(getattr(morp_operation, attr))
            final_name = f"{final_name}_{morp_operation.name}"

        if name is None:
            name = final_name

        res.name = name

        return res

    def get_selem(self, layer_idx: int, chan_input: int, chan_output: int) -> np.ndarray:
        return self.selems[layer_idx][chan_output][chan_input]

    def get_selem_name(self, layer_idx: int, chan_input: int, chan_output: int) -> str:
        return self.selem_names[layer_idx][chan_output][chan_input]

    def get_selem_args(self, layer_idx: int, chan_input: int, chan_output: int) -> Any:
        return self.selem_args[layer_idx][chan_output][chan_input]

    def get_operation_name(self, layer_idx: int, chan_input: int, chan_output: int) -> Callable:
        return self.operation_names[layer_idx][chan_output][chan_input]



class ParallelMorpOperationsGrayTopHats(ParallelMorpOperations):

    def __init__(self, *args, tophat_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.tophat_fn = tophat_fn

    def apply_ops(self, ar):
        return self.tophat_fn(ar[..., 0], self.selem_tophat)[..., None]

    @property
    def selem_tophat(self):
        return self.selems[0][1][0]

    @staticmethod
    def white_tophat_gray(selem: Union[Callable, np.ndarray, Tuple[Union[Callable, str], Any]], *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'white_tophat')
        identity = ('dilation', ('disk', 0), False)
        return ParallelMorpOperationsGrayTopHats(
            operations=[
                [
                    [identity, 'union'],
                    [('erosion', selem, False), 'union'],
                ],
                [[identity, ('dilation', selem, True), 'intersection']]
            ],
            tophat_fn=morp.white_tophat,
            **kwargs
        )

    @staticmethod
    def black_tophat_gray(selem: Union[Callable, np.ndarray, Tuple[Union[Callable, str], Any]], *args, **kwargs):
        kwargs["name"] = kwargs.get("name", 'black_tophat')
        identity1 = ('dilation', ('disk', 0), False)
        identity2 = ('dilation', ('disk', 0), True)
        return ParallelMorpOperationsGrayTopHats(
            operations=[
                [
                    [identity1, 'union'],
                    [('dilation', selem, False), 'union'],
                ],
                [[identity2, ('erosion', selem, False), 'intersection']]
            ],
            tophat_fn=morp.black_tophat,
            **kwargs
        )
