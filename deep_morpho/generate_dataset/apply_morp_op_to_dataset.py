import os
from os.path import join
import pathlib

import torch
import numpy as np
from tqdm import tqdm

from deep_morpho.morp_operations import SequentialMorpOperations
from general.structuring_elements import *
from general.utils import get_next_same_name, load_json, save_json


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dataset_folder = "data/deep_morpho/dataset_0"



morp_operations_args = []

morp_operations_args.append({
    "name": "opening",
    "selems_fn": [disk, disk],
    "selems_args": [3, 3],
    "operations": ["erosion", "dilation"]
})
morp_operations_args.append({
    "name": "closing",
    "selems_fn": [disk, disk],
    "selems_args": [3, 3],
    "operations": ["dilation", 'erosion']
})

for operation in [hstick, vstick, scross, dcross, square]:
    morp_operations_args.append({
        "name": "opening",
        "selems_fn": [operation, operation],
        "selems_args": [7, 7],
        "operations": ["erosion", "dilation"]
    })
    morp_operations_args.append({
        "name": "closing",
        "selems_fn": [operation, operation],
        "selems_args": [7, 7],
        "operations": ["dilation", 'erosion']
    })


for args_idx, args in enumerate(morp_operations_args):
    print(f'Args nb {args_idx + 1} / {len(morp_operations_args)}')
    morp_operation = SequentialMorpOperations(
        name=args['name'],
        selems=[fn(arg) for fn, arg in zip(args['selems_fn'], args['selems_args'])],
        operations=args['operations'],
        device=device,
    )

    folder_op = get_next_same_name(join(dataset_folder, f"{morp_operation.name}"), 'seq', sep='_')
    pathlib.Path(join(folder_op, 'images')).mkdir(exist_ok=True, parents=True)
    pathlib.Path(join(folder_op, 'selems')).mkdir(exist_ok=True, parents=True)

    path_selem = []
    for selem_idx, selem in enumerate(morp_operation.selems):
        savepath = join(folder_op, 'selems', f'{selem_idx}.npy')
        np.save(savepath, selem.astype(np.uint8))
        path_selem.append(savepath)

    key_id = (
        '=>'.join(args['operations']) +
        " -- " +
        "=>".join([f'{fn.__name__}({arg})' for fn, arg in zip(args['selems_fn'], args['selems_args'])])
    )

    metadata = load_json(join(dataset_folder, 'metadata.json'))
    metadata['seqs'] = metadata.get('seqs', dict())
    metadata['seqs'][key_id] = {
        "path_target": join(folder_op, 'images'),
        "path_selems": path_selem
    }
    save_json(metadata, join(dataset_folder, 'metadata.json'))

    for img_file in tqdm(os.listdir(join(dataset_folder, 'images'))):
        img = np.load(join(dataset_folder, 'images', img_file))
        target = morp_operation(img)
        np.save(join(folder_op, "images", img_file), img.astype(np.int8))
