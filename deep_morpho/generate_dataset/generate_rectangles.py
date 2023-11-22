import pathlib
from os.path import join

import numpy as np
from tqdm import tqdm

from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect
from general.utils import get_next_same_name, save_json, load_json

##### TO CHANGE #####
generator_args = {
    'size': (50, 50), 'n_shapes': 30, 'max_shape': (15, 15), 'p_invert': 0.5, 'n_holes': 15, 'max_shape_holes': (7, 7)
}
generator_fn = get_random_rotated_diskorect
n_rectangles = 100000
dataset_idx = None  # choose either an existing dataset idx, or create a new one with "None"
######################

path_data = "data/deep_morpho/"
name_folder = "dataset"

metadata = {
    "generator_args": generator_args,
    "fn_name": generator_fn.__name__,
}

if dataset_idx is not None:
    dest_path = join(path_data, f'{name_folder}_{dataset_idx}')
    metadata = load_json(join(dest_path, 'metadata.json'))
    assert metadata["fn_name"] == generator_fn.__name__, "wrong generator given"
    generator_args = metadata['generator_args']
else:
    dest_path = get_next_same_name(path_data, name_folder, sep="_")
    pathlib.Path(join(dest_path, 'images')).mkdir(exist_ok=True, parents=True)
    save_json(metadata, join(dest_path, 'metadata.json'))

for idx in tqdm(range(n_rectangles)):
    rect = get_random_rotated_diskorect(**generator_args)
    np.save(join(dest_path, "images", f"{idx}.npy"), rect.astype(np.uint8))


