# A foundation for exact binarized morphological neural networks

This  repository is associated with the paper "A foundation for exact binarized morphological neural networks" written by Theodore Aouad and Hugues Talbot, accepted in the 2023 NeurIPS workshop on Advancing Neural Network Training ([WANT](https://want-ai-hpc.github.io/)) .

## Requirements

We use python 3.8.16. The requirements are listed in `requirements.txt`. To install them, run:

```bash
pip install -r requirements.txt
```

Write down the path to the dataset in `deep_morpho/datasets/root_mnist_dir.txt`. By default, the path will be the project directory. If the MNIST dataset is not present, it will be downloaded.


## Training

To train the models, the commands are:

### Morphological experiment on noised sticks datset

```bash
python deep_morpho/train.py --args deep_morpho/saved_args/noisti/args_noisti.py
```

### Classification on MNIST dataset

For each configuration, please see the associated arguments file for more details on the hyperparameters searching. Usually, 100 runs will be launched on random search.

- No regularization

```bash
python deep_morpho/train.py --args deep_morpho/saved_args/mnist/args_no_regu.py 
```

- Regularization of exact projection onto constant set

```bash
python deep_morpho/train.py --args deep_morpho/saved_args/mnist/args_regu_constant_exact.py
```

- Regularization of approximate projection onto constant set

```bash
python deep_morpho/train.py --args deep_morpho/saved_args/mnist/args_regu_constant_approx.py
```

- Regularization of approximate projection onto constant set with dual reparametrization for the weights

```bash
python deep_morpho/train.py --args args_regu_constant_approx_dual_weights
```

## Results

The tensorboards will be saved inside the `results` folder. Example:

```bash
tensorboard --logdir deep_morpho/results/results_tensorboards/regu_constant_approx/0/mnistclassifchannel/BimonnDenseNotBinary/version_0
```