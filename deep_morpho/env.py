CLASSIF_DATASETS_CHANNEL = ["classif_mnist_channel", "cifar10", "cifar100", ]
CLASSIF_DATASETS = CLASSIF_DATASETS_CHANNEL + ["classif_mnist"]

SPECIFIC_BIMONNS_DENSE = ["LightningBimonnDense", "LightningBimonnDenseNotBinary"]
SPECIFIC_BIMONNS_BISEL = ["LightningBimonnBiselDenseNotBinary"]
SPECIFIC_BIMONNS = SPECIFIC_BIMONNS_BISEL + SPECIFIC_BIMONNS_DENSE
