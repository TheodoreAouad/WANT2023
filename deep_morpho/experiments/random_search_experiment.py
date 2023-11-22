import random
from .multi_experiment import MultiExperiment
from .parser import Parser


class RandomSearchExperiment(MultiExperiment):
    """ Args are taken randomly from the list of args
    """
    def __init__(self, n_tries: int, selector: DefaultSelector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_experiments = n_tries

    def setup_args(self, arg_idx: int) -> Parser:
        return random.choice(self.multi_args.multi_args)
