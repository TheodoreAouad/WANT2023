from copy import deepcopy
from typing import Union, Callable, Dict, List, Any, Tuple

import numpy as np

from .parser import Parser, GridParser
from general.function import Function


class UniformSearch(Function):
    def __init__(self, lb: float, ub: float):
        self.lb = lb
        self.ub = ub

    def func(self, *args, **kwargs):
        return np.random.uniform(self.lb, self.ub)


class DefaultSelector:
    """ Default selector for the random search.

    Args:
        selection (Dict[str, Tuple[List[Any], List[float]]]): Dictionary of the selection. The key is the name of
        the argument, the value is lb tuple of two lists. The first list is the list of the values of the argument,
        the second list is the list of the probabilities of the values.

    Example:
        >>> selection = {
        >>>     "lr": lambda x: 10 ** (np.random.uniform(-6, -1)),
        >>>     "batch_size": ([32, 64, 128], [0.5, 0.3, 0.2]),
        >>>     "optimizer": ([torch.optim.Adam, torch.optim.SGD], [0.5, 0.5]),
        >>> }
    """
    def __init__(self, selection: Dict[
        str,  # Name of the argument
        Union[
            Callable,  # function to generate the output
            List[Any],  # List of values. Equiprobability between the values.
            Tuple[List[Any], List[float]],  # List of values and list of probabilities.
        ],
    ]):
        self.selection = selection



class RandomParser(GridParser):
    """
    Args:
        selection (Dict[str, Tuple[List[Any], List[float]]]): Dictionary of the selection. The key is the name of
        the argument, the value is lb tuple of two lists. The first list is the list of the values of the argument,
        the second list is the list of the probabilities of the values.

    Example:
        >>> selection = {
        >>>     "lr": lambda x: 10 ** (np.random.uniform(-6, -1)),
        >>>     "batch_size": ([32, 64, 128], [0.5, 0.3, 0.2]),
        >>>     "optimizer": ([torch.optim.Adam, torch.optim.SGD], [0.5, 0.5]),
        >>> }
    """
    def __init__(
        self,
        n_tries: int,
        selector: Dict[
            str,  # Name of the argument
            Union[
                Callable,  # function to generate the output
                List[Any],  # List of values. Equiprobability between the values.
                Tuple[List[Any], List[float]],  # List of values and list of probabilities.
            ],
        ] = {},
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.selector = selector
        self.n_tries = n_tries

    def __len__(self):
        return self.n_tries

    def get_args(self, arg_idx: int = None) -> Parser:
        args = {}
        for key in self.keys():
            args[key] = self.get_value(key)
        args = deepcopy(args)
        return Parser(args).parse_args()

    def parse_args(self, args=None, namespace=None, add_argv: bool = True) -> "Parser":
        for key in self.keys():
            if key not in self.selector:
                self.selector[key] = self[key]  # Default selection: random choice from the list of values.
        return self

    def get_value(self, arg_name: str) -> Any:
        """ Return the value of the argument.

        Args:
            arg_name (str): Name of the argument.

        Returns:
            Any: Value of the argument.
        """
        if arg_name not in self.selector:
            return self[arg_name][np.random.choice(range(len(self[arg_name])))]
            # return np.random.choice(self[arg_name])

        arg = self.selector[arg_name]

        if isinstance(arg, Callable):
            return arg()

        if isinstance(arg, list):
            return arg[np.random.choice(range(len(arg)))]

        if isinstance(arg, tuple):
            return arg[0][np.random.choice(range(len(arg[0])), p=arg[1])]

        raise ValueError(f"Unknown type for {arg_name}.")
