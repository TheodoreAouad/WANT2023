from typing import List, Callable, Union

import numpy as np

from general.nn.observables import Observable
from general.utils import save_json, log_console

from .batch_early_stopping import EarlyStoppingBase



class CombineEarlyStopping(EarlyStoppingBase):

    def __init__(
        self,
        early_stoppers: List[EarlyStoppingBase],
        decision_rule: Union[str, Callable] = "and",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.early_stoppers = early_stoppers
        self.decision_rule_str = f"{decision_rule}"
        self.decision_rule = self.init_decision_rule(decision_rule)

        self.reason_codes = []


    def init_decision_rule(self, decision_rule: Union[str, Callable]) -> Callable:
        if decision_rule == "and":
            return np.all

        if decision_rule == "or":
            return np.any

        return decision_rule

    def run_early_stopping_check(self, trainer, pl_module):
        should_stops = []
        reason_strs = []
        reason_codes = []

        for early_stopper in self.early_stoppers:
            should_stop, reason_str, reason_code = early_stopper.run_early_stopping_check(trainer, pl_module)
            should_stops.append(should_stop)
            reason_strs.append(reason_str)
            reason_codes.append(reason_code)

        should_stop = self.decision_rule(should_stops)
        reason_str = f"{self.decision_rule_str}:\n" + "\n".join([f"{reason}: {stop}" for reason, stop in zip(reason_strs, should_stops)])
        return should_stop, reason_str, reason_codes

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs) -> None:
        return self.perform_early_stopping(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:
        return self.perform_early_stopping(trainer, pl_module)
