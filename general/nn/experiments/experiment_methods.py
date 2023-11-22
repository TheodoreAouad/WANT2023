import inspect
from typing import Any, Dict, List, Optional
from abc import ABC
from enum import EnumMeta


class ExperimentMethods(ABC):
    # Enforce signature of `__init__` method not to have `*args` or `**kwargs` for smooth recursiveness
    # in `default_args` method.
    def __init__(self):
        super().__init__()

    @classmethod
    def select_(cls, name: str) -> Optional["cls"]:
        """
        Recursive class method iterating over all subclasses to return the
        desired model class.
        """
        if cls.__name__.lower() == name:
            return cls

        for subclass in cls.__subclasses__():
            selected = subclass.select_(name)
            if selected is not None:
                return selected

        return None

    @classmethod
    def select(cls, name: str, **kwargs: Any) -> "cls":
        """
        Class method iterating over all subclasses to instantiate the desired
        child.
        """
        name = name.lower()
        selected = cls.select_(name)
        if selected is None:
            err_msg = (
                "" f"The selected child {name} was not found for {cls}.\n" f"Available children are: {cls.listing()}" ""
            )
            raise ValueError(err_msg)

        return selected

    @classmethod
    def listing(cls) -> List[str]:
        """List all the available models."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__.lower()}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)

    @classmethod
    def listing_subclasses(cls) -> List[str]:
        """List all the available models."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing_subclasses())

        return list(subclasses)

    @classmethod
    def default_args(cls) -> Dict[str, dict]:
        """Return the default arguments of the model, in the format of argparse.ArgumentParser"""
        res = {}
        for name, p in inspect.signature(cls.__init__).parameters.items():
            if p.kind in [inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD]:
                for parent_class in cls.__bases__:
                    if parent_class in ExperimentMethods.listing_subclasses():
                        res.update({k: v for k, v in parent_class.default_args().items() if k not in res})

            elif name != "self":
                param_dict = {"default": p.default}

                if isinstance(p.default, (int, float, str)):
                    param_dict["type"] = cls.convert_or_str_fn(type(p.default))

                if isinstance(p.default, bool):
                    param_dict["type"] = cls.convert_to_bool_or_str_fn()

                if p.annotation is not None:
                    for type_ in [int, float, str]:
                        if (p.annotation == type_) or (p.annotation == Optional[type_]):
                            param_dict["type"] = cls.convert_or_str_fn(type_)

                    if (p.annotation == bool) or (p.annotation == Optional[bool]):
                        param_dict["type"] = cls.convert_to_bool_or_str_fn()

                    if isinstance(p.annotation, EnumMeta) or (p.annotation == Optional[EnumMeta]):
                        param_dict["type"] = cls.enum_to_str_fn(p.annotation)

                res[name] = param_dict

        return res

    @staticmethod
    def convert_or_str_fn(type_: type) -> Any:
        def convert_or_str(string: str) -> Any:
            """Convert a string to a type, or return the string."""
            try:
                return type_(string)
            except ValueError:
                return string

        return convert_or_str

    @staticmethod
    def convert_to_bool_or_str_fn() -> Any:
        def convert_to_bool(string: str) -> Any:
            """Convert a string to a type, or return the string."""
            try:
                if string.lower() in ["true", "t", "yes", "y", "1"]:
                    return True
                return False
            except ValueError:
                return string

        return convert_to_bool

    @staticmethod
    def enum_to_str_fn(enum: EnumMeta) -> str:
        """Returns a function that converts the enum to a string."""

        def enum_to_str(string) -> str:
            return enum[string]

        return enum_to_str
