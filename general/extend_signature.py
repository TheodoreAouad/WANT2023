from types import FunctionType
import inspect
from functools import wraps


def extend_signature(fn):
    """ Decorator to extend the signature of a function, giving the possiblity of
    receiving with any keyword arguments.
    """
    sig = inspect.signature(fn)

    if inspect._ParameterKind.VAR_KEYWORD in [p.kind for p in sig.parameters.values()]:
        return fn


    if not isinstance(fn, FunctionType):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return fn(*args, **kwargs)
        return wrapped

        # wrapped.__signature__ = sig
    fn.__call__ = extend_signature(fn.__call__)
    return fn
