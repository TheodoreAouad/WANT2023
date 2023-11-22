from ..extend_signature import extend_signature


def extend_signature_and_forward(fn):
    """ Decorator to extend the signature of a function, giving the possiblity of
    receiving with any keyword arguments.
    """
    fn = extend_signature(fn)

    if hasattr(fn, "forward"):
        fn.forward = extend_signature(fn.forward)

    return fn
