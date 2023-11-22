from typing import Callable


class Function:
    """Class to define a function, with the possibility to apply basic algebra with another function or constant."""
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __add__(self, other):
        if callable(other):
            def new_func(x):
                return self.func(x) + other(x)
        else:
            def new_func(x):
                return self.func(x) + other

        return Function(new_func)

    def __mul__(self, other):
        if callable(other):
            def new_func(*args, **kwargs):
                return self.func(*args, **kwargs) * other(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return self.func(*args, **kwargs) * other

        return Function(new_func)

    def __sub__(self, other):
        if callable(other):
            def new_func(*args, **kwargs):
                return self.func(*args, **kwargs) - other(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return self.func(*args, **kwargs) - other

        return Function(new_func)

    def __truediv__(self, other):
        if callable(other):
            def new_func(*args, **kwargs):
                return self.func(*args, **kwargs) / other(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return self.func(*args, **kwargs) / other

        return Function(new_func)

    def __pow__(self, other):
        if callable(other):
            def new_func(*args, **kwargs):
                return self.func(*args, **kwargs) ** other(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return self.func(*args, **kwargs) ** other

        return Function(new_func)
    
    def __radd__(self, other):
        if callable(other):
            def new_func(x):
                return other(x) + self.func(x)
        else:
            def new_func(x):
                return other + self.func(x)

        return Function(new_func)
    
    def __rmul__(self, other):
        if callable(other):
            def new_func(*args, **kwargs):
                return other(*args, **kwargs) * self.func(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return other * self.func(*args, **kwargs)

        return Function(new_func)

    def __rsub__(self, other):
        if callable(other):
            def new_func(*args, **kwargs):
                return other(*args, **kwargs) - self.func(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return other - self.func(*args, **kwargs)

        return Function(new_func)

    def __rtruediv__(self, other):
        if callable(other):
            def new_func(*args, **kwargs):
                return other(*args, **kwargs) / self.func(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return other / self.func(*args, **kwargs)

        return Function(new_func)

    def __rpow__(self, other):
        if callable(other):
            def new_func(*args, **kwargs):
                return other(*args, **kwargs) ** self.func(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return other ** self.func(*args, **kwargs)

        return Function(new_func)
