import sys
import numpy as np

__all__ = [
    'whoami',
    'check_array_like',
    'map_as_float_array',
    'convert_negative_to_nan'
]

from jax.numpy import ndarray, isscalar


def whoami():
    return sys._getframe(1).f_code.co_name


def is_array_like(x):
    return isinstance(x, ndarray) or isscalar(x)


def check_array_like(*args, func_name=None):
    arg_pos, arg_type = next(
        ((i, type(arg)) for i, arg in enumerate(args) if not is_array_like(arg)),
        (None, None)  # default
    )
    if arg_pos or arg_type:
        msg = f"Function{f' `{func_name}()` ' if func_name else ' '}" \
              f"requires `ndarray` or scalar arguments but got `{arg_type}` at position {arg_pos}."
        raise TypeError(msg)


def map_as_float_array(*args):
    return map(np.asfarray, args)


def convert_negative_to_nan(*args):
    return map(lambda x: np.where(np.less(x, 0.), np.nan, x), args)
