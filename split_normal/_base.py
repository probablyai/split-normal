import numpy as np

__all__ = [
    'map_as_float_array',
    'convert_negative_to_nan'
]


def map_as_float_array(*args):
    return map(np.asfarray, args)


def convert_negative_to_nan(*args):
    return map(lambda x: np.where(np.less(x, 0.), np.nan, x), args)
