import numpy
import scipy.special

from ._base import map_as_float_array, convert_negative_to_nan

__all__ = [
    'pdf',
    'cdf',
    'ppf'
]

_SQRT_2 = numpy.sqrt(2)
_SQRT_2_DIV_PI = numpy.sqrt(2 / numpy.pi)


def pdf(x, loc, scale_1, scale_2):
    """
    Probability density function.

    Parameters
    ----------
    x : array_like
        The sample at which the PDF is evaluated.
    loc : array_like
        Mode.
    scale_1 : array_like
        Left-hand-side standard deviation (square root of variance).
    scale_2 : array_like
        Right-hand-side standard deviation (square root of variance).

    Returns
    -------
    array_like
        The value of PDF for `x`.
    """
    x, loc, scale_1, scale_2 = map_as_float_array(x, loc, scale_1, scale_2)
    scale_1, scale_2 = convert_negative_to_nan(scale_1, scale_2)
    scale = numpy.where(x <= loc, scale_1, scale_2)
    a = _SQRT_2_DIV_PI / (scale_1 + scale_2)
    return a * numpy.exp(- (x - loc) ** 2 / (2.0 * scale ** 2))


def cdf(x, loc, scale_1, scale_2):
    """
    Cummulative density function.

    Parameters
    ----------
    x : array_like
        The sample at which the CDF is evaluated.
    loc : array_like
        Mode.
    scale_1 : array_like
        Left-hand-side standard deviation (square root of variance).
    scale_2 : array_like
        Right-hand-side standard deviation (square root of variance).

    Returns
    -------
    array_like
        The probability that a random variable will take a value less than or equal to `x`.
    """
    x, loc, scale_1, scale_2 = map_as_float_array(x, loc, scale_1, scale_2)
    scale_1, scale_2 = convert_negative_to_nan(scale_1, scale_2)
    scale = numpy.where(x <= loc, scale_1, scale_2)
    return (scale_1 + scipy.special.erf((x - loc) / (_SQRT_2 * scale)) * scale) / (scale_1 + scale_2)


def ppf(p, loc, scale_1, scale_2):
    """
    Percent point function (also called inverse CDF or quantile function).

    Parameters
    ----------
    p : array_like
        The cummulative probability.
    loc : array_like
        Mode.
    scale_1 : array_like
        Left-hand-side standard deviation (square root of variance).
    scale_2 : array_like
        Right-hand-side standard deviation (square root of variance).

    Returns
    -------
    array_like
        The `p`-quantile, the value such that a random variable will be less
        than or equal to that value with probability `p`.
    """
    p, loc, scale_1, scale_2 = map_as_float_array(p, loc, scale_1, scale_2)
    scale_1, scale_2 = convert_negative_to_nan(scale_1, scale_2)
    scale = numpy.where(p <= cdf(loc, loc, scale_1, scale_2), scale_1, scale_2)
    a = (p * (scale_1 + scale_2) - scale_1)
    return loc + _SQRT_2 * scale * scipy.special.erfinv(a / scale)
