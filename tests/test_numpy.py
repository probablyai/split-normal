import scipy as sp
import numpy as np
import numpy.testing as npt

import split_normal as sn


def test_pdf():
    x = np.linspace(-5., 5., 40)
    params_split_norm = dict(
        loc=0,
        scale_1=1,
        scale_2=1
    )
    params_norm = dict(
        loc=0,
        scale=1
    )
    actual = sn.numpy.pdf(x, **params_split_norm)
    expected = sp.stats.norm.pdf(x, **params_norm)
    npt.assert_almost_equal(actual, expected)

    x = [
        -2.,
        -0.5,
        0.,
        1.,
        2.,
        3.,
        4.
    ]
    params_split_norm = dict(
        loc=1,
        scale_1=1,
        scale_2=2
    )
    actual = sn.numpy.pdf(x, **params_split_norm)
    expected = [
        0.00295457,
        0.08634506,
        0.16131382,
        0.26596152,
        0.23471022,
        0.16131382,
        0.08634506
    ]
    npt.assert_almost_equal(actual, expected)


def test_cdf():
    x = [-1.96, 1.96]
    actual = sn.numpy.cdf(x, 0, 1, 1)
    expected = [0.025, 0.975]
    npt.assert_almost_equal(actual, expected, decimal=2)

    expected = [-2.43953147, 2.43953147]
    params = dict(
        loc=[-1, 1],
        scale_1=[1, 2],
        scale_2=[2, 1]
    )
    p = sn.numpy.cdf(expected, **params)
    actual = sn.numpy.ppf(p, **params)
    npt.assert_almost_equal(actual, expected, decimal=5)


def test_ppf():
    p = [0.025, 0.975]
    actual = sn.numpy.ppf(p, 0, 1, 1)
    expected = [-1.96, 1.96]
    npt.assert_almost_equal(actual, expected, decimal=2)

    expected = [0.05, 0.95]
    params = dict(
        loc=[-1, 1],
        scale_1=[1, 2],
        scale_2=[2, 1]
    )
    x = sn.numpy.ppf(expected, **params)
    actual = sn.numpy.cdf(x, **params)
    npt.assert_almost_equal(actual, expected)


def test_invalid_params():
    """
    If it make sense, invalid input values are handled similarly as in Numpy or SciPy.
    """
    x = [1, 1, 1, 1, 1, 1, 1]
    params_split_norm = dict(
        loc=1,
        scale_1=[1, -1, 1, -1, None, 1, None],
        scale_2=[2, 2, -2, -2, 2, None, None]
    )

    expected = [0.26596152, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    actual = sn.numpy.pdf(x, **params_split_norm)
    npt.assert_almost_equal(actual, expected)

    expected = [0.3333333, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    actual = sn.numpy.cdf(x, **params_split_norm)
    npt.assert_almost_equal(actual, expected)

    p = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0., 1., -2, 2]
    params_split_norm = dict(
        loc=1,
        scale_1=[1, -1, 1, -1, None, 1, None, 1, 1, 1, 1],
        scale_2=[2, 2, -2, -2, 2, None, None, 2, 2, 2, 2]
    )
    expected = [1.6372787, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -np.inf, np.inf, np.nan, np.nan]
    actual = sn.numpy.ppf(p, **params_split_norm)
    npt.assert_almost_equal(actual, expected)
