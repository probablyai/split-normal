# Split Normal Distribution aka Two-Piece Normal Distribution

A tiny package implementing functions of the split normal distribution compatible with [Numpy](https://github.com/numpy/numpy) and [JAX](https://github.com/google/jax).

## Install

```shell script
pip install split-normal
```

or

```shell script
conda install split-normal
```

## Usage

```python
import split_normal as sn

x = [-2.43953147, -1.31863936, -0.36272127, 0.77429312, 2.56092868]
p = [0.05, 0.25, 0.5, 0.75, 0.95]
loc = -1.
scale_1 = 1.
scale_2 = 2.

y_np = sn.numpy.pdf(x, loc, scale_1, scale_2)
print(y_np)
# [0.09437028 0.25279683 0.25279683 0.17943932 0.05450677]
p_np = sn.numpy.cdf(x, loc, scale_1, scale_2)
print(p_np)
# [0.05 0.25 0.5 0.75 0.95]
x_np = sn.numpy.ppf(p, loc, scale_1, scale_2)
print(x_np)
# [-2.43953147 -1.31863936 -0.36272127 0.77429312 2.56092868]

y_jax = sn.jax.pdf(x, loc, scale_1, scale_2)
print(y_jax)
# [0.09437027 0.2527968 0.2527968 0.17943932 0.05450677]
p_jax = sn.jax.cdf(x, loc, scale_1, scale_2)
print(p_jax)
# [0.04999999 0.25 0.5 0.75 0.95]
x_jax = sn.jax.ppf(p, loc, scale_1, scale_2)
print(x_jax)
# [-2.4395318 -1.3186394 -0.36272126 0.77429295 2.5609286]
```

## Equations

### PDF

Probability density function.

<p align="center"><img src="tex/e40bd5758ad08099e2a9805856a727ab.svg?invert_in_darkmode" align=middle width=347.04474255pt height=59.178683850000006pt/></p>

where <img src="tex/eddd50b8f927af24f6d449e758f03fd0.svg?invert_in_darkmode" align=middle width=163.06123019999998pt height=31.360807499999982pt/>.

### CDF

Cummulative density function.

<p align="center"><img src="tex/deb1a65aeacbbfaa6fce2f79904a298f.svg?invert_in_darkmode" align=middle width=334.0820736pt height=69.0417981pt/></p>

### PPF

Percent point function (also called inverse CDF or quantile function).

<p align="center"><img src="tex/3cc792438a7c9ddd285f79ab9e167ccf.svg?invert_in_darkmode" align=middle width=493.03176119999995pt height=59.178683850000006pt/></p>

## Literature

Wallis, Kenneth F. “The Two-Piece Normal, Binormal, or Double Gaussian Distribution: Its Origin and Rediscoveries.” Statistical Science, vol. 29, no. 1, 2014, pp. 106–112. JSTOR, www.jstor.org/stable/43288461.



