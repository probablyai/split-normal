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

$$
  f\left(x ; \mu, \sigma_{1}, \sigma_{2}\right)=
  \begin{cases}
  A \exp \left(-\frac{(x-\mu)^{2}}{2 \sigma_{1}^{2}}\right) &
  \text{if } x \le \mu, \\
  A \exp \left(-\frac{(x-\mu)^{2}}{2 \sigma_{2}^{2}}\right) &
  \text{otherwise},
  \end{cases}
$$

where $A=\sqrt{2 / \pi}\left(\sigma_{1}+\sigma_{2}\right)^{-1}$.

### CDF

Cummulative density function.

$$
  F\left(x ; \mu, \sigma_{1}, \sigma_{2}\right) =
  \begin{cases}
  \frac{\sigma_{1} + \operatorname{erf}\left(\frac{x - \mu}{\sqrt{2} \sigma_{1}}\right) \sigma_{1}}{\sigma_{1} + \sigma_{2}} &
  \text{if } x \le \mu, \\
  \frac{\sigma_{1} + \operatorname{erf}\left(\frac{x - \mu}{\sqrt{2} \sigma_{2}}\right) \sigma_{2}}{\sigma_{1} + \sigma_{2}} &
  \text{otherwise}.
  \end{cases}
$$

### PPF

Percent point function (also called inverse CDF or quantile function).

$$
  F^{-1}\left(p ; \mu, \sigma_{1}, \sigma_{2}\right) =
  \begin{cases}
  \mu + \sqrt{2} \operatorname{erf}^{-1}\left(\frac{p (\sigma_{1} + \sigma_{2}) - \sigma_{1}}{\sigma_{1}}\right) \sigma_{1} &
  \text{if } p \le F\left(\mu ; \cdot\right), \\
  \mu + \sqrt{2} \operatorname{erf}^{-1}\left(\frac{p (\sigma_{1} + \sigma_{2}) - \sigma_{1}}{\sigma_{2}}\right) \sigma_{2} &
  \text{otherwise}.
  \end{cases}
$$

## Literature

Wallis, Kenneth F. “The Two-Piece Normal, Binormal, or Double Gaussian Distribution: Its Origin and Rediscoveries.” Statistical Science, vol. 29, no. 1, 2014, pp. 106–112. JSTOR, www.jstor.org/stable/43288461.



