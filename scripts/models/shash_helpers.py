# This code is taken from https://arxiv.org/pdf/2109.07250.pdf
# Authors: "Randal J. Barnes and Elizabeth A. Barnes"
# date: "10 September 2021"


"""sinh-arcsinh normal distribution helper functions.
    Functions--------
    mean(mu, sigma, gamma, tau)
    distribution mean.
    median(mu, sigma, gamma, tau)
    distribution median.
    stddev(mu, sigma, gamma, tau)
    distribution standard deviation.
    variance(mu, sigma, gamma, tau)
    distribution variance.
    Notes----
    * The sinh-arcsinh normal distribution was defined in [1]. A more accessible
    presentation is given in [2].
    * The notation and formulation used in this code was taken from [3], page 143.
    In the gamlss.dist/CRAN package the distribution is called SHASHo.
    * There is a typographical error in the presentation of the probability
    density function on page 143 of [3]. There is an extra "2" in the denomenator
    preceeding the "sqrt{1 + z^2}" term.
    References---------
    [1] Jones, M. C. & Pewsey, A., Sinh-arcsinh distributions,
    Biometrika, Oxford University Press, 2009, 96, 761-780.
    DOI: 10.1093/biomet/asp053.
    [2] Jones, C. & Pewsey, A., The sinh-arcsinh normal distribution,
    Significance, Wiley, 2019, 16, 6-7.
    DOI: 10.1111/j.1740-9713.2019.01245.x.
    https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1740-9713.2019.01245.x
    [3] Stasinopoulos, Mikis, et al. (2021), Distributions for Generalized
    Additive Models for Location Scale and Shape, CRAN Package.
    https://cran.r-project.org/web/packages/gamlss.dist/gamlss.dist.pdf
 """

import scipy
import numpy as np


def _jones_pewsey_P(q):
    """P_q function from page 764 of [1].
    This is a module private helper function. This function will not be
    called externally.
    Arguments
    --------
    q : float or double, array like
    Returns
    ------
    P_q : array like of same shape as q.
    Notes
    ----
    * The strange constant 0.25612... is "sqrt( sqrt(e) / (8*pi) )" computed
    with a high-precision calculator.
    """
    return 0.25612601391340369863537463 * (scipy.special.kv((q + 1) / 2, 0.25) + scipy.special.kv((q - 1) / 2, 0.25))


def mean(mu, sigma, gamma, tau):
    """The distribution mean.
    Arguments
    --------
    mu : float or double (batch size x 1) Tensor
    The location parameter.
    sigma : float or double (batch size x 1) Tensor
    The scale parameter. Must be strictly positive. Must be the same shape
    and dtype as mu.
    gamma : float or double (batch size x 1) Tensor
    The skewness parameter. Must be the same shape and dtype as mu.
    tau : float or double (batch size x 1) Tensor
    The tail-weight parameter. Must be strictly positive. Must be the same
    shape and dtype as mu.
    Returns
    ------
    x : float or double (batch size x 1) Tensor.
    The computed distribution mean values.
    """
    evX = np.sinh(gamma / tau) * _jones_pewsey_P(1.0 / tau)
    if evX > 1e12:
        print("Warning: evX is very large. This may cause numerical instability.")
        print((1.0/tau+1)/2, (1.0/tau-1)/2, 
              scipy.special.kv((1.0/tau + 1) / 2, 0.25), scipy.special.kv((1.0/tau - 1) / 2, 0.25))
    return mu + sigma * evX



def stddev(mu, sigma, gamma, tau):
    """The distribution standard deviation.
    Arguments
    --------
    mu : float or double (batch size x 1) Tensor
    The location parameter.
    sigma : float or double (batch size x 1) Tensor
    The scale parameter. Must be strictly positive. Must be the same shape
    and dtype as mu.
    gamma : float or double (batch size x 1) Tensor
    The skewness parameter. Must be the same shape and dtype as mu.
    tau : float or double (batch size x 1) Tensor
    The tail-weight parameter. Must be strictly positive. Must be the same
    shape and dtype as mu.
    Returns
    ------
    x : float or double (batch size x 1) Tensor.
    The computed distribution mean values.
    """
    return np.sqrt(variance(mu, sigma, gamma, tau))


def variance(mu, sigma, gamma, tau):
    """The distribution variance.
    Arguments
    --------
    mu : float or double (batch size x 1) Tensor
    The location parameter.
    sigma : float or double (batch size x 1) Tensor
    The scale parameter. Must be strictly positive. Must be the same shape
    and dtype as mu.
    gamma : float or double (batch size x 1) Tensor
    The skewness parameter. Must be the same shape and dtype as mu.
    tau : float or double (batch size x 1) Tensor
    The tail-weight parameter. Must be strictly positive. Must be the same
    shape and dtype as mu.
    Returns
    ------
    x : float or double (batch size x 1) Tensor.
    The computed distribution mean values.
    Notes----
    * This code uses two basic formulas:
    var(X) = E(X^2)- (E(X))^2
    var(a*X + b) = a^2 * var(X)
    * The E(X) and E(X^2) are computed using the moment equations given on
    page 764 of [1].
    """
    evX = np.sinh(gamma / tau) * _jones_pewsey_P(1.0 / tau)
    evX2 = (np.cosh(2 * gamma / tau) * _jones_pewsey_P(2.0 / tau)- 1.0) / 2
    return np.square(sigma) * (evX2 - np.square(evX))
