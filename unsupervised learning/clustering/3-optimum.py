#!/usr/bin/env python3
""" module containing function that tests for the optimum number of
    clusters by variance."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ function that tests for the optimum number of clusters by variance.

    Parameters
    ----------
    X : numpy.ndarray
        Dataset with shape (n, d).
    kmin : int, optional
        Minimum number of clusters to check for (inclusive). Default is 1.
    kmax : int or None, optional
        Maximum number of clusters to check for (inclusive). Default is None.
    iterations : int, optional
        Maximum number of iterations for K-means. Default is 1000.
    """

    # check valid X
    if not isinstance(X, type(np.array([]))) or len(
            X.shape) < 2 or isinstance(X[0][0], type(np.array([]))):
        return None, None

    # check validity of kmin and kmax
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is not None:
        if not isinstance(kmax, int) or kmax <= 0 or kmin >= kmax:
            return None, None

    # check validity of iterations
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]
    results, d_vars = [], []
    for i in range(kmin, kmax + 1):
        res, lab = kmeans(X, i, iterations)
        results.append((res, lab))
        var = variance(X, res)
        small_var = variance(X, results[0][0])
        d_vars.append(small_var - var)
    return results, d_vars
