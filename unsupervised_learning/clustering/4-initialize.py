#!/usr/bin/env python3
""" module containing function that initializes variables for a
    Gaussian Mixture Model."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ function that initializes variables for a Gaussian Mixture Model.

    Parameters
    ----------
    X : numpy.ndarray
        Dataset with shape (n, d).
    k : int
        Number of clusters.

    Returns
    -------
    pi : numpy.ndarray or None
        Priors for each cluster with shape (k,).
        Returns None on failure.
    m : numpy.ndarray or None
        Centroid means for each cluster with shape (k, d).
        Returns None on failure.
    S : numpy.ndarray or None
        Covariance matrices for each cluster with shape (k, d, d).
        Returns None on failure.
    """

    # check X validity
    if not isinstance(X, type(np.array([]))) or len(X.shape) != 2:
        return None, None, None

    # check k validity
    if not isinstance(k, int) or k < 1:
        return None, None, None

    # get shapes
    n, d = X.shape

    # get centroids
    m, _ = kmeans(X, k)

    # create identity matrix and adjust dimensions
    s = np.identity(d)
    S = np.broadcast_to(s, (k, d, d))

    # get initial mixing coefficients
    prior = (100 / k) * 0.01
    pi = np.full((k,), prior)

    return pi, m, S
