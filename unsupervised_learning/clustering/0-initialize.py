#!/usr/bin/env python3
""" module containing function that initialize cluster centroids for
K-means clustering. """
import numpy as np


def initialize(X, k):
    """ function that initialize cluster centroids for K-means clustering.

    Parameters
    ----------
    X : numpy.ndarray
        Dataset with shape (n, d).
    k : int
        Number of clusters.

    Returns
    -------
    centroids : numpy.ndarray or None
        Initialized centroids for each cluster with shape (k, d).
        Returns None on failure."""

    if not isinstance(X, type(np.array([]))) or len(X.shape) < 2 or isinstance(
            X[0][0], type(np.array([]))):
        return None

    if type(k) is not int or k <= 0:
        return None

    d = X.shape[1]

    min_v = np.min(X, axis=0)
    max_v = np.max(X, axis=0)

    return np.random.uniform(low=min_v, high=max_v, size=(k, d))
