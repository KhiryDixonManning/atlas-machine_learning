#!/usr/bin/env python3
""" module containing two functions, get_centroid gets the centroid that each
    data point belongs to and variance calculates the total intra-cluster
    variance for a dataset """
import numpy as np


def get_centroid(X, centroids):
    """ function that gets the index of the closest centroid for each
    data point.

    Parameters
    ----------
    X : numpy.ndarray
        Dataset with shape (n, d).
    centroids : numpy.ndarray
        Centroid means for each cluster with shape (k, d).

    Returns
    -------
    clss : numpy.ndarray
        Index of the closest centroid in centroids for each data point
        with shape (n,).
    """
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1)
    return np.argmin(distances, axis=-1), distances ** 2


def variance(X, C):
    """ function that calculate the total intra-cluster variance for a dataset

    Parameters
    ----------
    X : numpy.ndarray
        Dataset with shape (n, d).
    C : numpy.ndarray
        Centroid means for each cluster with shape (k, d).

    Returns
    -------
    var : float or None
        Total intra-cluster variance. Returns None on failure.
    """
    if not isinstance(X, type(np.array([]))) or len(
            X.shape) < 2 or isinstance(X[0][0], type(np.array([]))):
        return None

    if not isinstance(C, type(np.array([]))) or len(
            C.shape) < 2 or isinstance(C[0][0], type(
                np.array([]))) or X.shape[1] != C.shape[1]:
        return None

    # get labels with which centroids each point belongs to, along with the
    # squared distances
    labels, distances = get_centroid(X, C)

    # create empty matrix to store the squared distance from each datapoint
    # to its closest centroid in the corresponding index position
    masked = np.zeros_like(distances)

    # store the distance to the closest centroid in the index that
    # corresponds to that centroid. leave every other position in
    # that row blank
    masked[np.arange(len(X)), labels] = distances[np.arange(len(X)), labels]

    # add all of the squared distances together to get total
    # intracluster variance
    total_variance = np.sum(masked)

    return total_variance
