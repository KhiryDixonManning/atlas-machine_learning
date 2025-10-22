#!/usr/bin/env python3
""" module containing function that perform k-means along with a helper
fucntion that finds which centroid the datapoints belong to """
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
    return np.argmin(distances, axis=-1)


def kmeans(X, k, iterations=1000):
    """ function perform K-means clustering on a dataset.

    Parameters
    ----------
    X : numpy.ndarray
        Dataset with shape (n, d).
    k : int
        Number of clusters.
    iterations : int, optional
        Maximum number of iterations that should be performed. Default is 1000

    Returns
    -------
    C : numpy.ndarray or None
        Centroid means for each cluster with shape (k, d).
        Returns None on failure.
    clss : numpy.ndarray or None
        Index of the cluster in C that each data point belongs to with
        shape (n,). Returns None on failure."""

    if not isinstance(X, type(np.array([]))) or len(
            X.shape) < 2 or isinstance(X[0][0], type(np.array([]))):
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    d = X.shape[1]

    min_v = np.min(X, axis=0)
    max_v = np.max(X, axis=0)

    centroids = np.random.uniform(low=min_v, high=max_v, size=(k, d))
    for x in range(iterations):
        labels = get_centroid(X, centroids)

        new_centroids = np.zeros((k, d))

        for i in range(k):
            if np.sum(labels == i) <= 0:
                new_centroids[i] = np.random.uniform(
                    low=min_v, high=max_v, size=d)
            else:
                new_centroids[i] = np.mean(X[labels == i], axis=0)

        if np.allclose(new_centroids, centroids):
            labels = get_centroid(X, centroids)
            return centroids, labels

        centroids = new_centroids

    labels = get_centroid(X, centroids)

    return centroids, labels
