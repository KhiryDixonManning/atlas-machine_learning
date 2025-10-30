#!/usr/bin/env python3
""" module containing function that perform K-means clustering on a
dataset using sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """ function that perform K-means clustering on a dataset using sklearn

    Parameters
    ----------
    X : numpy.ndarray
        Dataset with shape (n, d).
    k : int
        Number of clusters.

    Returns
    -------
    C : numpy.ndarray
        Centroid means for each cluster with shape (k, d).
    clss : numpy.ndarray
        Index of the cluster in C that each data point belongs to
        with shape (n,). """
    k_means = sklearn.cluster.KMeans(k).fit(X)
    return k_means.cluster_centers_, k_means.labels_
