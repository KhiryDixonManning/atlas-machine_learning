#!/usr/bin/env python3
""" module containing funtion that calculates GMM of dataset using sklearn """
import sklearn.mixture


def gmm(X, k):
    """ function that calculate a Gaussian Mixture Model (GMM) from a dataset.

    Parameters
    ----------
    X : numpy.ndarray
        Dataset with shape (n, d).
    k : int
        Number of clusters.

    Returns
    -------
    pi : numpy.ndarray
        Cluster priors with shape (k,).
    m : numpy.ndarray
        Centroid means with shape (k, d).
    S : numpy.ndarray
        Covariance matrices with shape (k, d, d).
    clss : numpy.ndarray
        Cluster indices for each data point with shape (n,).
    bic : numpy.ndarray
        Bayesian Information Criterion (BIC) values for each cluster size
        tested.
        """
    g_mm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = g_mm.weights_
    m = g_mm.means_
    S = g_mm.covariances_
    clss = g_mm.predict(X)
    bic = g_mm.bic(X)
    return pi, m, S, clss, bic
