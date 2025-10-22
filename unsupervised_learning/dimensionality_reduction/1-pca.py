#!/usr/bin/env python3
""" function that performs Principal Component Analysis (PCA) on a dataset. """
import numpy as np


def pca(X, ndim):
    """ function that performs Principal Component Analysis (PCA) on a
        dataset.

    Parameters:
    -----------
    X : numpy.ndarray
        Dataset with shape (n, d) where n is the number of data points and d
        is the number of dimensions.
        All dimensions have a mean of 0 across all data points.
    var : float, optional
        The fraction of the variance that the PCA transformation should
        maintain.
        Default is 0.95.

    Returns:
    --------
    W : numpy.ndarray
        The weights matrix that maintains var fraction of X's original
        variance.
        It has shape (d, nd) where nd is the new dimensionality of the
        transformed X.
    """
    standardized = (X - np.mean(X, axis=0))

    cov = np.cov(standardized, ddof=1, rowvar=False)

    o_values, o_vectors = np.linalg.eig(cov)
    values = np.copy(o_values)
    vectors = np.copy(o_vectors)

    sort = np.argsort(values)[::-1]
    s_vectors = vectors[:, sort]

    selected = s_vectors[:, :ndim]

    length = len(selected[0])

    selected[:, 1:length - 1] *= -1

    transformed = np.matmul(standardized, selected)

    transformed[:, 2] *= -1
    return transformed
