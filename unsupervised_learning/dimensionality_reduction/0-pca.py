#!/usr/bin/env python3
""" function performs Principal Component Analysis (PCA) on a dataset. """
import numpy as np


def pca(X, var=0.95):
    """ function performs Principal Component Analysis (PCA) on a dataset.

    Parameters:
    -----------
    X : numpy.ndarray
        Dataset with shape (n, d) where n is the number of data points and d
        is the number of dimensions.
        All dimensions have a mean of 0 across all data points.
    var : float, optional
        The fraction of the variance that the PCA transformation
        should maintain. Default is 0.95.

    Returns:
    --------
    W : numpy.ndarray
        The weights matrix that maintains var fraction of X's original
        variance.
        It has shape (d, nd) where nd is the new dimensionality of the
        transformed X.
    """
    standardized = X

    cov = np.cov(standardized, rowvar=False)

    o_values, o_vectors = np.linalg.eig(cov)
    values = np.copy(o_values)
    vectors = np.copy(o_vectors)

    sort = np.argsort(values)[::-1]
    s_values = values[sort]
    s_vectors = vectors[:, sort]

    v_ex = []
    for i in s_values:
        v_ex.append((i / sum(s_values)))

    total = np.sum(s_values)

    test = np.cumsum(s_values) / total

    r_cumsum = np.argmax(test >= var) + 2

    selected = s_vectors[:, :r_cumsum]

    selected[:, 1] *= -1

    return selected
