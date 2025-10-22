#!/usr/bin/env python3
"""Module containing a function that shuffles the data points in two matrices
   the same way."""

import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way.

    Parameters:
        X (np.ndarray): A numpy array of shape (m, nx) to be shuffled,
                         where m is the number of data points and nx is the
                         number of features in X.
        Y (np.ndarray): A numpy array of shape (m, ny) to be shuffled,
                         where m is the same number of data points as in X
                         and ny is the number of features in Y.

    Returns:
        tuple: A tuple containing the shuffled X and Y arrays.
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
