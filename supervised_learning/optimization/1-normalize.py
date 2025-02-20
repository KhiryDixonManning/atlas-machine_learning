#!/usr/bin/env python3
"""Module containing a function that normalizes a matrix."""

import numpy as np


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix.

    Parameters:
        X (np.ndarray): A numpy array of shape (m, nx) to be normalized,
                         where m is the number of data points and nx is the
                         number of features.
        m (np.ndarray): A numpy array of shape (nx, ) containing the mean
                        of each feature in X.
        s (np.ndarray): A numpy array of shape (nx, ) containing the standard
                        deviation of each feature in X.

    Returns:
        np.ndarray: A numpy array of the normalized matrix.
    """
    return (X - m) / s
