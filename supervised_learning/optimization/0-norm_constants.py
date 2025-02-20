#!/usr/bin/env python3
"""Module containing a function that calculates the normalization
   constants of a matrix."""

import numpy as np


def normalization_constants(X):
    """Calculates the normalization (standardization) constants of a matrix.

    Parameters:
        X (np.ndarray): A numpy array of shape (m, nx) to be normalized,
                        where m is the number of data points and nx is the
                        number of features.

    Returns:
        tuple: A tuple containing:
            - The mean of X along axis 0 (for each feature).
            - The standard deviation of X along axis 0 (for each feature).
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
